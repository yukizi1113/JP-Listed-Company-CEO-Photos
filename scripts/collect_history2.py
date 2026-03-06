#!/usr/bin/env python3
"""
collect_history2.py  –  歴代社長情報の再収集（改良版）

修正点:
  - Wikipedia Strategy 2 でもページタイトル検証を追加（汚染防止）
  - 会社名との一致スコアを強化（2文字ではなく主要語句で判定）
  - 追加ソース: 会社IRページの沿革 + Bing検索「会社名 歴代社長」
  - 50社ごとにコミット&プッシュ&ローカル削除

対象:
  - data/recollect_targets.json に含まれる会社（汚染除去後に空になった会社）
  - data/history_supplement.json に未登録の会社
"""

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

# ── パス設定 ──────────────────────────────────────────────────────────
PROJECT_DIR      = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR         = PROJECT_DIR / "data"
CEO_DATA_FILE    = DATA_DIR / "ceo_data.json"
HISTORY_OUT      = DATA_DIR / "history_supplement.json"
LOG_FILE         = PROJECT_DIR / "collect_history2.log"

GH_TOKEN  = os.environ.get("GH_TOKEN", "")
GH_USER   = "yukizi1113"
REPO_NAME = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = (
    f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"
    if GH_TOKEN else f"https://github.com/{GH_USER}/{REPO_NAME}.git"
)

# ── ログ設定 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── photos_dir ────────────────────────────────────────────────────────
def photos_dir(ticker: str) -> Path:
    try:
        t = int(ticker)
        if t < 3500: return PROJECT_DIR / "photos_1"
        if t < 5000: return PROJECT_DIR / "photos_2"
        if t < 7000: return PROJECT_DIR / "photos_3"
        if t < 9000: return PROJECT_DIR / "photos_4"
        return PROJECT_DIR / "photos_5"
    except ValueError:
        return PROJECT_DIR / "photos_5"


# ── 名前バリデーション ────────────────────────────────────────────────
_NAME_BAD = [
    "代表", "取締役", "社長", "会社", "株式", "在任", "就任", "年月", "氏名",
    "単独", "創業", "受賞", "評議", "主筆", "締役", "東経", "年度", "新任",
    "実兄", "以上", "合計", "出典", "参照", "注記", "情報", "事業", "採用",
    "光章", "吾氏", "役員", "経歴", "部長", "常務", "専務", "議長", "理事",
    "その他", "期間", "備考", "名前", "読み", "から", "まで", "年間",
    "会長", "業務", "表明", "検査", "一覧", "管理", "担当", "運営",
    "設立", "退任", "現在", "歴代", "前任", "後任", "交代", "変更",
    "合併", "分割", "解散", "破綻", "再建", "刷新", "監査", "執行",
    "現職",
]

_JP_CHARS = re.compile(r'^[\u4e00-\u9fff\u3040-\u30ff]+$')


def _is_valid_jp_name(text: str) -> bool:
    text = text.strip()
    if not text or "\n" in text or "\r" in text:
        return False
    if any(b in text for b in _NAME_BAD):
        return False
    if re.search(r'[\s\u3000]', text):
        parts = re.split(r'[\s\u3000]+', text.strip())
        if len(parts) != 2:
            return False
        family, given = parts
        # 姓: 2-5文字, 名: 1-5文字
        if not (2 <= len(family) <= 5 and 1 <= len(given) <= 5):
            return False
        return bool(_JP_CHARS.match(family) and _JP_CHARS.match(given))
    if not (2 <= len(text) <= 7):
        return False
    return bool(_JP_CHARS.match(text))


# ── 日付パーサ ────────────────────────────────────────────────────────
_DATE_PATS = [
    (r"(\d{4})年(\d{1,2})月(\d{1,2})日", True),
    (r"(\d{4})-(\d{2})-(\d{2})",         True),
    (r"(\d{4})/(\d{1,2})/(\d{1,2})",     True),
    (r"(\d{4})年(\d{1,2})月",             False),
]


def parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    for pat, has_day in _DATE_PATS:
        m = re.search(pat, s)
        if m:
            g = m.groups()
            y, mo = int(g[0]), int(g[1])
            d = int(g[2]) if has_day else 1
            try:
                return datetime(y, mo, d)
            except ValueError:
                pass
    return None


def _extract_dates(text: str) -> tuple[str | None, str | None]:
    dates = []
    for pat, has_day in _DATE_PATS:
        for m in re.finditer(pat, text):
            g = m.groups()
            y, mo = int(g[0]), int(g[1])
            d = int(g[2]) if has_day else 1
            try:
                dates.append(datetime(y, mo, d).strftime("%Y-%m-%d"))
            except ValueError:
                pass
    dates = sorted(set(dates))
    appt = dates[0]  if dates else None
    rsgn = dates[-1] if len(dates) > 1 else None
    return appt, rsgn


# ── セッション設定 ────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ja-JP,ja;q=0.9",
})


# ── ページタイトル検証（汚染防止） ────────────────────────────────────
def _page_matches_company(sp: BeautifulSoup, company_name: str) -> bool:
    """
    WikipediaページのH1タイトルまたはページ本文が対象会社と関連するか確認。
    社名変更した会社（例: マルハニチロ→Umios）にも対応。
    """
    h1 = sp.find("h1", {"id": "firstHeading"})
    if not h1:
        return False
    page_title = h1.get_text().strip()

    # 正規化: ホールディングス, 株式会社 等を除去
    def normalize(s: str) -> str:
        s = re.sub(r'株式会社|ホールディングス|ホールデイングス|HD|Holdings?', '', s)
        s = re.sub(r'[　\s]', '', s)
        return s.strip()

    company_normalized = normalize(company_name)
    page_normalized = normalize(page_title)

    # 最低4文字の一致チェック（3文字以下の会社名は2文字で可）
    min_len = min(4, len(company_normalized) - 1, len(company_name))
    min_len = max(2, min_len)

    # タイトルマッチ: 会社名の先頭N文字がページタイトルに含まれるか
    if len(company_normalized) >= min_len:
        key = company_normalized[:min_len]
        if key in page_normalized:
            return True

    # 逆方向: ページタイトルの主要部分が会社名に含まれるか
    if len(page_normalized) >= min_len:
        key = page_normalized[:min_len]
        if key in company_normalized:
            return True

    # ページ本文で会社名を確認（社名変更への対応）
    # infoboxや最初の段落に会社名が含まれるか
    page_text_top = sp.get_text()[:3000]
    if len(company_normalized) >= 3 and company_normalized[:4] in page_text_top:
        return True
    # 旧社名が本文に含まれる場合（例: マルハニチロ→Umios ページ内に「マルハニチロ」と記載）
    if len(company_name) >= 4 and company_name[:4] in page_text_top:
        return True

    return False


# ── Wikipedia 関連関数 ────────────────────────────────────────────────
def wiki_get_page(title: str) -> BeautifulSoup | None:
    url = f"https://ja.wikipedia.org/wiki/{quote(title)}"
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None


def wiki_search(query: str) -> list[str]:
    """検索クエリに一致するWikipediaページタイトルを複数返す。"""
    params = {
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": 5,
        "srprop": "", "format": "json",
    }
    try:
        r = SESSION.get("https://ja.wikipedia.org/w/api.php", params=params, timeout=10)
        results = r.json().get("query", {}).get("search", [])
        return [res["title"] for res in results]
    except Exception:
        return []


def _wiki_link_name(tag) -> str:
    for a in tag.find_all("a", href=True):
        href = a.get("href", "")
        if "/wiki/" in href and ":" not in href.split("/wiki/")[-1]:
            candidate = a.get_text(strip=True)
            if _is_valid_jp_name(candidate):
                return candidate
    return ""


def _is_mw_heading_div(tag) -> bool:
    if tag.name != "div":
        return False
    return "mw-heading" in " ".join(tag.get("class", []))


def _parse_wiki_table(table, ticker: str) -> list[dict]:
    results = []
    header_texts = [th.get_text(strip=True) for th in table.find_all("th")]
    name_col = None
    for i, h in enumerate(header_texts):
        if any(k in h for k in ["氏名", "名前", "社長名", "代表者", "氏　名"]):
            name_col = i
            break

    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        name = ""
        if name_col is not None and name_col < len(cells):
            name = _wiki_link_name(cells[name_col])
            if not name:
                name = cells[name_col].get_text(separator=" ", strip=True)
        else:
            for cell in cells:
                name = _wiki_link_name(cell)
                if name:
                    break
            if not name:
                for cell in cells:
                    cand = cell.get_text(separator=" ", strip=True)
                    if _is_valid_jp_name(cand):
                        name = cand
                        break
        if not _is_valid_jp_name(name):
            continue
        all_text = " ".join(c.get_text(separator=" ", strip=True) for c in cells)
        appt, rsgn = _extract_dates(all_text)
        results.append({
            "name":             name,
            "appointment_date": appt,
            "resignation_date": rsgn,
            "source":           "wikipedia_table",
        })
    return results


def _parse_wiki_list(text: str) -> list[dict]:
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name = ""
        for part in re.split(r"[　 、・,，\t]", line):
            part = part.strip()
            if _is_valid_jp_name(part):
                name = part
                break
        if not name:
            continue
        appt, rsgn = _extract_dates(line)
        results.append({
            "name":             name,
            "appointment_date": appt,
            "resignation_date": rsgn,
            "source":           "wikipedia_list",
        })
    return results


def _wiki_find_ceo_section(sp: BeautifulSoup, ticker: str) -> list[dict]:
    CEO_KW = [
        "歴代社長", "代表取締役一覧", "社長一覧",
        "歴代代表", "歴代の社長", "歴代会長兼社長", "歴代の代表取締役",
    ]
    results = []
    for h in sp.find_all(["h2", "h3", "h4"]):
        if not any(kw in h.get_text() for kw in CEO_KW):
            continue
        anchor = h.parent if _is_mw_heading_div(h.parent) else h
        nxt = anchor.find_next_sibling()
        while nxt:
            if nxt.name in ["h2", "h3", "h4"] or _is_mw_heading_div(nxt):
                break
            if nxt.name == "table":
                results.extend(_parse_wiki_table(nxt, ticker))
            elif nxt.name in ["ul", "ol"]:
                results.extend(_parse_wiki_list(nxt.get_text()))
            nxt = nxt.find_next_sibling()
        if results:
            break
    return results


def wiki_company_ceos(company_name: str, ticker: str) -> list[dict]:
    """会社名からWikipediaで歴代社長一覧を取得する（ページ検証強化版）。"""
    results = []
    tried: set[str] = set()

    # Strategy 1: 直接URLアクセス
    candidates = [
        company_name,
        re.sub(r"株式会社$", "", company_name).strip(),
        re.sub(r"^株式会社", "", company_name).strip(),
        company_name + "株式会社",
    ]
    for cand in dict.fromkeys(candidates):
        if not cand or cand in tried:
            continue
        tried.add(cand)
        sp = wiki_get_page(cand)
        if not sp:
            time.sleep(0.3)
            continue
        # ページタイトル検証（重要: 汚染防止）
        if not _page_matches_company(sp, company_name):
            time.sleep(0.2)
            continue
        found = _wiki_find_ceo_section(sp, ticker)
        if found:
            results.extend(found)
            break
        time.sleep(0.2)

    # Strategy 2: 検索API（ページタイトル検証必須）
    if not results:
        for q in [f"{company_name} 歴代社長", company_name]:
            titles = wiki_search(q)
            for title in titles:
                if title in tried:
                    continue
                tried.add(title)
                sp = wiki_get_page(title)
                if not sp:
                    continue
                # ページタイトル検証（汚染防止）
                if not _page_matches_company(sp, company_name):
                    time.sleep(0.2)
                    continue
                found = _wiki_find_ceo_section(sp, ticker)
                if found:
                    results.extend(found)
                    break
                time.sleep(0.3)
            if results:
                break

    # 重複排除・バリデーション
    seen, unique = set(), []
    for r in results:
        name = r.get("name", "")
        if name and name not in seen and _is_valid_jp_name(name):
            seen.add(name)
            unique.append(r)
    return unique[:20]


# ── 会社IRページから歴代社長を取得 ────────────────────────────────────
def ir_history_ceos(company_url: str, company_name: str, ticker: str) -> list[dict]:
    """会社IRページの沿革セクションから社長交代情報を抽出する。"""
    if not company_url:
        return []
    try:
        # 沿革ページを試す
        history_urls = [
            urljoin(company_url, "/ir/history/"),
            urljoin(company_url, "/company/history/"),
            urljoin(company_url, "/about/history/"),
            urljoin(company_url, "/corporate/history/"),
        ]
        for url in history_urls:
            try:
                r = SESSION.get(url, timeout=8)
                if r.status_code != 200:
                    continue
                r.encoding = r.apparent_encoding or "utf-8"
                sp = BeautifulSoup(r.text, "lxml")
                text = sp.get_text()
                # 社長交代を示すパターン
                ceos = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # 「代表取締役社長 山田太郎 就任」パターン
                    if re.search(r'(代表取締役|社長|CEO).*(就任|退任|選任)', line):
                        # 名前を抽出
                        for part in re.split(r'[　\s、・,]', line):
                            part = part.strip()
                            if _is_valid_jp_name(part):
                                appt, rsgn = _extract_dates(line)
                                ceos.append({
                                    "name": part,
                                    "appointment_date": appt,
                                    "resignation_date": rsgn,
                                    "source": url,
                                })
                                break
                if ceos:
                    return ceos[:15]
            except Exception:
                continue
    except Exception:
        pass
    return []


# ── Bing検索で歴代社長情報を取得 ──────────────────────────────────────
def bing_search_ceos(company_name: str, ticker: str) -> list[dict]:
    """Bing検索「会社名 歴代社長」で情報を取得する。"""
    query = f"{company_name} 歴代社長 一覧"
    try:
        r = SESSION.get(
            "https://www.bing.com/search",
            params={"q": query, "setlang": "ja", "cc": "JP"},
            timeout=10
        )
        if r.status_code != 200:
            return []
        r.encoding = "utf-8"
        sp = BeautifulSoup(r.text, "lxml")

        # 検索結果テキストから社長名を抽出
        results = []
        for snippet in sp.find_all(["p", "li", "span"], limit=50):
            text = snippet.get_text(strip=True)
            if not text or len(text) > 500:
                continue
            if not re.search(r'\d{4}年', text):
                continue
            for part in re.split(r'[　\s、・,，→→]', text):
                part = part.strip()
                if _is_valid_jp_name(part):
                    appt, rsgn = _extract_dates(text)
                    results.append({
                        "name": part,
                        "appointment_date": appt,
                        "resignation_date": rsgn,
                        "source": f"bing:{query}",
                    })
                    break

        # 重複排除
        seen, unique = set(), []
        for r in results:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
        return unique[:10]
    except Exception:
        return []


# ── 写真取得 ──────────────────────────────────────────────────────────
def search_photo_for_ceo(name: str) -> str | None:
    """WikipediaのinfoboxまたはBing画像検索で写真URLを取得。"""
    # Wikipedia infobox
    sp = wiki_get_page(name)
    if sp:
        infobox = sp.find("table", {"class": lambda c: c and "infobox" in " ".join(c)})
        img = infobox.find("img") if infobox else None
        if not img:
            img = sp.find("img", src=re.compile(r"upload\.wikimedia"))
        if img and img.get("src"):
            src = img["src"]
            if src.startswith("//"):
                src = "https:" + src
            # SVG/disambig は除外
            if not any(bad in src.lower() for bad in ["svg", "disambig", "question", "noimage"]):
                return src
    return None


def download_photo(url: str, dest: Path) -> bool:
    try:
        r = SESSION.get(url, timeout=20, stream=True)
        if r.status_code != 200:
            return False
        data = r.content
        if len(data) < 2000:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s　]', "_", s).strip("_")


# ── 株価取得 ──────────────────────────────────────────────────────────
_topix_cache: dict[str, dict] = {}


def _nearest_row(hist, dt):
    if hist is None or hist.empty:
        return None
    idx = max(0, hist.index.get_indexer([dt], method="nearest")[0])
    row = hist.iloc[idx]
    return {
        "open":         round(float(row["Open"]),  2),
        "close":        round(float(row["Close"]), 2),
        "trading_date": str(hist.index[idx].date()),
    }


def get_stock(ticker: str, date_str: str) -> dict | None:
    if not _HAS_YF:
        return None
    dt = parse_dt(date_str)
    if not dt or dt > datetime.now():
        return None
    try:
        start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=10)).strftime("%Y-%m-%d")
        hist  = yf.Ticker(f"{ticker}.T").history(start=start, end=end, auto_adjust=True)
        row   = _nearest_row(hist, dt)
        if not row:
            return None
        return {"target_date": dt.strftime("%Y-%m-%d"), **row}
    except Exception:
        return None


def get_topix(date_str: str) -> dict | None:
    if not _HAS_YF:
        return None
    dt = parse_dt(date_str)
    if not dt or dt > datetime.now():
        return None
    key = dt.strftime("%Y-%m-%d")
    if key in _topix_cache:
        return _topix_cache[key]
    try:
        start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=10)).strftime("%Y-%m-%d")
        hist  = yf.Ticker("^TOPIX").history(start=start, end=end, auto_adjust=True)
        row   = _nearest_row(hist, dt)
        if not row:
            return None
        result = {"target_date": key, **row}
        _topix_cache[key] = result
        return result
    except Exception:
        return None


# ── 会社処理 ──────────────────────────────────────────────────────────
def process_company(company: dict, existing_entry: dict | None) -> dict | None:
    """
    一社の歴代CEOデータを収集する。
    既存エントリがある場合は差分マージ。
    新規データがなければ None を返す。
    """
    ticker = str(company.get("ticker", ""))
    cname  = company.get("company_name", "")
    url    = company.get("url", "")

    log.info(f"[{ticker}] {cname} を処理中...")

    # 既存のCEO名セット（重複防止）
    existing_names = set()
    existing_ceos = []
    if existing_entry:
        existing_ceos = existing_entry.get("previous_ceos", [])
        existing_names = {c.get("name", "") for c in existing_ceos}

    all_found: list[dict] = []

    # 1. Wikipedia
    wiki_ceos = wiki_company_ceos(cname, ticker)
    if wiki_ceos:
        log.info(f"  Wikipedia: {len(wiki_ceos)}名")
        all_found.extend(wiki_ceos)
    time.sleep(0.5)

    # 2. 会社IRページ沿革（Wikipediaで見つからなかった場合）
    if not wiki_ceos and url:
        ir_ceos = ir_history_ceos(url, cname, ticker)
        if ir_ceos:
            log.info(f"  IR沿革: {len(ir_ceos)}名")
            all_found.extend(ir_ceos)

    # 3. Bing検索（他ソースで見つからなかった場合）
    if not all_found:
        bing_ceos = bing_search_ceos(cname, ticker)
        if bing_ceos:
            log.info(f"  Bing: {len(bing_ceos)}名")
            all_found.extend(bing_ceos)
        time.sleep(0.3)

    if not all_found:
        return None

    # 重複排除・既存とのマージ
    new_ceos = []
    seen = set(existing_names)
    for ceo in all_found:
        name = ceo.get("name", "")
        if not name or name in seen:
            continue
        seen.add(name)
        new_ceos.append(ceo)

    if not new_ceos:
        return None

    # 写真取得
    pdir = photos_dir(ticker)
    folder = pdir / f"{ticker}_{safe_name(cname)}" / "history"

    for i, ceo in enumerate(new_ceos):
        name = ceo.get("name", "")
        ceo_folder = folder / f"{i+1:02d}_{safe_name(name)}"
        photo_url = search_photo_for_ceo(name)
        if photo_url:
            ceo_folder.mkdir(parents=True, exist_ok=True)
            dest = ceo_folder / "photo_01.jpg"
            if download_photo(photo_url, dest):
                # パス記録（相対パス）
                rel = dest.relative_to(PROJECT_DIR)
                ceo["photo_path"] = str(rel)
                log.info(f"    写真取得: {name}")
            else:
                log.debug(f"    写真DL失敗: {name}")
        time.sleep(0.2)

        # 株価データ
        if ceo.get("appointment_date"):
            stock = get_stock(ticker, ceo["appointment_date"])
            if stock:
                ceo["stock_at_appointment"] = stock
            topix = get_topix(ceo["appointment_date"])
            if topix:
                ceo["topix_at_appointment"] = topix

    # マージ
    merged = existing_ceos + new_ceos
    return {
        "ticker":       ticker,
        "company_name": cname,
        "previous_ceos": merged,
    }


# ── コミット&プッシュ ──────────────────────────────────────────────────
def commit_and_push(batch_num: int, n_new: int, n_done: int) -> None:
    """ローカルの写真をGitHubにプッシュし、ローカルから削除。"""
    add_paths = ["data/history_supplement.json"]
    for d in ["photos_1", "photos_2", "photos_3", "photos_4", "photos_5"]:
        if (PROJECT_DIR / d).exists():
            add_paths.append(f"{d}/")

    r = subprocess.run(
        ["git", "-c", "core.quotePath=false", "add", "--ignore-removal"] + add_paths,
        cwd=str(PROJECT_DIR), capture_output=True
    )
    if r.returncode != 0:
        log.error(f"git add 失敗: {r.stderr.decode(errors='replace')}")
        return

    # コミット
    msg = f"[history2] batch {batch_num}: {n_new}社分 歴代CEO追加 ({n_done}社完了)"
    r = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=str(PROJECT_DIR), capture_output=True
    )
    if r.returncode != 0:
        log.warning("コミット対象なし、スキップ")
        return

    # プッシュ
    remote = REMOTE_URL or subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(PROJECT_DIR), capture_output=True
    ).stdout.decode().strip()

    r = subprocess.run(
        ["git", "push", remote if REMOTE_URL else "origin", "master"],
        cwd=str(PROJECT_DIR), capture_output=True
    )
    if r.returncode != 0:
        log.error(f"push失敗: {r.stderr.decode(errors='replace')}")
        return

    log.info(f"プッシュ完了 batch {batch_num}")

    # ローカルの写真を削除
    for d in ["photos_1", "photos_2", "photos_3", "photos_4", "photos_5"]:
        p = PROJECT_DIR / d
        if p.exists():
            import shutil
            shutil.rmtree(p)
            log.info(f"ローカル削除: {d}/")


# ── メイン ────────────────────────────────────────────────────────────
def main():
    # 既存データ読み込み
    ceo_data = json.loads(CEO_DATA_FILE.read_text(encoding="utf-8"))
    ceo_by_ticker = {str(c["ticker"]): c for c in ceo_data}

    hs = json.loads(HISTORY_OUT.read_text(encoding="utf-8"))
    hs_by_ticker = {str(x["ticker"]): x for x in hs}

    # 対象会社の決定:
    # 1) recollect_targets.json (汚染後に空になった会社)
    # 2) hs に未登録のすべての会社
    recollect_path = DATA_DIR / "recollect_targets.json"
    recollect_tickers = set()
    if recollect_path.exists():
        recollect = json.loads(recollect_path.read_text(encoding="utf-8"))
        recollect_tickers = {str(c["ticker"]) for c in recollect}

    # 歴代データが少ない/ない会社
    targets = []
    for c in ceo_data:
        ticker = str(c["ticker"])
        entry = hs_by_ticker.get(ticker)
        n_prev = len(entry.get("previous_ceos", [])) if entry else 0
        # 歴代CEOが0名 or 再収集対象
        if n_prev == 0 or ticker in recollect_tickers:
            targets.append(c)

    log.info(f"対象会社数: {len(targets)}社")

    # ── 処理ループ ──────────────────────────────────────────────────
    BATCH_SIZE = 50
    batch_new_count = 0
    total_done = 0
    total_added = 0

    for ci, company in enumerate(targets, 1):
        ticker = str(company["ticker"])
        existing = hs_by_ticker.get(ticker)

        result = process_company(company, existing)
        total_done += 1

        if result:
            # history_supplement を更新
            if ticker in hs_by_ticker:
                # 既存エントリを更新
                for i, x in enumerate(hs):
                    if str(x["ticker"]) == ticker:
                        hs[i] = result
                        hs_by_ticker[ticker] = result
                        break
            else:
                hs.append(result)
                hs_by_ticker[ticker] = result

            n_new = len(result["previous_ceos"])
            total_added += n_new
            batch_new_count += 1
            log.info(f"  [{ci}/{len(targets)}] {ticker} {company.get('company_name')}: {n_new}名追加")
        else:
            log.info(f"  [{ci}/{len(targets)}] {ticker} {company.get('company_name')}: データなし")

        # データ保存（常時）
        HISTORY_OUT.write_text(
            json.dumps(hs, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # バッチコミット
        if ci % BATCH_SIZE == 0:
            commit_and_push(ci // BATCH_SIZE, batch_new_count, total_done)
            batch_new_count = 0

    # 最終コミット
    if batch_new_count > 0 or total_done % BATCH_SIZE != 0:
        commit_and_push("final", batch_new_count, total_done)

    log.info(f"\n完了: {total_done}社処理, {total_added}名追加")


if __name__ == "__main__":
    main()
