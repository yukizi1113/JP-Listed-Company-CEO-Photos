#!/usr/bin/env python3
"""
collect_history.py  –  歴代社長情報の収集スクリプト

・Wikipediaから歴代社長リストを取得
・就任日の始値・退任日の終値（株価＋TOPIX）を取得
・写真をWikipediaから取得し photos_1/ または photos_2/ に保存
・30社ごとにGitHubへコミット＆プッシュ、ローカルjpg削除
"""

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

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
HISTORY_PROGRESS = DATA_DIR / "history_progress.json"
LOG_FILE         = PROJECT_DIR / "collect_history.log"

GH_TOKEN  = os.environ.get("GH_TOKEN", "")
GH_USER   = "yukizi1113"
REPO_NAME = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = (
    f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"
    if GH_TOKEN else f"https://github.com/{GH_USER}/{REPO_NAME}.git"
)


def photos_dir(ticker: str) -> Path:
    """ティッカー < 5500 → photos_1、それ以外 → photos_2"""
    try:
        if int(ticker) < 5500:
            return PROJECT_DIR / "photos_1"
    except ValueError:
        pass
    return PROJECT_DIR / "photos_2"


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

# ── 名前バリデーション ────────────────────────────────────────────────
_NAME_BAD = [
    "代表", "取締役", "社長", "会社", "株式", "在任", "就任", "年月", "氏名",
    "単独", "創業", "受賞", "評議", "主筆", "締役", "東経", "年度", "新任",
    "実兄", "以上", "合計", "出典", "参照", "注記", "情報", "事業", "採用",
    "光章", "吾氏", "役員", "経歴", "部長", "常務", "専務", "議長", "理事",
    "その他", "期間", "備考", "名前", "読み", "から", "まで", "年間",
]


def _is_valid_jp_name(text: str) -> bool:
    text = text.strip()
    if len(text) < 2 or len(text) > 10:
        return False
    if "\n" in text or "\r" in text:
        return False
    if not re.fullmatch(
        r"[\u4e00-\u9fff\u3040-\u30ff]{1,4}[ \u3000]?[\u4e00-\u9fff\u3040-\u30ff]{1,5}",
        text,
    ):
        return False
    return not any(b in text for b in _NAME_BAD)


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


# ── 株価・TOPIX取得（就任日始値・退任日終値） ─────────────────────────
_topix_cache: dict[str, dict] = {}


def _nearest_row(hist, dt) -> dict | None:
    if hist is None or hist.empty:
        return None
    idx = max(0, hist.index.get_indexer([dt], method="nearest")[0])
    row = hist.iloc[idx]
    return {
        "open":         round(float(row["Open"]),  2),
        "close":        round(float(row["Close"]), 2),
        "high":         round(float(row["High"]),  2),
        "low":          round(float(row["Low"]),   2),
        "trading_date": str(hist.index[idx].date()),
    }


def get_stock(ticker: str, date_str: str) -> dict | None:
    """指定日に最も近い取引日の株価（始値・終値）を返す。"""
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
        return {
            "target_date":  dt.strftime("%Y-%m-%d"),
            "trading_date": row["trading_date"],
            "open":         row["open"],
            "close":        row["close"],
            "high":         row["high"],
            "low":          row["low"],
            "currency":     "JPY",
        }
    except Exception as e:
        log.debug(f"株価取得失敗 [{ticker}] {date_str}: {e}")
        return None


def get_topix(date_str: str) -> dict | None:
    """指定日に最も近い取引日のTOPIX（始値・終値）を返す。"""
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
        result = {
            "target_date":  key,
            "trading_date": row["trading_date"],
            "open":         row["open"],
            "close":        row["close"],
        }
        _topix_cache[key] = result
        return result
    except Exception as e:
        log.debug(f"TOPIX取得失敗 {date_str}: {e}")
        return None


def add_stock(entry: dict, ticker: str) -> None:
    """就任日始値・退任日終値および同日TOPIXをエントリに追加（in-place）。"""
    appt = entry.get("appointment_date")
    if appt and "open_at_appointment" not in entry:
        p = get_stock(ticker, appt)
        if p:
            entry["stock_price_at_appointment"] = p
            entry["open_at_appointment"]        = p["open"]   # 就任日始値
        t = get_topix(appt)
        if t:
            entry["topix_at_appointment"]       = t
            entry["topix_open_at_appointment"]  = t["open"]   # 同日TOPIX始値

    rsgn = entry.get("resignation_date")
    if rsgn and "close_at_resignation" not in entry:
        p = get_stock(ticker, rsgn)
        if p:
            entry["stock_price_at_resignation"] = p
            entry["close_at_resignation"]       = p["close"]  # 退任日終値
        t = get_topix(rsgn)
        if t:
            entry["topix_at_resignation"]       = t
            entry["topix_close_at_resignation"] = t["close"]  # 同日TOPIX終値


# ── Wikipedia ────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CEO-History-Bot/2.0 (academic research; github.com/yukizi1113)"
})


def wiki_get_page(title: str) -> BeautifulSoup | None:
    url = f"https://ja.wikipedia.org/wiki/{quote(title)}"
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            return None
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None


def wiki_search(query: str) -> str | None:
    params = {
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": 3,
        "srprop": "", "format": "json",
    }
    try:
        r = SESSION.get("https://ja.wikipedia.org/w/api.php", params=params, timeout=10)
        results = r.json().get("query", {}).get("search", [])
        return results[0]["title"] if results else None
    except Exception:
        return None


def _wiki_link_name(tag) -> str:
    """タグ内のWikipediaリンクから人名を抽出する。"""
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
    # ヘッダ行を取得して名前カラムを特定
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

        # 名前を取得
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


def _parse_wiki_list(text: str, ticker: str) -> list[dict]:
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


def _parse_text_for_ceos(text: str, src_url: str, ticker: str) -> list[dict]:
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not re.search(r"\d{4}年", line):
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
            "source":           src_url,
        })
    return results


def _wiki_find_ceo_section(sp: BeautifulSoup, ticker: str, src_url: str) -> list[dict]:
    """
    歴代社長セクションを探してCEO一覧を返す。
    新旧Wikipedia構造両対応:
      旧: <h3>タイトル</h3> の直後がコンテンツ
      新: <div class="mw-heading"><h3>...</h3></div> の次sibling がコンテンツ
    """
    CEO_KW = [
        "歴代社長", "代表取締役一覧", "社長一覧",
        "歴代代表", "歴代の社長", "歴代会長兼社長", "歴代の代表取締役",
    ]
    results = []
    for h in sp.find_all(["h2", "h3", "h4"]):
        if not any(kw in h.get_text() for kw in CEO_KW):
            continue
        # 新構造: h の親が mw-heading div ならそこから次siblingを辿る
        anchor = h.parent if _is_mw_heading_div(h.parent) else h
        nxt = anchor.find_next_sibling()
        while nxt:
            if nxt.name in ["h2", "h3", "h4"] or _is_mw_heading_div(nxt):
                break
            if nxt.name == "table":
                results.extend(_parse_wiki_table(nxt, ticker))
            elif nxt.name in ["ul", "ol"]:
                results.extend(_parse_wiki_list(nxt.get_text(), ticker))
            else:
                txt = nxt.get_text()
                if re.search(r"\d{4}年", txt):
                    results.extend(_parse_text_for_ceos(txt, src_url, ticker))
            nxt = nxt.find_next_sibling()
        if results:
            break
    return results


def wiki_company_ceos(company_name: str, ticker: str) -> list[dict]:
    """会社名からWikipediaで歴代社長一覧を取得する。"""
    results = []
    tried: set[str] = set()

    # Strategy 1: 直接URLアクセス（最も信頼性が高い）
    candidates = [
        company_name,
        re.sub(r"株式会社$", "", company_name).strip(),
        re.sub(r"^株式会社", "", company_name).strip(),
        company_name + "株式会社",
    ]
    for cand in dict.fromkeys(candidates):  # 重複排除・順序維持
        if not cand or cand in tried:
            continue
        tried.add(cand)
        sp = wiki_get_page(cand)
        if not sp:
            time.sleep(0.3)
            continue
        h1 = sp.find("h1", {"id": "firstHeading"})
        if h1:
            page_title = h1.get_text()
            # 会社名の先頭2〜3文字が一致しない場合はスキップ
            if not any(c in page_title for c in [company_name[:3], company_name[:2]]):
                time.sleep(0.2)
                continue
        src = f"https://ja.wikipedia.org/wiki/{quote(cand)}"
        found = _wiki_find_ceo_section(sp, ticker, src)
        if found:
            results.extend(found)
            break
        time.sleep(0.2)

    # Strategy 2: 検索API（フォールバック）
    if not results:
        for q in [f"{company_name} 歴代社長", company_name]:
            title = wiki_search(q)
            if not title or title in tried:
                continue
            tried.add(title)
            sp = wiki_get_page(title)
            if not sp:
                continue
            src = f"https://ja.wikipedia.org/wiki/{quote(title)}"
            found = _wiki_find_ceo_section(sp, ticker, src)
            if found:
                results.extend(found)
                break
            time.sleep(0.3)

    # 重複排除・バリデーション
    seen, unique = set(), []
    for r in results:
        name = r.get("name", "")
        if name and name not in seen and _is_valid_jp_name(name):
            seen.add(name)
            unique.append(r)
    return unique[:20]


# ── ファイル名サニタイズ ──────────────────────────────────────────────
def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s　]', "_", s).strip("_")


# ── 写真ダウンロード ──────────────────────────────────────────────────
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


def search_photo_for_ceo(name: str) -> str | None:
    """Wikipediaの人物記事からサムネイル画像URLを取得。"""
    sp = wiki_get_page(name)
    if not sp:
        return None
    # infobox 内の画像を探す
    infobox = sp.find("table", {"class": lambda c: c and "infobox" in " ".join(c)})
    img = infobox.find("img") if infobox else None
    if not img:
        # 記事冒頭の最初の画像
        img = sp.find("img", src=re.compile(r"upload\.wikimedia"))
    if img and img.get("src"):
        src = img["src"]
        if src.startswith("//"):
            src = "https:" + src
        return src
    return None


# ── 会社処理 ──────────────────────────────────────────────────────────
def process(company: dict) -> dict:
    ticker = company.get("ticker", "")
    name   = company.get("company_name", "")

    result = {
        "ticker":        ticker,
        "company_name":  name,
        "previous_ceos": [],
    }

    prev_ceos = wiki_company_ceos(name, ticker)
    log.info(f"  [{ticker}] {name}: {len(prev_ceos)}名")

    if not prev_ceos:
        return result

    pdir = photos_dir(ticker)
    sn   = safe_name(name)

    for i, entry in enumerate(prev_ceos, 1):
        add_stock(entry, ticker)

        pn       = safe_name(entry.get("name", "unknown"))
        hist_dir = pdir / f"{ticker}_{sn}" / "history" / f"{i:02d}_{pn}"
        hist_dir.mkdir(parents=True, exist_ok=True)

        # 写真取得
        photo_url = search_photo_for_ceo(entry.get("name", ""))
        if photo_url:
            dest = hist_dir / "photo_01.jpg"
            if download_photo(photo_url, dest):
                entry["photo_path"] = str(dest.relative_to(PROJECT_DIR))
        time.sleep(0.1)

        (hist_dir / "info.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    result["previous_ceos"] = prev_ceos
    return result


# ── Git操作 ──────────────────────────────────────────────────────────
def _git(*args, timeout: int = 240) -> tuple[bool, str]:
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def commit_and_push_history(batch_num: int, done_cnt: int, total: int, with_prev: int) -> bool:
    _, status = _git("status", "--porcelain")
    if not status.strip():
        log.info("コミット対象なし（変更なし）")
        return True

    _git("add", "--no-all", "data/")
    _git("add", "--ignore-removal", "--no-all", "photos_1/", "photos_2/")

    msg = (
        f"History batch {batch_num}: {done_cnt}/{total}社, 歴代社長{with_prev}社分\n\n"
        f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M JST')}\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    ok, out = _git("commit", "-m", msg)
    if not ok and "nothing to commit" not in out.lower():
        log.warning(f"コミット失敗: {out[:200]}")
        return False

    for attempt in range(3):
        ok, out = _git("push", REMOTE_URL, "master", timeout=300)
        if ok:
            log.info(f"GitHub push成功 (batch {batch_num})")
            return True
        log.warning(f"push失敗 attempt {attempt+1}: {out[:120]}")
        time.sleep(15 * (attempt + 1))
    return False


def delete_local_history(batch_tickers: list[str]) -> int:
    """push成功後にローカルの全jpg・jsonを削除してディスクを解放。"""
    deleted = 0
    for ticker in batch_tickers:
        for pdir in [PROJECT_DIR / "photos_1", PROJECT_DIR / "photos_2"]:
            for company_dir in pdir.glob(f"{ticker}_*"):
                for f in company_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".json"):
                        try:
                            f.unlink()
                            deleted += 1
                        except Exception:
                            pass
    return deleted


# ── メイン ───────────────────────────────────────────────────────────
def main():
    log.info("=== collect_history.py 開始 ===")

    companies = json.loads(CEO_DATA_FILE.read_text(encoding="utf-8"))
    log.info(f"企業数: {len(companies)}")

    done: set[str] = set()
    if HISTORY_PROGRESS.exists():
        done = set(json.loads(HISTORY_PROGRESS.read_text(encoding="utf-8")))
        log.info(f"進捗復元: {len(done)} 社完了済み")

    results: dict[str, dict] = {}
    if HISTORY_OUT.exists():
        for r in json.loads(HISTORY_OUT.read_text(encoding="utf-8")):
            results[r["ticker"]] = r

    todo = [c for c in companies if c.get("ticker") not in done]
    log.info(f"残り: {len(todo)} 社")

    BATCH_SIZE    = 30
    batch_num     = len(done) // BATCH_SIZE + 1
    batch_tickers: list[str] = []

    for i, company in enumerate(todo, 1):
        ticker = company.get("ticker", "")
        name   = company.get("company_name", ticker)
        try:
            r = process(company)
            results[ticker] = r
            n_prev = len(r.get("previous_ceos", []))
            log.info(f"[{len(done)+i}/{len(companies)}] {name}: 歴代{n_prev}名")
        except Exception as e:
            log.warning(f"エラー [{ticker}] {name}: {e}")
            results[ticker] = {"ticker": ticker, "company_name": name, "previous_ceos": []}

        done.add(ticker)
        batch_tickers.append(ticker)

        is_batch_end = (i % BATCH_SIZE == 0) or (i == len(todo))
        if is_batch_end:
            # 進捗保存
            HISTORY_PROGRESS.write_text(
                json.dumps(list(done)), encoding="utf-8"
            )
            HISTORY_OUT.write_text(
                json.dumps(list(results.values()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with_prev = sum(1 for r in results.values() if r.get("previous_ceos"))
            log.info(f"進捗保存: {len(done)}/{len(companies)} | 歴代あり: {with_prev} 社")

            if GH_TOKEN:
                pushed = commit_and_push_history(
                    batch_num, len(done), len(companies), with_prev
                )
                if pushed:
                    deleted = delete_local_history(batch_tickers)
                    log.info(f"ローカルjpg削除: {deleted} 枚")

            batch_num += 1
            batch_tickers = []

    log.info("=== collect_history.py 完了 ===")


if __name__ == "__main__":
    main()
