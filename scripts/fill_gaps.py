#!/usr/bin/env python3
"""
fill_gaps.py  —  写真なし企業の社長写真を補完するスクリプト

戦略:
  1. kabutan.jp から現任CEOの名前を取得
  2. Wikipedia から歴代CEO名を改良パースで取得
  3. Yahoo Japan 画像検索で写真を取得
  4. 50社ごとに GitHub へコミット・プッシュ・ローカル削除

対象: data/no_photo_tickers.json に列挙された企業
"""

import json, logging, os, re, subprocess, sys, time, ctypes
from pathlib import Path
from urllib.parse import quote, urljoin, unquote

import requests
from bs4 import BeautifulSoup

# ── パス設定 ───────────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR      = PROJECT_DIR / "data"
HISTORY_FILE  = DATA_DIR / "history_supplement.json"
CEO_DATA_FILE = DATA_DIR / "ceo_data.json"
LOG_FILE      = PROJECT_DIR / "fill_gaps.log"
LOCK_FILE     = PROJECT_DIR / "fill_gaps.lock"

GH_TOKEN  = os.environ.get("GH_TOKEN", "")
GH_USER   = "yukizi1113"
REPO_NAME = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"

BATCH_SIZE = 50
_MAX_PHOTO_BYTES = 8 * 1024 * 1024

# ── ロギング ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
})

# ── 名前バリデーション ─────────────────────────────────────────────────
_JP_CHARS = re.compile(r'^[\u4e00-\u9fff\u3040-\u30ff]+$')
_NAME_BAD = [
    "代表","取締役","社長","会社","株式","在任","就任","年月","氏名",
    "単独","創業","評議","主筆","東証","年度","新任","実績","以下",
    "合計","出典","参考","注記","事業","採用","役員","経歴","部長",
    "常務","専務","議長","そのほか","期間","備考","名前","読み",
    "会長","業務","表明","検査","一覧","管理","拡充","運営","設立",
    "退任","現在","歴代","前任","後任","交代","変更","合併","解散",
    "破綻","再建","刷新","監査","執行","ホールディングス",
]

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
        # 姓2-5文字、名1-5文字（1文字名も許容）
        if not (2 <= len(family) <= 5 and 1 <= len(given) <= 5):
            return False
        return bool(_JP_CHARS.match(family) and _JP_CHARS.match(given))
    if not (2 <= len(text) <= 8):
        return False
    return bool(_JP_CHARS.match(text))


def normalize_name(s: str) -> str:
    """全角スペース・半角スペースを除去して名前を正規化"""
    return re.sub(r'[\s\u3000]+', '', s).strip()


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s　]', "_", s).strip("_")


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


# ── 写真ダウンロード ───────────────────────────────────────────────────
def download_photo(url: str, dest: Path) -> bool:
    try:
        r = SESSION.get(url, timeout=20, stream=True)
        if r.status_code != 200:
            return False
        clen = int(r.headers.get("Content-Length", 0))
        if clen > _MAX_PHOTO_BYTES:
            r.close(); return False
        chunks, total = [], 0
        for chunk in r.iter_content(65536):
            total += len(chunk)
            if total > _MAX_PHOTO_BYTES:
                r.close(); return False
            chunks.append(chunk)
        data = b"".join(chunks)
        if len(data) < 5000:
            return False
        # magic bytes check
        sig = data[:4]
        if not (sig[:2] == b'\xff\xd8' or sig[:4] == b'\x89PNG' or sig[:4] == b'GIF8'):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception:
        return False


# ── Yahoo Finance Japan から現任CEO名を取得 ───────────────────────────
def yahoo_finance_ceo(ticker: str) -> str | None:
    """Yahoo Finance Japan の企業情報ページから代表者名を取得"""
    url = f"https://finance.yahoo.co.jp/quote/{ticker}.T/profile"
    try:
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None
        r.encoding = "utf-8"
        sp = BeautifulSoup(r.text, "lxml")
        for th in sp.find_all("th"):
            if "代表者名" in th.get_text():
                td = th.find_next_sibling("td")
                if td:
                    name = td.get_text(strip=True).replace("\u3000", " ").strip()
                    if name and name not in ("―", "-", ""):
                        return name
    except Exception:
        pass
    return None


# ── minkabu.jp から現任CEO名を取得 ────────────────────────────────────
def minkabu_ceo(ticker: str) -> str | None:
    try:
        url = f"https://minkabu.jp/stock/{ticker}/outline"
        r = SESSION.get(url, timeout=10)
        if r.status_code != 200:
            return None
        r.encoding = r.apparent_encoding or "utf-8"
        sp = BeautifulSoup(r.text, "lxml")
        for row in sp.find_all("tr"):
            cells = row.find_all(["td", "th"])
            for i, c in enumerate(cells):
                if "代表取締役" in c.get_text() and i + 1 < len(cells):
                    name = cells[i+1].get_text(strip=True)
                    if _is_valid_jp_name(name):
                        return name
    except Exception:
        pass
    return None


# ── Wikipedia 改良パース ───────────────────────────────────────────────
def wiki_get_page(title: str) -> BeautifulSoup | None:
    url = f"https://ja.wikipedia.org/wiki/{quote(title)}"
    try:
        r = SESSION.get(url, timeout=12)
        if r.status_code != 200:
            return None
        r.encoding = "utf-8"
        return BeautifulSoup(r.text, "lxml")
    except Exception:
        return None


def wiki_search(query: str) -> str | None:
    params = {
        "action": "query", "list": "search",
        "srsearch": query, "srlimit": 3,
        "format": "json", "srprop": "snippet",
    }
    try:
        r = SESSION.get("https://ja.wikipedia.org/w/api.php", params=params, timeout=10)
        data = r.json()
        results = data.get("query", {}).get("search", [])
        return results[0]["title"] if results else None
    except Exception:
        return None


def _extract_names_from_table(table) -> list[str]:
    names = []
    for row in table.find_all("tr"):
        for cell in row.find_all(["td", "th"]):
            # リンクからまず探す
            for a in cell.find_all("a"):
                href = a.get("href", "")
                if "/wiki/" in href and ":" not in href.split("/wiki/")[-1]:
                    name = a.get_text(strip=True)
                    if _is_valid_jp_name(name):
                        names.append(name)
            # テキストから抽出
            text = cell.get_text(strip=True)
            if _is_valid_jp_name(text):
                names.append(text)
    return names


def _extract_names_from_text(text: str) -> list[str]:
    names = []
    # 「○○ ○○」形式の名前パターン
    pattern = r'[\u4e00-\u9fff]{2,5}[\s　][\u4e00-\u9fff]{2,5}'
    for m in re.finditer(pattern, text):
        name = m.group().strip()
        if _is_valid_jp_name(name):
            names.append(name)
    # スペースなし形式
    pattern2 = r'[\u4e00-\u9fff]{3,7}'
    for m in re.finditer(pattern2, text):
        name = m.group()
        if _is_valid_jp_name(name):
            names.append(name)
    return names


CEO_SECTIONS = [
    "歴代社長", "代表取締役社長", "社長一覧", "歴代代表取締役",
    "役員", "経営陣", "代表取締役", "社長", "会社概要", "沿革",
]


def wiki_get_ceos(company_name: str, ticker: str) -> list[str]:
    """Wikipediaから歴代社長名リストを取得（改良版）"""
    tried = set()
    all_names = []

    # 検索候補
    candidates = [
        company_name,
        re.sub(r"株式会社$|^株式会社|ホールディングス$|HD$", "", company_name).strip(),
        company_name + "株式会社",
    ]
    candidates = [c for c in dict.fromkeys(candidates) if c]

    for cand in candidates:
        if cand in tried:
            continue
        tried.add(cand)
        sp = wiki_get_page(cand)
        if not sp:
            time.sleep(0.3)
            continue

        # ページタイトル確認
        h1 = sp.find("h1", {"id": "firstHeading"})
        page_title = h1.get_text() if h1 else ""
        # 会社名の一部がタイトルに含まれるか確認（緩い条件）
        name_short = re.sub(r'[株式会社ホールディングスHD\s　]', '', company_name)[:4]
        if name_short and name_short not in page_title and company_name[:2] not in page_title:
            time.sleep(0.2)
            continue

        # 全ヘッダーを探してCEO関連セクションを見つける
        for section_name in CEO_SECTIONS:
            for tag in sp.find_all(["h2", "h3", "h4"]):
                if section_name in tag.get_text():
                    # このセクション以降の要素を処理
                    for sibling in tag.find_next_siblings():
                        if sibling.name in ["h2", "h3", "h4"]:
                            break
                        if sibling.name == "table":
                            names = _extract_names_from_table(sibling)
                            all_names.extend(names)
                        elif sibling.name in ["ul", "ol"]:
                            for li in sibling.find_all("li"):
                                for a in li.find_all("a"):
                                    href = a.get("href", "")
                                    if "/wiki/" in href and ":" not in href.split("/wiki/")[-1]:
                                        name = a.get_text(strip=True)
                                        if _is_valid_jp_name(name):
                                            all_names.append(name)
                                text = li.get_text(strip=True)
                                names = _extract_names_from_text(text)
                                all_names.extend(names)
                        elif sibling.name == "p":
                            names = _extract_names_from_text(sibling.get_text())
                            all_names.extend(names)

        # 基本情報ボックスから現在の社長も取得
        infobox = sp.find("table", class_=re.compile(r"infobox|wikitable"))
        if infobox:
            for row in infobox.find_all("tr"):
                cells = row.find_all(["td", "th"])
                for i, c in enumerate(cells):
                    if "代表" in c.get_text() or "社長" in c.get_text():
                        if i + 1 < len(cells):
                            # Parse "名前（代表取締役 社長）" format - extract 社長 specifically
                            raw = cells[i+1].get_text(" ", strip=True)
                            # Try to find person with 社長 title
                            for segment in re.split(r'[\n\r]', raw):
                                segment = segment.strip()
                                # Format: "名前（...社長...）" or "名前 （...社長...）"
                                m = re.match(r'^([\u4e00-\u9fff\u3040-\u30ff\s\u3000]{2,12})[（(].*?社長.*?[）)]', segment)
                                if m:
                                    name = m.group(1).replace("\u3000", " ").strip()
                                    if _is_valid_jp_name(name):
                                        all_names.append(name)
                                        break
                                # Also try plain name if 社長 is in header cell
                                elif "社長" in c.get_text():
                                    name = segment.split("（")[0].split("(")[0].replace("\u3000", " ").strip()
                                    if _is_valid_jp_name(name):
                                        all_names.append(name)

        if all_names:
            break
        time.sleep(0.2)

    # フォールバック: Wikipedia検索
    if not all_names:
        for q in [f"{company_name} 歴代社長", f"{company_name} 代表取締役"]:
            title = wiki_search(q)
            if not title or title in tried:
                continue
            tried.add(title)
            sp = wiki_get_page(title)
            if not sp:
                continue
            for section_name in CEO_SECTIONS:
                for tag in sp.find_all(["h2", "h3", "h4"]):
                    if section_name in tag.get_text():
                        for sibling in tag.find_next_siblings():
                            if sibling.name in ["h2", "h3", "h4"]:
                                break
                            if sibling.name == "table":
                                all_names.extend(_extract_names_from_table(sibling))
            if all_names:
                break
            time.sleep(0.3)

    # 重複除去
    seen, unique = set(), []
    for n in all_names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique[:20]


# ── Yahoo Japan 画像検索 ───────────────────────────────────────────────
_BAD_IMG_WORDS = ["logo", "icon", "banner", "noimage", "placeholder",
                  "button", "arrow", "spacer", "blank", "pixel"]


def _is_bad_img(url: str) -> bool:
    u = url.lower()
    return any(w in u for w in _BAD_IMG_WORDS)


def bing_image_search(name: str, company: str) -> str | None:
    """Bing 画像検索から人物写真URLを返す"""
    name_nospace = normalize_name(name)
    queries = [
        f"{name_nospace} {company} 社長",
        f"{name_nospace} 代表取締役",
        f"{name_nospace} 社長 経営者",
    ]
    for q in queries:
        try:
            r = SESSION.get(
                "https://www.bing.com/images/search",
                params={"q": q, "first": 1, "mkt": "ja-JP"},
                headers={"Accept-Language": "ja"},
                timeout=15,
            )
            # Extract direct image URLs from Bing's embedded JSON
            # Bing may HTML-encode quotes as &quot;
            urls = re.findall(
                r'"murl"\s*:\s*"(https?://[^"]+\.(?:jpg|jpeg|png))"', r.text
            )
            if not urls:
                # Try HTML-encoded form: &quot;murl&quot;:&quot;URL&quot;
                raw = re.findall(
                    r'&quot;murl&quot;:&quot;(https?://[^&]+\.(?:jpg|jpeg|png))&quot;',
                    r.text
                )
                urls = [u.replace("&amp;", "&") for u in raw]
            if not urls:
                raw2 = re.findall(
                    r'imgurl=(https?%3A%2F%2F[^&"]+(?:jpg|jpeg|png))', r.text
                )
                urls = [unquote(u) for u in raw2]
            filtered = [u for u in urls if not _is_bad_img(u)]
            if filtered:
                time.sleep(0.8)
                return filtered[0]
        except Exception:
            pass
        time.sleep(0.5)
    return None


# Keep old name as alias for compatibility
def yahoo_image_search(name: str, company: str) -> str | None:
    return bing_image_search(name, company)


# ── Wikipedia 肖像画 ───────────────────────────────────────────────────
def wikipedia_portrait(name: str) -> str | None:
    sp = wiki_get_page(name)
    if not sp:
        return None
    # infobox の画像
    for img in sp.find_all("img"):
        src = img.get("src", "")
        if src.startswith("//"):
            src = "https:" + src
        if src and any(ext in src.lower() for ext in [".jpg", ".jpeg", ".png"]):
            if not any(bad in src.lower() for bad in [
                "logo", "icon", "map", "flag", "svg", "disambig",
                "button", "arrow", "spacer", "blank",
            ]):
                return src
    return None


# ── Git コミット & プッシュ ────────────────────────────────────────────
def commit_and_push(batch_num: int, n_photos: int, n_done: int) -> bool:
    if n_photos == 0:
        return False
    try:
        cwd = str(PROJECT_DIR)
        # Only add directories that actually exist
        add_paths = ["data/history_supplement.json"]
        for d in ["photos_1", "photos_2", "photos_3", "photos_4", "photos_5"]:
            if (PROJECT_DIR / d).exists():
                add_paths.append(f"{d}/")
        subprocess.run(
            ["git", "-c", "core.quotePath=false", "add", "--ignore-removal"] + add_paths,
            cwd=cwd, check=True, capture_output=True
        )
        msg = f"fill_gaps batch {batch_num}: {n_photos}枚追加 (累計{n_done}社処理)"
        subprocess.run(
            ["git", "-c", "core.quotePath=false", "commit", "-m", msg],
            cwd=cwd, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-c", "core.quotePath=false", "push", REMOTE_URL, "master"],
            cwd=cwd, check=True, capture_output=True, timeout=120
        )
        log.info(f"push完了: batch {batch_num} ({n_photos}枚)")
        return True
    except Exception as e:
        log.error(f"push失敗: {e}")
        return False


def delete_local_files(files: list[Path]):
    for f in files:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


# ── Git インデックスから処理済みticker取得 ───────────────────────────
def get_already_done_tickers() -> set:
    """git ls-files で既にGitHubにある写真のticker一覧を取得"""
    import re as _re
    try:
        r = subprocess.run(
            ["git", "-c", "core.quotePath=false", "ls-files",
             "photos_1/", "photos_2/", "photos_3/", "photos_4/", "photos_5/"],
            capture_output=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_DIR)
        )
        done = set()
        for line in r.stdout.splitlines():
            m = _re.match(r"photos_\d+/([^/_]+)", line)
            if m:
                done.add(m.group(1))
        return done
    except Exception:
        return set()


# ── メイン処理 ────────────────────────────────────────────────────────
def main():
    log.info("=== fill_gaps.py 開始 ===")

    # 既に処理済み(GitHubに写真あり)のtickerをスキップ
    already_done = get_already_done_tickers()
    log.info(f"スキップ(既処理): {len(already_done)}社")

    # データ読み込み
    no_photo_list = json.loads((DATA_DIR / "no_photo_tickers.json").read_text(encoding="utf-8"))
    history_data_list = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    history_by_ticker = {str(item["ticker"]): item for item in history_data_list}
    ceo_data = json.loads(CEO_DATA_FILE.read_text(encoding="utf-8"))
    ceo_by_ticker = {str(c["ticker"]): c for c in ceo_data}

    log.info(f"対象: {len(no_photo_list)}社")

    batch_num = 1
    batch_photos = 0
    total_got = 0
    total_done = 0
    batch_jpgs: list[Path] = []

    for ci, (ticker, company_name) in enumerate(no_photo_list, 1):
        ticker = str(ticker)
        # 既にGitHubに写真がある場合はスキップ
        if ticker in already_done:
            log.debug(f"スキップ({ticker}): 既処理")
            total_done += 1
            continue
        log.info(f"[{ci}/{len(no_photo_list)}] {company_name} ({ticker})")
        got_this = 0
        pdir = photos_dir(ticker)
        sn = safe_name(company_name)
        comp_url = ceo_by_ticker.get(ticker, {}).get("url", "")

        # ── Step 1: 現任CEO取得 ─────────────────────────────────────────
        current_ceo_name = None
        curr = ceo_by_ticker.get(ticker, {}).get("current_ceo")
        if curr and curr.get("name") and _is_valid_jp_name(curr.get("name", "")):
            current_ceo_name = curr["name"]
        else:
            # Yahoo Finance Japan から取得
            current_ceo_name = yahoo_finance_ceo(ticker)
            if not current_ceo_name:
                current_ceo_name = minkabu_ceo(ticker)

        if current_ceo_name:
            curr_dir = pdir / f"{ticker}_{sn}" / "current"
            curr_photo = curr_dir / "photo_01.jpg"
            if not curr_photo.exists():
                # 写真検索
                photo_url = (
                    wikipedia_portrait(normalize_name(current_ceo_name))
                    or bing_image_search(current_ceo_name, company_name)
                )
                if photo_url and download_photo(photo_url, curr_photo):
                    batch_jpgs.append(curr_photo)
                    got_this += 1
                    batch_photos += 1
                    total_got += 1
                    log.info(f"  現任CEO写真取得: {current_ceo_name}")

        # ── Step 2: 歴代CEO取得 ─────────────────────────────────────────
        # 既存データ確認
        existing = history_by_ticker.get(ticker, {})
        existing_names = {c.get("name") for c in existing.get("previous_ceos", [])}

        # Wikipedia から取得
        wiki_names = wiki_get_ceos(company_name, ticker)
        new_names = [n for n in wiki_names if n not in existing_names]

        if new_names:
            log.info(f"  Wikipedia CEO名取得: {new_names[:5]}")

        # 既存 + 新規のうち写真なし分を処理
        all_ceo_names = list(existing_names - {None, ""}) + new_names

        for idx, name in enumerate(all_ceo_names[:15]):
            if not _is_valid_jp_name(name):
                continue
            pn = safe_name(name)
            hist_dir = pdir / f"{ticker}_{sn}" / "history" / f"{idx+1:02d}_{pn}"
            dest = hist_dir / "photo_01.jpg"
            if dest.exists():
                continue  # 既存スキップ

            photo_url = (
                wikipedia_portrait(normalize_name(name))
                or bing_image_search(name, company_name)
            )
            if photo_url and download_photo(photo_url, dest):
                # info.json 保存
                info = {"name": name, "title": "代表取締役社長",
                        "appointment_date": None, "resignation_date": None,
                        "source": "fill_gaps", "photo_path": str(dest.relative_to(PROJECT_DIR))}
                hist_dir.mkdir(parents=True, exist_ok=True)
                (hist_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
                batch_jpgs.append(dest)
                got_this += 1
                batch_photos += 1
                total_got += 1
                log.info(f"  歴代CEO写真取得: {name}")
                time.sleep(0.5)

        # history_supplement.json 更新
        if new_names:
            if ticker not in history_by_ticker:
                new_entry = {
                    "ticker": ticker,
                    "company_name": company_name,
                    "previous_ceos": [
                        {"name": n, "title": "代表取締役社長",
                         "appointment_date": None, "resignation_date": None,
                         "source": "fill_gaps_wiki", "photo_path": None}
                        for n in new_names
                    ],
                    "enriched_at": "2026-03-07",
                }
                history_data_list.append(new_entry)
                history_by_ticker[ticker] = new_entry

        log.info(f"  {company_name}: {got_this}枚取得")
        total_done += 1

        # ── バッチコミット ────────────────────────────────────────────
        is_batch_end = (ci % BATCH_SIZE == 0) or (ci == len(no_photo_list))
        if is_batch_end:
            tmp = HISTORY_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(history_data_list, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(HISTORY_FILE)
            pushed = commit_and_push(batch_num, batch_photos, total_done)
            if pushed:
                delete_local_files(batch_jpgs)
                log.info(f"  ローカルjpg削除: {len(batch_jpgs)}枚")
            batch_num += 1
            batch_photos = 0
            batch_jpgs = []
            time.sleep(3)

    log.info(f"=== 完了! 合計{total_got}枚取得 ===")


if __name__ == "__main__":
    def _pid_alive(pid: int) -> bool:
        try:
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle: return False
            code = ctypes.c_ulong(0)
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return code.value == 259
        except Exception:
            return False

    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            if _pid_alive(pid):
                print(f"Already running (PID {pid}). Exiting.")
                sys.exit(0)
        except Exception:
            pass

    LOCK_FILE.write_text(str(os.getpid()))
    try:
        main()
    finally:
        LOCK_FILE.unlink(missing_ok=True)
