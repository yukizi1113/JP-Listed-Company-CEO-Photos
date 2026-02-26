#!/usr/bin/env python3
"""
retry_photos.py  –  歴代社長写真 再取得スクリプト

history_supplement.json から写真のない歴代社長を読み込み、
以下の戦略で写真を取得して GitHub にコミット＆プッシュする。

戦略（順番に試みる）:
  1. Wikipedia 人物記事のサムネイル
  2. Wayback Machine + 在任中の会社ガバナンスページ
  3. DuckDuckGo 画像検索 ("{名前} {会社名} 社長" など)
  4. DuckDuckGo 画像検索 ("{名前} 経営者")
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlparse, urljoin

import requests
from bs4 import BeautifulSoup

# ── 設定 ─────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR      = PROJECT_DIR / "data"
HISTORY_FILE  = DATA_DIR / "history_supplement.json"
CEO_DATA_FILE = DATA_DIR / "ceo_data.json"
LOG_FILE      = PROJECT_DIR / "retry_photos.log"

GH_TOKEN  = os.environ.get("GH_TOKEN", "")
GH_USER   = "yukizi1113"
REPO_NAME = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"

BATCH_SIZE = 50   # 50社ごとにコミット＆プッシュ

# ── セッション ────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


# ── ログ ──────────────────────────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── ユーティリティ ────────────────────────────────────────────────────
def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s　]', "_", s).strip("_")


_NAME_BAD = [
    "代表", "取締役", "社長", "会社", "株式", "在任", "就任", "年月", "氏名",
    "単独", "創業", "受賞", "評議", "主筆", "締役", "東経", "年度", "新任",
    "実兄", "以上", "合計", "出典", "参照", "注記", "情報", "事業", "採用",
    "光章", "吾氏", "役員", "経歴", "部長", "常務", "専務", "議長", "理事",
    "その他", "期間", "備考", "名前", "読み", "から", "まで", "年間",
    # 追加: 役職・状態・業務用語
    "会長", "業務", "表明", "検査", "他者", "一覧", "管理", "担当", "運営",
    "設立", "退任", "現在", "歴代", "前任", "後任", "交代", "変更", "廃止",
    "増加", "減少", "合併", "分割", "移転", "解散", "破綻", "再建", "刷新",
]

_JP_CHARS = re.compile(r'^[\u4e00-\u9fff\u3040-\u30ff]+$')


def _is_valid_jp_name(text: str) -> bool:
    """有効な日本人名かどうかを検証する。

    - スペースなし: 2〜7字の漢字/仮名のみ
    - スペースあり: 姓(2〜4字) + スペース + 名(2〜5字) の形式のみ
    """
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    if any(c in text for c in ('\n', '\r', '\t')):
        return False
    if any(b in text for b in _NAME_BAD):
        return False

    # スペース（半角・全角）を含む場合: 姓名分割形式のみ許可
    if re.search(r'[\s\u3000]', text):
        parts = re.split(r'[\s\u3000]+', text)
        if len(parts) != 2:
            return False
        family, given = parts
        # 姓: 2〜4字、名: 2〜5字、それぞれ漢字/仮名のみ
        if not (2 <= len(family) <= 4 and 2 <= len(given) <= 5):
            return False
        if not _JP_CHARS.match(family) or not _JP_CHARS.match(given):
            return False
        return True

    # スペースなし: 2〜7字の漢字/仮名のみ
    if not (2 <= len(text) <= 7):
        return False
    return bool(_JP_CHARS.match(text))


def photos_dir(ticker: str) -> Path:
    try:
        if int(ticker) < 5500:
            return PROJECT_DIR / "photos_1"
    except ValueError:
        pass
    return PROJECT_DIR / "photos_2"


def download_photo(url: str, dest: Path) -> bool:
    """URLから写真をダウンロードして保存。最低5KB必要。"""
    try:
        r = SESSION.get(url, timeout=20, stream=True)
        if r.status_code != 200:
            return False
        data = r.content
        if len(data) < 5000:
            return False
        # 画像ファイルのマジックバイトを確認
        if not (data[:3] == b'\xff\xd8\xff' or  # JPEG
                data[:4] == b'\x89PNG' or         # PNG
                data[:4] == b'GIF8'):             # GIF
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as e:
        log.debug(f"  ダウンロード失敗 {url[:80]}: {e}")
        return False


def _resolve_img_url(src: str, page_url: str) -> str:
    """相対URLや//形式のURLを絶対URLに変換。"""
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http"):
        return src
    return urljoin(page_url, src)


def _is_bad_img(url: str) -> bool:
    """ロゴ・バナー等の不要画像を除外。"""
    bad = ['logo', 'banner', 'icon', 'arrow', 'spacer', 'blank', 'line_',
           'bg_', 'header', 'footer', 'nav', 'button', 'mark', 'back',
           'next', 'prev', '.gif', 'pixel', 'dot.', '1x1']
    return any(b in url.lower() for b in bad)


# ── 戦略 1: Wikipedia 人物記事 ────────────────────────────────────────
def wiki_person_photo(name: str) -> str | None:
    """Wikipedia の人物記事からサムネイル画像URLを返す。"""
    for search_name in [name, name.replace(" ", "　")]:
        url = f"https://ja.wikipedia.org/wiki/{quote(search_name)}"
        try:
            r = SESSION.get(url, timeout=15)
            if r.status_code != 200:
                continue
            sp = BeautifulSoup(r.text, "lxml")
            # infobox の画像
            infobox = sp.find("table", {"class": lambda c: c and "infobox" in " ".join(c) if c else False})
            if infobox:
                img = infobox.find("img")
                if img and img.get("src"):
                    src = img["src"]
                    if not _is_bad_img(src):
                        return _resolve_img_url(src, url)
            # ページ冒頭の最初の人物写真
            for img in sp.find_all("img", src=re.compile(r"upload\.wikimedia\.org")):
                src = img.get("src", "")
                if not _is_bad_img(src) and int(img.get("width", 0) or 0) > 50:
                    return _resolve_img_url(src, url)
            time.sleep(0.3)
        except Exception:
            pass
    return None


# ── 戦略 2: Wayback Machine ───────────────────────────────────────────
_WB_API = "https://archive.org/wayback/available"

_IR_SUFFIXES = [
    "/company/officer/", "/ir/governance/", "/governance/",
    "/company/governance/", "/about/officer/", "/company/officer.html",
    "/corporate/governance/", "/ir/officer/", "/company/executives/",
    "/about/executives/", "/ir/", "/company/", "/about/company/",
    "/", ""
]


def _wayback_snapshot(target_url: str, year: int) -> str | None:
    """Wayback Machine で指定URLの在任年に近いスナップショットURLを返す。"""
    ts = f"{year}0601120000"
    try:
        r = SESSION.get(_WB_API, params={"url": target_url, "timestamp": ts}, timeout=12)
        snap = r.json().get("archived_snapshots", {}).get("closest", {})
        if not snap.get("url"):
            return None
        snap_ts = snap.get("timestamp", "")
        snap_year = int(snap_ts[:4]) if snap_ts else 0
        if abs(snap_year - year) > 4:
            return None
        return snap["url"]
    except Exception:
        return None


def _find_img_near_name(sp: BeautifulSoup, name_fragments: list[str], page_url: str) -> str | None:
    """ページ内でCEO名の断片の近くにある写真URLを探す。"""
    for frag in name_fragments:
        if len(frag) < 2:
            continue
        for el in sp.find_all(string=re.compile(re.escape(frag))):
            parent = el.parent
            for _ in range(6):
                if parent is None:
                    break
                img = parent.find("img")
                if img:
                    src = img.get("src", "")
                    if src and not _is_bad_img(src):
                        w = int(img.get("width", 0) or 0)
                        h = int(img.get("height", 0) or 0)
                        if w == 0 or w > 30:  # サイズ不明か人物写真サイズ
                            return _resolve_img_url(src, page_url)
                parent = getattr(parent, "parent", None)
    return None


def _cdx_find_officer_pages(domain: str, year: int) -> list[str]:
    """CDX API でガバナンス/役員関連ページのスナップショットURLを列挙する。"""
    kws = ["officer", "executive", "governance", "director",
           "management", "yakuin", "torishimari", "keiei"]
    snap_urls = []
    ts_from = str(year - 1)
    ts_to   = str(year + 2)
    for kw in kws:
        try:
            r = SESSION.get(
                "http://web.archive.org/cdx/search/cdx",
                params={
                    "url": f"{domain}/*{kw}*",
                    "output": "json", "fl": "timestamp,original",
                    "from": ts_from, "to": ts_to,
                    "limit": "3", "statuscode": "200",
                },
                timeout=12,
            )
            rows = r.json()[1:]
            for ts, orig in rows:
                snap_urls.append(f"http://web.archive.org/web/{ts}/{orig}")
        except Exception:
            pass
    return snap_urls[:6]


def wayback_photo(name: str, company_url: str, year: int, company_name: str) -> str | None:
    """Wayback Machine で在任中のガバナンスページから写真を探す。"""
    if not company_url or year < 2000:
        return None
    parsed = urlparse(company_url)
    domain = parsed.netloc or parsed.path.split("/")[0]
    name_fragments = [name[-2:], name[:2]]

    # CDX API で役員関連ページを列挙
    snap_urls = _cdx_find_officer_pages(domain, year)

    # フォールバック: 既知のIRページ構造で探す
    if not snap_urls:
        root = f"{parsed.scheme}://{parsed.netloc}"
        for suffix in _IR_SUFFIXES[:6]:
            target = root.rstrip("/") + suffix
            snap_url = _wayback_snapshot(target, year)
            if snap_url:
                snap_urls.append(snap_url)

    for snap_url in snap_urls:
        try:
            r = SESSION.get(snap_url, timeout=20)
            if r.status_code != 200:
                continue
            sp = BeautifulSoup(r.text, "lxml")
            img_url = _find_img_near_name(sp, name_fragments, snap_url)
            if img_url:
                log.info(f"  Wayback hit: {snap_url[:80]}")
                return img_url
        except Exception:
            pass
        time.sleep(0.4)
    return None


# ── 戦略 3 & 4: DuckDuckGo 画像検索 ─────────────────────────────────
_SEARCH_CACHE: dict[str, list] = {}


def _yahoo_jp_images(query: str) -> list[str]:
    """Yahoo Japan 画像検索で画像URLリストを返す。"""
    if query in _SEARCH_CACHE:
        return _SEARCH_CACHE[query]
    try:
        r = SESSION.get(
            "https://search.yahoo.co.jp/image/search",
            params={"p": query},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        urls = re.findall(
            r'"(https?://[^"]{10,300}\.(?:jpg|jpeg|png))"', r.text
        )
        filtered = [u for u in urls if not _is_bad_img(u)]
        _SEARCH_CACHE[query] = filtered[:10]
        time.sleep(1.0)
        return filtered[:10]
    except Exception:
        return []


def _bing_images(query: str) -> list[str]:
    """Bing 画像検索の埋め込みJSONから画像URLリストを返す。"""
    if f"bing:{query}" in _SEARCH_CACHE:
        return _SEARCH_CACHE[f"bing:{query}"]
    try:
        r = SESSION.get(
            "https://www.bing.com/images/search",
            params={"q": query, "first": 1, "mkt": "ja-JP"},
            headers={"Accept-Language": "ja"},
            timeout=15,
        )
        # Bing は画像URLを複数パターンで埋め込む
        urls = re.findall(
            r'"murl"\s*:\s*"(https?://[^"]+\.(?:jpg|jpeg|png))"', r.text
        )
        if not urls:
            urls = re.findall(
                r'imgurl=(https?%3A%2F%2F[^&"]+(?:jpg|jpeg|png))', r.text
            )
            urls = [requests.utils.unquote(u) for u in urls]
        filtered = [u for u in urls if not _is_bad_img(u)]
        _SEARCH_CACHE[f"bing:{query}"] = filtered[:10]
        time.sleep(1.0)
        return filtered[:10]
    except Exception:
        return []


def image_search_photo(name: str, company_name: str) -> str | None:
    """Yahoo Japan / Bing 画像検索で人物写真URLを返す。"""
    queries = [
        f"{name} {company_name} 社長",
        f"{name} 代表取締役 {company_name}",
        f"{name} 社長 経営者",
    ]
    for q in queries:
        # Yahoo Japan を先に試す
        urls = _yahoo_jp_images(q)
        if urls:
            log.debug(f"  Yahoo hit for query: {q}")
            return urls[0]
        # Bing にフォールバック
        urls = _bing_images(q)
        if urls:
            log.debug(f"  Bing hit for query: {q}")
            return urls[0]
    return None


# ── メイン写真検索 ────────────────────────────────────────────────────
def find_photo(name: str, company_name: str, company_url: str,
               appointment_year: int) -> str | None:
    """複数戦略でCEOの写真URLを探す。"""
    log.debug(f"  検索: {name} / {company_name} / {appointment_year}年")

    # 1. Wikipedia
    url = wiki_person_photo(name)
    if url:
        log.info(f"  Wikipedia hit: {name}")
        return url
    time.sleep(0.3)

    # 2. Wayback Machine
    if company_url and appointment_year >= 2000:
        url = wayback_photo(name, company_url, appointment_year, company_name)
        if url:
            return url

    # 3. Yahoo Japan / Bing 画像検索
    url = image_search_photo(name, company_name)
    if url:
        log.info(f"  画像検索 hit: {name} ({url[:60]})")
        return url

    return None


# ── Git 操作 ──────────────────────────────────────────────────────────
def _git(*args, timeout=240):
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def commit_and_push(batch_num: int, new_photos: int, total_done: int) -> bool:
    _, status = _git("status", "--porcelain")
    if not status.strip():
        return True
    _git("add", "--ignore-removal", "--no-all", "photos_1/", "photos_2/", "data/")
    msg = (
        f"retry_photos batch {batch_num}: {new_photos}枚追加 (累計{total_done}社処理)\n\n"
        f"処理日時: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M JST')}\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    ok, out = _git("commit", "-m", msg)
    if not ok and "nothing to commit" not in out.lower():
        log.warning(f"コミット失敗: {out[:150]}")
        return False
    for attempt in range(3):
        ok, out = _git("push", REMOTE_URL, "master", timeout=300)
        if ok:
            log.info(f"push完了 batch {batch_num} ({new_photos}枚)")
            return True
        log.warning(f"push失敗 {attempt+1}/3: {out[:80]}")
        time.sleep(15)
    return False


def delete_local_files(paths: list[Path]) -> int:
    deleted = 0
    for p in paths:
        if p.is_file():
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
    return deleted


# ── メイン ───────────────────────────────────────────────────────────
def main():
    log.info("=== retry_photos.py 開始 ===")

    # データ読み込み
    history_data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    ceo_data_raw = json.loads(CEO_DATA_FILE.read_text(encoding="utf-8"))
    # ticker -> company_url のマップ
    url_map: dict[str, str] = {c["ticker"]: c.get("url", "") for c in ceo_data_raw}

    # 写真のないCEOを持つ企業を列挙
    todo: list[dict] = []
    for company in history_data:
        ceos_no_photo = [
            (i, p) for i, p in enumerate(company.get("previous_ceos", []))
            if not p.get("photo_path") and p.get("name")
        ]
        if ceos_no_photo:
            todo.append({
                "company": company,
                "ceos_no_photo": ceos_no_photo,
            })

    total_no_photo = sum(len(t["ceos_no_photo"]) for t in todo)
    log.info(f"写真未取得企業: {len(todo)} 社 / CEOs: {total_no_photo} 名")

    batch_num    = 1
    batch_photos = 0      # このバッチで取得した写真数
    total_got    = 0      # 累計取得写真数
    total_done   = 0      # 処理済み企業数
    batch_jpgs: list[Path] = []

    for ci, item in enumerate(todo, 1):
        company    = item["company"]
        ticker     = company["ticker"]
        comp_name  = company["company_name"]
        comp_url   = url_map.get(ticker, "")
        sn         = safe_name(comp_name)
        pdir       = photos_dir(ticker)
        got_this   = 0

        log.info(f"[{ci}/{len(todo)}] {comp_name} ({ticker}): {len(item['ceos_no_photo'])}名")

        for idx, ceo in item["ceos_no_photo"]:
            name = ceo.get("name", "")
            if not name or not _is_valid_jp_name(name):
                log.debug(f"  無効名スキップ: {repr(name)}")
                continue

            # 就任年を取得
            appt_date = ceo.get("appointment_date", "")
            appt_year = int(appt_date[:4]) if appt_date and len(appt_date) >= 4 else 2010

            photo_url = find_photo(name, comp_name, comp_url, appt_year)
            if not photo_url:
                log.debug(f"  写真なし: {name}")
                continue

            # 保存先
            pn       = safe_name(name)
            hist_dir = pdir / f"{ticker}_{sn}" / "history" / f"{idx+1:02d}_{pn}"
            dest     = hist_dir / "photo_01.jpg"

            if download_photo(photo_url, dest):
                ceo["photo_path"] = str(dest.relative_to(PROJECT_DIR))
                batch_jpgs.append(dest)
                got_this  += 1
                batch_photos += 1
                total_got  += 1
                log.info(f"  取得成功: {name} ({photo_url[:60]})")

                # info.json も更新
                info_file = hist_dir / "info.json"
                info_file.write_text(
                    json.dumps(ceo, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            time.sleep(0.5)

        log.info(f"  {comp_name}: {got_this}枚取得")
        total_done += 1

        # バッチコミット
        is_batch_end = (ci % BATCH_SIZE == 0) or (ci == len(todo))
        if is_batch_end:
            # history_supplement.json をアトミックに更新 (クラッシュ対策)
            tmp = HISTORY_FILE.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(history_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(HISTORY_FILE)
            pushed = commit_and_push(batch_num, batch_photos, total_done)
            if pushed:
                delete_local_files(batch_jpgs)
                log.info(f"  ローカルjpg削除: {len(batch_jpgs)}枚")
            batch_num   += 1
            batch_photos = 0
            batch_jpgs   = []
            time.sleep(3)

    log.info(f"=== 完了: 合計 {total_got} 枚取得 ===")


if __name__ == "__main__":
    import os, sys, ctypes
    lock_file = PROJECT_DIR / "retry_photos.lock"

    def _pid_alive(pid: int) -> bool:
        """Windows APIでPIDが生きているか確認。"""
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            # GetExitCodeProcess で終了済みかどうか確認
            code = ctypes.c_ulong(0)
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)
            STILL_ACTIVE = 259
            return code.value == STILL_ACTIVE
        except Exception:
            return False

    # 既存ロックを確認
    if lock_file.exists():
        try:
            existing_pid = int(lock_file.read_text().strip())
            if _pid_alive(existing_pid):
                print(f"Already running (PID {existing_pid}). Exiting.")
                sys.exit(0)
        except Exception:
            pass

    # ロック取得
    lock_file.write_text(str(os.getpid()))
    try:
        main()
    finally:
        try:
            lock_file.unlink()
        except Exception:
            pass
