"""
apply_manual_dates.py
manual_ceo_dates.json の内容を history_supplement.json に追加/更新し、
新規追加エントリの写真を retry_photos.py と同じ戦略で取得する。

Usage:
  python scripts/apply_manual_dates.py
"""

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

PROJECT_DIR   = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR      = PROJECT_DIR / "data"
MANUAL_FILE   = DATA_DIR / "manual_ceo_dates.json"
HISTORY_FILE  = DATA_DIR / "history_supplement.json"
CEO_DATA_FILE = DATA_DIR / "ceo_data.json"
LOG_FILE      = PROJECT_DIR / "apply_manual_dates.log"

GH_TOKEN   = os.environ.get("GH_TOKEN", "")
GH_USER    = "yukizi1113"
REPO_NAME  = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s\u3000]', "_", s).strip("_")


def photos_dir(ticker: str) -> Path:
    try:
        if int(ticker) < 5500:
            return PROJECT_DIR / "photos_1"
    except ValueError:
        pass
    return PROJECT_DIR / "photos_2"


def _is_bad_img(url: str) -> bool:
    bad = ['logo', 'banner', 'icon', 'arrow', 'spacer', 'blank', 'line_',
           'bg_', 'header', 'footer', 'nav', 'button', 'mark', 'back',
           'next', 'prev', '.gif', 'pixel', 'dot.', '1x1']
    return any(b in url.lower() for b in bad)


def download_photo(url: str, dest: Path) -> bool:
    try:
        r = SESSION.get(url, timeout=20, stream=True)
        if r.status_code != 200:
            return False
        data = r.content
        if len(data) < 5000:
            return False
        if not (data[:3] == b'\xff\xd8\xff' or
                data[:4] == b'\x89PNG' or
                data[:4] == b'GIF8'):
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def _yahoo_jp_images(query: str) -> list[str]:
    try:
        r = SESSION.get(
            "https://search.yahoo.co.jp/image/search",
            params={"p": query}, timeout=15,
        )
        if r.status_code != 200:
            return []
        urls = re.findall(r'"(https?://[^"]{10,300}\.(?:jpg|jpeg|png))"', r.text)
        filtered = [u for u in urls if not _is_bad_img(u)]
        time.sleep(1.0)
        return filtered[:5]
    except Exception:
        return []


def _resolve_img_url(src: str, page_url: str) -> str:
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("http"):
        return src
    return urljoin(page_url, src)


def wiki_person_photo(name: str) -> str | None:
    url = f"https://ja.wikipedia.org/wiki/{quote(name)}"
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            return None
        sp = BeautifulSoup(r.text, "lxml")
        infobox = sp.find("table", {"class": lambda c: c and "infobox" in " ".join(c) if c else False})
        if infobox:
            img = infobox.find("img")
            if img and img.get("src") and not _is_bad_img(img["src"]):
                return _resolve_img_url(img["src"], url)
        for img in sp.find_all("img", src=re.compile(r"upload\.wikimedia\.org")):
            src = img.get("src", "")
            if not _is_bad_img(src) and int(img.get("width", 0) or 0) > 50:
                return _resolve_img_url(src, url)
    except Exception:
        pass
    return None


def find_photo(name: str, company_name: str) -> str | None:
    # 1. Wikipedia
    url = wiki_person_photo(name)
    if url:
        log.info(f"  Wikipedia hit: {name}")
        return url
    time.sleep(0.3)
    # 2. Yahoo Japan image search
    for q in [f"{name} {company_name} 社長", f"{name} 代表取締役"]:
        urls = _yahoo_jp_images(q)
        if urls:
            log.info(f"  画像検索 hit: {name} ({urls[0][:60]})")
            return urls[0]
    return None


def _git(*args, timeout=240):
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def commit_and_push(msg: str) -> bool:
    _, status = _git("status", "--porcelain")
    if not status.strip():
        log.info("変更なし、コミットスキップ")
        return True
    _git("add", "--ignore-removal", "--no-all", "photos_1/", "photos_2/", "data/")
    ok, out = _git("commit", "-m", msg)
    if not ok and "nothing to commit" not in out.lower():
        log.warning(f"コミット失敗: {out[:150]}")
        return False
    for attempt in range(3):
        ok, out = _git("push", REMOTE_URL, "master", timeout=300)
        if ok:
            log.info("push完了")
            return True
        log.warning(f"push失敗 {attempt+1}/3: {out[:80]}")
        time.sleep(15)
    return False


def main():
    log.info("=== apply_manual_dates.py 開始 ===")

    manual_data = json.loads(MANUAL_FILE.read_text(encoding="utf-8"))
    history_data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    ceo_data_raw = json.loads(CEO_DATA_FILE.read_text(encoding="utf-8"))
    url_map = {c["ticker"]: c.get("url", "") for c in ceo_data_raw}

    # history_data を ticker でインデックス化
    history_by_ticker = {c["ticker"]: c for c in history_data}

    new_jpgs: list[Path] = []
    updated = 0

    for manual_entry in manual_data:
        ticker = manual_entry["ticker"]
        comp_name = manual_entry["company_name"]
        sn = safe_name(comp_name)
        pdir = photos_dir(ticker)

        if ticker in history_by_ticker:
            log.info(f"  {ticker} {comp_name}: 既存エントリに上書きマージ")
            existing = history_by_ticker[ticker]
        else:
            log.info(f"  {ticker} {comp_name}: 新規追加")
            existing = {
                "ticker": ticker,
                "company_name": comp_name,
                "previous_ceos": [],
            }
            history_data.append(existing)
            history_by_ticker[ticker] = existing

        # 既存名前リスト
        existing_names = {p["name"] for p in existing.get("previous_ceos", [])}

        for idx, ceo in enumerate(manual_entry.get("previous_ceos", [])):
            name = ceo["name"]

            # 既存エントリの日付を更新 or 新規追加
            existing_ceo = next(
                (p for p in existing.get("previous_ceos", []) if p["name"] == name),
                None,
            )
            if existing_ceo:
                # 日付だけ上書き
                if ceo.get("appointment_date"):
                    existing_ceo["appointment_date"] = ceo["appointment_date"]
                if ceo.get("resignation_date"):
                    existing_ceo["resignation_date"] = ceo["resignation_date"]
                log.info(f"    {name}: 日付更新")
                target_ceo = existing_ceo
            else:
                # 新規追加
                new_ceo = {
                    "name": name,
                    "title": ceo.get("title", "代表取締役社長"),
                    "appointment_date": ceo.get("appointment_date"),
                    "resignation_date": ceo.get("resignation_date"),
                    "photo_path": None,
                }
                existing.setdefault("previous_ceos", []).append(new_ceo)
                log.info(f"    {name}: 新規追加")
                target_ceo = new_ceo
                updated += 1

            # 写真がない場合は取得試みる
            if not target_ceo.get("photo_path"):
                photo_url = find_photo(name, comp_name)
                if photo_url:
                    pn = safe_name(name)
                    # indexを現在のリスト長から取得
                    all_ceos = existing.get("previous_ceos", [])
                    pos = next((i for i, p in enumerate(all_ceos) if p["name"] == name), len(all_ceos)-1)
                    hist_dir = pdir / f"{ticker}_{sn}" / "history" / f"{pos+1:02d}_{pn}"
                    dest = hist_dir / "photo_01.jpg"
                    if download_photo(photo_url, dest):
                        target_ceo["photo_path"] = str(dest.relative_to(PROJECT_DIR))
                        new_jpgs.append(dest)
                        log.info(f"    {name}: 写真取得成功")
                        # info.json
                        info_file = hist_dir / "info.json"
                        info_file.write_text(
                            json.dumps(target_ceo, ensure_ascii=False, indent=2),
                            encoding="utf-8"
                        )

    # 保存
    HISTORY_FILE.write_text(
        json.dumps(history_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log.info(f"history_supplement.json 更新完了 ({updated}件追加)")

    # コミット&プッシュ
    msg = (
        f"manual_dates: {updated}名追加, {len(new_jpgs)}枚写真取得\n\n"
        "極洋・ニッスイ・サカタのタネ等の正確な就任・退任日を反映\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    pushed = commit_and_push(msg)
    if pushed and new_jpgs:
        for p in new_jpgs:
            try:
                p.unlink()
            except Exception:
                pass
        log.info(f"ローカル写真削除: {len(new_jpgs)}枚")

    log.info("=== 完了 ===")


if __name__ == "__main__":
    main()
