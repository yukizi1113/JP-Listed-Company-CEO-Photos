#!/usr/bin/env python3
"""
collect_history_photos.py  –  歴代CEO写真の一括収集

対象: history_supplement.json の previous_ceos で photo_path が空のエントリ
手順: 1) Wikipedia人物ページ → infobox画像
      2) Bing画像検索 (名前 + 会社名 + 社長)
      3) 50件ごとにコミット&プッシュ&ローカル削除
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR    = PROJECT_DIR / "data"
LOG_FILE    = PROJECT_DIR / "collect_history_photos.log"

GH_TOKEN = os.environ.get("GH_TOKEN", "")

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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ja-JP,ja;q=0.9",
})

_BAD_IMG_WORDS = [
    "svg", "disambig", "question", "noimage", "flag", "logo",
    "icon", "blank", "silhouette", "shadow", "nophoto",
]

_JP_CHARS = re.compile(r'^[\u4e00-\u9fff\u3040-\u30ff]+$')


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


def _is_bad_img(url: str) -> bool:
    u = url.lower()
    return any(b in u for b in _BAD_IMG_WORDS)


def wiki_portrait(name: str) -> str | None:
    """Wikipedia人物ページのinfobox画像URLを取得。"""
    try:
        r = SESSION.get(
            f"https://ja.wikipedia.org/wiki/{quote(name)}",
            timeout=12
        )
        if r.status_code != 200:
            return None
        sp = BeautifulSoup(r.text, "lxml")
        # 人物ページか確認（カテゴリに「人物」が含まれるか）
        cats = sp.get_text()
        if not re.search(r'(政治家|実業家|経営者|企業人|官僚|社長|会長|取締役)', cats[:5000]):
            # Wikipedia disambiguation check
            h1 = sp.find("h1", {"id": "firstHeading"})
            if h1 and "曖昧さ回避" in sp.get_text()[:2000]:
                return None
        infobox = sp.find("table", {"class": lambda c: c and "infobox" in " ".join(c)})
        img = infobox.find("img") if infobox else None
        if not img:
            img = sp.find("img", src=re.compile(r"upload\.wikimedia"))
        if img and img.get("src"):
            src = img["src"]
            if src.startswith("//"):
                src = "https:" + src
            if not _is_bad_img(src):
                return src
    except Exception:
        pass
    return None


def bing_image(name: str, company: str) -> str | None:
    """Bing画像検索でCEO写真URLを取得。"""
    name_nospace = re.sub(r'\s+', '', name)
    queries = [
        f"{name_nospace} {company} 社長",
        f"{name_nospace} 代表取締役",
        f"{name_nospace} 経営者",
    ]
    for q in queries:
        try:
            r = SESSION.get(
                "https://www.bing.com/images/search",
                params={"q": q, "first": 1, "mkt": "ja-JP"},
                headers={"User-Agent": SESSION.headers["User-Agent"]},
                timeout=12
            )
            if r.status_code != 200:
                continue
            # HTML-encoded murl パターン
            urls = re.findall(
                r'&quot;murl&quot;:&quot;(https?://[^&]+\.(?:jpg|jpeg|png))&quot;',
                r.text
            )
            if not urls:
                urls = re.findall(
                    r'"murl"\s*:\s*"(https?://[^"]+\.(?:jpg|jpeg|png))"',
                    r.text
                )
            filtered = [u for u in urls if not _is_bad_img(u)]
            if filtered:
                return filtered[0]
            time.sleep(0.5)
        except Exception:
            pass
    return None


def download_photo(url: str, dest: Path) -> bool:
    try:
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200 or len(r.content) < 2000:
            return False
        # 画像ヘッダチェック（JPEG/PNG）
        if not (r.content[:2] == b'\xff\xd8' or r.content[:8] == b'\x89PNG\r\n\x1a\n'):
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def get_done_set() -> set[tuple[str, str]]:
    """GitHubで既に写真があるCEOの (ticker, name) セットを返す。"""
    r = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files",
         "photos_1/", "photos_2/", "photos_3/", "photos_4/", "photos_5/"],
        capture_output=True, encoding="utf-8", errors="replace",
        cwd=str(PROJECT_DIR)
    )
    done = set()
    for line in r.stdout.splitlines():
        # photos_X/TICKER_COMPANY/history/NN_NAME/photo_01.jpg
        m = re.match(r"photos_\d+/([^/]+)/history/[^/]+_([^/]+)/photo", line)
        if m:
            ticker = m.group(1).split("_")[0]
            done.add((ticker, m.group(2)))
    return done


def commit_and_push(batch_num: int) -> None:
    add_paths = ["data/history_supplement.json"]
    for d in ["photos_1", "photos_2", "photos_3", "photos_4", "photos_5"]:
        if (PROJECT_DIR / d).exists():
            add_paths.append(f"{d}/")
    r = subprocess.run(
        ["git", "-c", "core.quotePath=false", "add", "--ignore-removal"] + add_paths,
        cwd=str(PROJECT_DIR), capture_output=True
    )
    msg = f"[hist-photos] batch {batch_num}: 歴代CEO写真追加"
    r = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=str(PROJECT_DIR), capture_output=True
    )
    if r.returncode != 0:
        log.warning("コミット対象なし")
        return
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(PROJECT_DIR), capture_output=True
    ).stdout.decode().strip()
    subprocess.run(["git", "push", remote, "master"],
                   cwd=str(PROJECT_DIR), capture_output=True)
    log.info(f"プッシュ完了 batch {batch_num}")
    for d in ["photos_1", "photos_2", "photos_3", "photos_4", "photos_5"]:
        p = PROJECT_DIR / d
        if p.exists():
            import shutil
            shutil.rmtree(p)


def main():
    hs = json.loads((DATA_DIR / "history_supplement.json").read_text(encoding="utf-8"))
    log.info(f"歴代CEO総数: {sum(len(x.get('previous_ceos', [])) for x in hs)}名")

    # 写真なしリスト
    targets = []
    for company in hs:
        ticker = str(company["ticker"])
        cname = company.get("company_name", "")
        for i, ceo in enumerate(company.get("previous_ceos", [])):
            if not ceo.get("photo_path") and ceo.get("name"):
                targets.append((ticker, cname, i, ceo["name"]))

    log.info(f"写真なし歴代CEO: {len(targets)}名")

    BATCH = 50
    batch_count = 0
    success_count = 0

    for ci, (ticker, cname, ceo_idx, name) in enumerate(targets, 1):
        # 写真検索
        photo_url = wiki_portrait(name)
        if not photo_url:
            photo_url = bing_image(name, cname)
            time.sleep(0.3)

        if not photo_url:
            log.debug(f"[{ci}/{len(targets)}] {name} ({cname}): 写真なし")
            continue

        # 保存先
        pdir = photos_dir(ticker)
        folder = pdir / f"{ticker}_{safe_name(cname)}" / "history" / f"{ceo_idx+1:02d}_{safe_name(name)}"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / "photo_01.jpg"

        if download_photo(photo_url, dest):
            # JSON更新
            for company in hs:
                if str(company["ticker"]) == ticker:
                    company["previous_ceos"][ceo_idx]["photo_path"] = str(dest.relative_to(PROJECT_DIR))
                    break
            success_count += 1
            batch_count += 1
            log.info(f"[{ci}/{len(targets)}] {name} ({ticker} {cname}): 写真取得")
        else:
            log.debug(f"[{ci}/{len(targets)}] {name}: DL失敗 {photo_url[:60]}")

        # データ保存
        if ci % 10 == 0:
            (DATA_DIR / "history_supplement.json").write_text(
                json.dumps(hs, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        # バッチコミット
        if batch_count >= BATCH:
            (DATA_DIR / "history_supplement.json").write_text(
                json.dumps(hs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            commit_and_push(ci // BATCH)
            batch_count = 0

        time.sleep(0.2)

    # 最終保存
    (DATA_DIR / "history_supplement.json").write_text(
        json.dumps(hs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if batch_count > 0:
        commit_and_push("final")

    log.info(f"完了: {success_count}/{len(targets)}名の写真取得")


if __name__ == "__main__":
    main()
