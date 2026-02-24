"""
全件収集 + バッチcommit/push + ローカル写真削除の自動化スクリプト
GitHubへのpush完了後、ローカルの写真ファイルを削除してディスク節約。
"""

import sys
import os
import json
import subprocess
import time
import logging
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
PHOTOS_DIR = PROJECT_DIR / "photos"
DATA_DIR = PROJECT_DIR / "data"
COMPANIES_FILE = DATA_DIR / "companies.json"
CEO_DATA_FILE = DATA_DIR / "ceo_data.json"
PROGRESS_FILE = DATA_DIR / "progress.json"
LOG_FILE = PROJECT_DIR / "run_all.log"

GH_TOKEN = os.environ.get("GH_TOKEN", "")
REMOTE_URL = (
    f"https://yukizi1113:{GH_TOKEN}@github.com/yukizi1113/JP-Listed-Company-CEO-Photos.git"
    if GH_TOKEN else "https://github.com/yukizi1113/JP-Listed-Company-CEO-Photos.git"
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─── Git helpers ──────────────────────────────────────────────────────────────

def git_run(*args, cwd=PROJECT_DIR) -> tuple[bool, str]:
    """Run a git command, return (success, output)."""
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        ok = r.returncode == 0
        out = (r.stdout + r.stderr).strip()
        if not ok:
            log.debug(f"git {' '.join(args)} failed: {out}")
        return ok, out
    except Exception as e:
        log.error(f"git {' '.join(args)} exception: {e}")
        return False, str(e)


def commit_and_push(batch_num: int, company_count: int, photo_count: int) -> bool:
    """Add all changes, commit, push to GitHub. Return True on success."""
    # Check if there's anything to commit
    ok, status = git_run("status", "--porcelain")
    if not status.strip():
        log.info("Nothing to commit.")
        return True

    # Add all changes (photos, data files)
    git_run("add", "photos/", "data/", "scripts/", "README.md")

    msg = (
        f"Batch {batch_num}: {company_count}社処理完了, 写真{photo_count}枚取得\n\n"
        f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    ok, out = git_run("commit", "-m", msg)
    if not ok and "nothing to commit" in out.lower():
        log.info("Nothing to commit.")
        return True
    if not ok:
        log.warning(f"Commit failed: {out}")
        return False

    # Push with token-embedded URL
    ok, out = git_run("push", REMOTE_URL, "master", "--force-with-lease")
    if ok:
        log.info(f"Pushed batch {batch_num} to GitHub")
        return True
    else:
        # Retry once
        time.sleep(5)
        ok, out = git_run("push", REMOTE_URL, "master")
        if ok:
            log.info(f"Pushed batch {batch_num} to GitHub (retry)")
            return True
        log.error(f"Push failed: {out}")
        return False


def delete_local_photos(pushed_tickers: list[str]):
    """Delete local photo files for companies already pushed to GitHub."""
    deleted = 0
    for ticker in pushed_tickers:
        # Find directories matching this ticker
        for d in PHOTOS_DIR.glob(f"{ticker}_*"):
            if d.is_dir():
                for photo in d.rglob("photo.jpg"):
                    try:
                        photo.unlink()
                        deleted += 1
                    except Exception as e:
                        log.debug(f"Delete error {photo}: {e}")
    log.info(f"Deleted {deleted} local photos (pushed to GitHub)")


# ─── Collection ──────────────────────────────────────────────────────────────

def load_module():
    """Load the collect_ceo module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "collect_ceo",
        str(PROJECT_DIR / "scripts" / "collect_ceo.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(done: set, results: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done), f)
    with open(CEO_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("全件収集スクリプト 開始")
    log.info(f"Project: {PROJECT_DIR}")
    log.info("=" * 70)

    mod = load_module()

    with open(COMPANIES_FILE, encoding="utf-8") as f:
        companies = json.load(f)
    log.info(f"対象企業数: {len(companies)}")

    # Load existing results
    results = {}
    if CEO_DATA_FILE.exists():
        with open(CEO_DATA_FILE, encoding="utf-8") as f:
            for r in json.load(f):
                results[r["ticker"]] = r

    done_tickers = load_progress()
    log.info(f"処理済み: {len(done_tickers)}")

    todo = [c for c in companies if c["ticker"] not in done_tickers]
    log.info(f"残り: {len(todo)}")

    BATCH_SIZE = 100   # Companies per batch before committing to GitHub
    WORKERS = 4        # Parallel workers

    total_photos = sum(
        1 for r in results.values()
        if r.get("current_ceo") and r["current_ceo"].get("photo_saved")
    )
    batch_num = len(done_tickers) // BATCH_SIZE

    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start:batch_start + BATCH_SIZE]
        batch_num += 1
        log.info(f"\n{'='*50}")
        log.info(f"バッチ {batch_num}: {len(batch)}社処理中")
        log.info(f"{'='*50}")

        batch_tickers = []
        batch_photos = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(mod.process_company, c): c for c in batch}
            for future in as_completed(futures):
                c = futures[future]
                try:
                    r = future.result(timeout=180)
                    results[r["ticker"]] = r
                    done_tickers.add(r["ticker"])
                    batch_tickers.append(r["ticker"])
                    if r.get("current_ceo") and r["current_ceo"].get("photo_saved"):
                        total_photos += 1
                        batch_photos += 1
                        log.info(f"  写真保存: [{r['ticker']}] {r['company_name']} - {r['current_ceo'].get('name','?')}")
                except Exception as e:
                    log.error(f"エラー {c['ticker']}: {e}")
                    done_tickers.add(c["ticker"])
                    batch_tickers.append(c["ticker"])

        # Save progress
        save_progress(done_tickers, results)
        log.info(f"バッチ{batch_num}完了: {len(batch)}社 | 写真{batch_photos}枚 | 累計{len(done_tickers)}/{len(companies)}")

        # Commit and push to GitHub
        pushed = commit_and_push(batch_num, len(done_tickers), total_photos)

        # Delete local photos after successful push
        if pushed:
            delete_local_photos(batch_tickers)

        # Brief pause between batches
        time.sleep(2)

    # Final commit
    log.info("\n最終コミット...")
    commit_and_push(batch_num + 1, len(done_tickers), total_photos)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("全件収集完了!")
    with_ceo = sum(1 for r in results.values() if r.get("current_ceo"))
    with_photos = sum(1 for r in results.values()
                      if r.get("current_ceo") and r["current_ceo"].get("photo_saved"))
    with_prev = sum(1 for r in results.values() if r.get("previous_ceos"))
    log.info(f"CEO情報取得: {with_ceo}/{len(results)}社")
    log.info(f"写真取得数: {with_photos}枚")
    log.info(f"前任社長情報あり: {with_prev}社")
    log.info(f"GitHub: https://github.com/yukizi1113/JP-Listed-Company-CEO-Photos")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
