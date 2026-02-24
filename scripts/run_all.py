"""
全件収集 + バッチcommit/push + ML dataset生成の自動化スクリプト
GH_TOKEN 環境変数から GitHub Personal Access Token を取得して使用。

Usage:
  GH_TOKEN=ghp_... python scripts/run_all.py
"""

import sys
import os
import json
import subprocess
import time
import logging
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
GH_USER = "yukizi1113"
REPO = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = (
    f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO}.git"
    if GH_TOKEN
    else f"https://github.com/{GH_USER}/{REPO}.git"
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

def git(*args) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=str(PROJECT_DIR),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=120,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except Exception as e:
        return False, str(e)


def commit_and_push(batch: int, done: int, total: int, photos: int) -> bool:
    _, status = git("status", "--porcelain")
    if not status.strip():
        log.info("Nothing new to commit.")
        return True

    git("add", "photos/", "data/", "scripts/", "README.md")

    msg = (
        f"Batch {batch}: {done}/{total}社処理, 写真{photos}枚取得済み\n\n"
        f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M JST')}\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    ok, out = git("commit", "-m", msg)
    if not ok:
        if "nothing to commit" in out.lower():
            return True
        log.warning(f"Commit failed: {out}")
        return False

    for attempt in range(3):
        ok, out = git("push", REMOTE_URL, "master")
        if ok:
            log.info(f"Pushed batch {batch} to GitHub")
            return True
        log.warning(f"Push attempt {attempt+1} failed: {out[:200]}")
        time.sleep(5 * (attempt + 1))

    log.error("Push failed after 3 attempts")
    return False


def delete_pushed_photos(tickers: list[str]) -> int:
    """Delete local photo files after successful GitHub push."""
    deleted = 0
    for ticker in tickers:
        for d in PHOTOS_DIR.glob(f"{ticker}_*"):
            if d.is_dir():
                for photo in d.rglob("photo_*.jpg"):
                    try:
                        photo.unlink()
                        deleted += 1
                    except Exception:
                        pass
    log.info(f"Deleted {deleted} local photos after push")
    return deleted


# ─── Dataset generation ───────────────────────────────────────────────────────

def generate_ml_dataset():
    """Run make_ml_dataset.py to generate ML-ready CSVs."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "make_ml_dataset",
            str(PROJECT_DIR / "scripts" / "make_ml_dataset.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        log.info("ML dataset CSVs generated")
    except Exception as e:
        log.warning(f"ML dataset generation error: {e}")


# ─── Progress I/O ─────────────────────────────────────────────────────────────

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
    log.info("CEO 全件収集スクリプト v4 開始")
    log.info(f"Project: {PROJECT_DIR}")
    log.info(f"GitHub: https://github.com/{GH_USER}/{REPO}")
    log.info("=" * 70)

    # Load collect module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "collect_ceo",
        str(PROJECT_DIR / "scripts" / "collect_ceo.py")
    )
    collect_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect_mod)

    # Load companies
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

    if not todo:
        log.info("全企業の処理が完了しています。")
        generate_ml_dataset()
        return

    BATCH_SIZE = 100
    WORKERS = 4
    total_photos = sum(
        1 for r in results.values()
        if r.get("current_ceo") and r["current_ceo"].get("photo_saved")
    )
    batch_num = len(done_tickers) // BATCH_SIZE

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        batch_num += 1
        log.info(f"\n{'='*60}")
        log.info(f"バッチ {batch_num} / {-(-len(todo)//BATCH_SIZE)}: {len(batch)}社")
        log.info(f"{'='*60}")

        batch_tickers = []
        batch_photos = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(collect_mod.process_company, c): c for c in batch}
            for future in as_completed(futures):
                c = futures[future]
                try:
                    r = future.result(timeout=90)  # 90s hard limit per company
                    results[r["ticker"]] = r
                    done_tickers.add(r["ticker"])
                    batch_tickers.append(r["ticker"])
                    ceo = r.get("current_ceo") or {}
                    n_photos = ceo.get("photo_count", 1 if ceo.get("photo_saved") else 0)
                    if n_photos and n_photos > 0:
                        total_photos += 1
                        batch_photos += int(n_photos)
                        log.info(f"  ✓ [{r['ticker']}] {r['company_name']} | "
                                 f"CEO: {ceo.get('name','?')} | "
                                 f"写真{n_photos}枚 | 就任: {ceo.get('appointment_info','?')}")
                except Exception as e:
                    log.warning(f"  タイムアウト/エラー [{c['ticker']}]: {e}")
                    done_tickers.add(c["ticker"])
                    batch_tickers.append(c["ticker"])

        # Save progress
        save_progress(done_tickers, results)
        log.info(f"\n進捗: {len(done_tickers)}/{len(companies)} | 写真累計: {total_photos}社分")

        # Generate ML dataset CSVs
        generate_ml_dataset()

        # Commit & push to GitHub
        pushed = commit_and_push(batch_num, len(done_tickers), len(companies), total_photos)

        # Delete local photos after successful push
        if pushed:
            delete_pushed_photos(batch_tickers)

        time.sleep(1)

    # Final
    log.info("\n最終処理...")
    generate_ml_dataset()
    commit_and_push(batch_num + 1, len(done_tickers), len(companies), total_photos)

    log.info("\n" + "=" * 70)
    log.info("全件収集完了!")
    with_ceo = sum(1 for r in results.values() if r.get("current_ceo"))
    with_photos = total_photos
    with_prev = sum(1 for r in results.values() if r.get("previous_ceos"))
    log.info(f"CEO情報取得: {with_ceo}/{len(results)}社")
    log.info(f"写真取得: {with_photos}社分")
    log.info(f"前任社長情報: {with_prev}社")
    log.info(f"GitHub: https://github.com/{GH_USER}/{REPO}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
