"""
歴代社長データ統合スクリプト
history_supplement.json の内容を ceo_data.json にマージし、
ML dataset CSV を再生成、GitHub にコミット・プッシュする。

Usage:
  GH_TOKEN=ghp_... python scripts/merge_history.py
"""

import json, os, re, subprocess, sys, time, logging
from pathlib import Path
from datetime import datetime

PROJECT_DIR  = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR     = PROJECT_DIR / "data"
CEO_DATA_FILE    = DATA_DIR / "ceo_data.json"
HISTORY_OUT      = DATA_DIR / "history_supplement.json"
HISTORY_PROGRESS = DATA_DIR / "history_progress.json"
LOG_FILE         = PROJECT_DIR / "merge_history.log"

GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_USER  = "yukizi1113"
REPO     = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = (
    f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO}.git"
    if GH_TOKEN else f"https://github.com/{GH_USER}/{REPO}.git"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


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


def commit_and_push(done: int, total: int, prev_count: int) -> bool:
    _, status = git("status", "--porcelain")
    if not status.strip():
        log.info("Nothing to commit.")
        return True

    git("add", "data/", "photos/", "scripts/", "README.md")
    msg = (
        f"History supplement: {done}/{total}社, 歴代社長{prev_count}名分\n\n"
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
            log.info("Pushed to GitHub")
            return True
        log.warning(f"Push attempt {attempt+1} failed: {out[:200]}")
        time.sleep(5 * (attempt + 1))

    log.error("Push failed after 3 attempts")
    return False


def merge():
    log.info("="*70)
    log.info("歴代社長データ統合開始")
    log.info("="*70)

    if not CEO_DATA_FILE.exists():
        log.error("ceo_data.json が見つかりません。")
        sys.exit(1)

    with open(CEO_DATA_FILE, encoding="utf-8") as f:
        ceo_data = json.load(f)
    log.info(f"ceo_data.json: {len(ceo_data)}社")

    # Build lookup by ticker
    ceo_by_ticker = {c.get("ticker", ""): c for c in ceo_data}

    if not HISTORY_OUT.exists():
        log.warning("history_supplement.json が見つかりません。収集完了後に実行してください。")
        sys.exit(1)

    with open(HISTORY_OUT, encoding="utf-8") as f:
        history_data = json.load(f)
    log.info(f"history_supplement.json: {len(history_data)}社")

    merged_count = 0
    prev_total = 0

    for hist in history_data:
        ticker = hist.get("ticker", "")
        prev_ceos = hist.get("previous_ceos", [])
        if not ticker or not prev_ceos:
            continue

        if ticker in ceo_by_ticker:
            existing = ceo_by_ticker[ticker].get("previous_ceos") or []
            # Merge: add entries not already present by name
            existing_names = {e.get("name", "") for e in existing}
            added = 0
            for p in prev_ceos:
                if p.get("name") and p["name"] not in existing_names:
                    existing.append(p)
                    existing_names.add(p["name"])
                    added += 1
            ceo_by_ticker[ticker]["previous_ceos"] = existing
            if added > 0:
                merged_count += 1
                prev_total += added
        else:
            # Company not in main data – add it
            ceo_by_ticker[ticker] = {
                "ticker": ticker,
                "company_name": hist.get("company_name", ""),
                "previous_ceos": prev_ceos,
            }
            merged_count += 1
            prev_total += len(prev_ceos)

    log.info(f"統合: {merged_count}社に歴代社長情報を追加 (計{prev_total}名)")

    updated_list = list(ceo_by_ticker.values())
    with open(CEO_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_list, f, ensure_ascii=False, indent=2)
    log.info(f"ceo_data.json 更新完了: {len(updated_list)}社")

    # Regenerate ML dataset
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_ml_dataset",
        str(PROJECT_DIR / "scripts" / "make_ml_dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stats = mod.main()
    log.info("ML dataset CSV 再生成完了")

    # Commit and push
    done = len(history_data)
    commit_and_push(done, len(ceo_data), prev_total)

    log.info("="*70)
    log.info("統合完了!")
    log.info("="*70)


if __name__ == "__main__":
    merge()
