#!/usr/bin/env python3
"""
run_pipeline.py – collect_history.py 完了後の自動継続パイプライン

1. collect_history.py の完了を待つ
2. retry_photos.py を実行（新規発見CEO写真取得）
3. merge_history.py を実行
4. build_tracking_csv.py を実行
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
VENV_PYTHON = Path(r"C:\Users\hp\Documents\JP_Valuation_EDINET\.venv\Scripts\python.exe")
LOG_FILE    = PROJECT_DIR / "pipeline.log"
GH_TOKEN    = os.environ.get("GH_TOKEN", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def is_collect_running() -> bool:
    import ctypes
    r = subprocess.run(
        ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return "collect_history.py" in r.stdout


def run_script(script: str, extra_env: dict | None = None) -> bool:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    log.info(f"▶ {script} 実行開始")
    r = subprocess.run(
        [str(VENV_PYTHON), f"scripts/{script}"],
        cwd=str(PROJECT_DIR),
        env=env,
        encoding="utf-8", errors="replace",
        timeout=7200,
    )
    if r.returncode == 0:
        log.info(f"✅ {script} 完了")
        return True
    else:
        log.error(f"❌ {script} 失敗 (exit={r.returncode})")
        return False


def wait_for_collect_history():
    """collect_history.py が終わるまで待機"""
    log.info("collect_history.py の完了を待機中...")
    while is_collect_running():
        # 進捗確認
        try:
            prog = json.loads((PROJECT_DIR / "data/history_progress.json").read_text(encoding="utf-8"))
            log.info(f"  進捗: {len(prog)}/3727 社処理済み")
        except Exception:
            pass
        time.sleep(120)  # 2分ごとに確認
    log.info("collect_history.py 完了を検出")


def main():
    log.info("=== run_pipeline.py 開始 ===")
    env = {"GH_TOKEN": GH_TOKEN} if GH_TOKEN else {}

    # 1. collect_history.py 完了待ち
    wait_for_collect_history()

    # 少し待って確実に終了させる
    time.sleep(10)

    # 2. 進捗確認
    try:
        prog = json.loads((PROJECT_DIR / "data/history_progress.json").read_text(encoding="utf-8"))
        supplement = json.loads((PROJECT_DIR / "data/history_supplement.json").read_text(encoding="utf-8"))
        with_hist = sum(1 for c in supplement if c.get("previous_ceos"))
        log.info(f"collect_history 結果: {len(prog)}社処理済み, 歴代データあり: {with_hist}社")
    except Exception as e:
        log.warning(f"進捗確認失敗: {e}")

    # 3. retry_photos.py（歴代CEO写真取得）
    log.info("=== retry_photos.py 実行 ===")
    run_script("retry_photos.py", env)

    # 4. merge_history.py
    log.info("=== merge_history.py 実行 ===")
    run_script("merge_history.py", env)

    # 5. build_tracking_csv.py
    log.info("=== build_tracking_csv.py 実行 ===")
    run_script("build_tracking_csv.py", env)

    log.info("=== パイプライン完了 ===")


if __name__ == "__main__":
    main()
