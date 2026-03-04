"""
git履歴からCEO写真を復元し、GitHubにコミット・プッシュする。

photos/が git add photos/ で誤って削除された問題を修正します。
"""

import subprocess, os, sys, time, logging
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
PHOTOS_DIR  = PROJECT_DIR / "photos"
LOG_FILE    = PROJECT_DIR / "restore_photos.log"

GH_TOKEN   = os.environ.get("GH_TOKEN", "")
GH_USER    = "yukizi1113"
REPO       = "JP-Listed-Company-CEO-Photos"
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


def run_git(*args) -> tuple[bool, bytes]:
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True,
        timeout=300,
    )
    return r.returncode == 0, r.stdout


def run_git_text(*args) -> tuple[bool, str]:
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=300,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def decode_git_path(raw: str) -> str:
    """
    Decode a git-quoted path.
    Git quotes non-ASCII paths as "path\\343\\203\\213..." (octal UTF-8 bytes).
    This decodes them back to the actual Unicode string.
    """
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    result = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == '\\' and i + 3 < len(raw) and all(c in '01234567' for c in raw[i+1:i+4]):
            result.append(int(raw[i+1:i+4], 8))
            i += 4
        else:
            result.extend(raw[i].encode('ascii', errors='replace'))
            i += 1
    try:
        return result.decode('utf-8')
    except Exception:
        return raw


def restore_photos():
    log.info("="*70)
    log.info("CEO写真 git履歴からの復元スクリプト")
    log.info("="*70)

    # Step 1: Find all jpg files ever added (not just currently present)
    log.info("Step 1: git履歴からjpgファイル一覧を取得...")
    ok, output = run_git_text(
        "log", "--all", "--diff-filter=A",
        "--pretty=format:%H",
        "--name-only",
        "--", "*.jpg"
    )
    if not ok:
        log.error("git log failed")
        return

    # Parse: alternating commit hash and file paths
    # git log goes newest-first; we want oldest commit per file (original add with content)
    # so we OVERWRITE to keep the last (oldest) commit seen for each file
    lines = output.strip().split("\n")
    file_to_oldest_commit: dict[str, str] = {}
    current_commit = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
            current_commit = line
        elif current_commit and (".jpg" in line):
            # Decode git-quoted path (handles Japanese/non-ASCII paths)
            fpath = decode_git_path(line)
            # Always overwrite → keeps oldest commit (last in output = oldest in time)
            file_to_oldest_commit[fpath] = current_commit

    # Filter: only photo_NN.jpg files (skip old photo.jpg format)
    photo_files = {p: c for p, c in file_to_oldest_commit.items()
                   if "photo_0" in p and p.endswith(".jpg")}

    log.info(f"  発見: {len(photo_files)}個のphoto_NN.jpgファイル (履歴全体)")

    # Step 2: Restore files that don't currently exist locally
    restored = 0
    skipped = 0
    failed = 0

    for photo_path, commit in sorted(photo_files.items()):
        local = PHOTOS_DIR.parent / photo_path
        if local.exists() and local.stat().st_size > 1000:
            skipped += 1
            continue

        # git show uses the path as-is (Unicode). Pass as bytes on Windows to avoid encoding issues.
        ok, content = run_git("show", f"{commit}:{photo_path}")
        if not ok or len(content) < 1000:
            log.debug(f"  skip (empty/failed): {photo_path}")
            failed += 1
            continue

        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        restored += 1
        if restored % 100 == 0:
            log.info(f"  復元済み: {restored}枚 (失敗:{failed})")

    log.info(f"復元完了: {restored}枚 (スキップ:{skipped}, 失敗:{failed})")

    if restored == 0:
        log.info("新たに復元するファイルがありません。終了します。")
        return

    # Step 3: Add restored photos (without staging deletions)
    log.info("Step 3: 復元したファイルをgit addします...")
    # git add individual files to avoid staging deletions
    # Collect all restored jpg paths
    jpg_paths = []
    for photo_path in photo_files.keys():
        local = PHOTOS_DIR.parent / photo_path
        if local.exists() and local.stat().st_size > 1000:
            jpg_paths.append(photo_path)

    log.info(f"  git addするファイル数: {len(jpg_paths)}")

    # Add in batches to avoid command line length limits
    CHUNK = 200
    for i in range(0, len(jpg_paths), CHUNK):
        chunk = jpg_paths[i:i+CHUNK]
        ok, msg = run_git_text("add", "--", *chunk)
        if not ok:
            log.warning(f"  git add失敗 chunk {i//CHUNK}: {msg[:100]}")

    # Step 4: Also add data/ to make sure CSVs are up to date
    run_git_text("add", "data/")

    # Step 5: Commit
    log.info("Step 4: コミット...")
    _, status = run_git_text("status", "--porcelain")
    if not status.strip():
        log.info("Nothing to commit.")
        return

    msg = (
        f"Restore {restored} CEO photos from git history\n\n"
        f"修正: batch commit後のgit add photos/による写真削除を復元\n"
        f"処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M JST')}\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    ok, out = run_git_text("commit", "-m", msg)
    if not ok:
        if "nothing to commit" in out.lower():
            log.info("Nothing to commit.")
            return
        log.error(f"Commit failed: {out}")
        return

    # Step 6: Push
    log.info("Step 5: GitHubにプッシュ...")
    for attempt in range(3):
        ok, out = run_git_text("push", REMOTE_URL, "master")
        if ok:
            log.info(f"プッシュ成功!")
            break
        log.warning(f"Push attempt {attempt+1}: {out[:200]}")
        time.sleep(5 * (attempt + 1))
    else:
        log.error("Push failed after 3 attempts")

    log.info("="*70)
    log.info("写真復元完了!")
    log.info("="*70)


if __name__ == "__main__":
    restore_photos()
