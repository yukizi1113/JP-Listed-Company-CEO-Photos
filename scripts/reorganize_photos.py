#!/usr/bin/env python3
"""
reorganize_photos.py  v2
photos/ を photos_1/ (ticker<5500) と photos_2/ (ticker>=5500) に分割する。

git update-index を使うため、ローカルにファイルが存在しなくてもよい。
バッチコミット＆プッシュ後、ローカルjpg削除（すでに削除済みの場合はスキップ）。
"""
import os
import re
import subprocess
import time
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")

GH_TOKEN  = os.environ.get("GH_TOKEN", "")
GH_USER   = "yukizi1113"
REPO_NAME = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"

BATCH_SIZE = 200


_GIT_ENV = {**os.environ, "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.quotePath",
            "GIT_CONFIG_VALUE_0": "false"}


def _git(*args, timeout=300):
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout,
        env=_GIT_ENV,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def target_subdir(dir_name: str) -> str:
    """ティッカーに基づき 'photos_1' または 'photos_2' を返す。"""
    m = re.match(r"^(\d+)", dir_name)
    if m and int(m.group(1)) < 5500:
        return "photos_1"
    return "photos_2"


def index_move_dir(old_prefix: str, new_prefix: str) -> int:
    """
    git インデックス内の old_prefix/* を new_prefix/* へ移動する。
    ローカルファイルの存在は不要 (git update-index を直接操作)。
    戻り値: 移動したファイル数
    """
    # ls-files -s で mode/sha/stage/path を取得 (quotePath=false で日本語をそのまま出力)
    r = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files", "-s", "--", old_prefix + "/"],
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if not lines:
        return 0

    moved = 0
    for line in lines:
        # フォーマット: "100644 SHA STAGE\tPATH"
        tab = line.index("\t")
        meta = line[:tab].split()
        old_path = line[tab + 1:]
        mode, sha = meta[0], meta[1]

        new_path = new_prefix + old_path[len(old_prefix):]

        # 新パスにエントリを追加
        proc = subprocess.run(
            ["git", "update-index", "--add", "--cacheinfo",
             f"{mode},{sha},{new_path}"],
            cwd=str(PROJECT_DIR),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if proc.returncode == 0:
            # 旧パスを削除
            subprocess.run(
                ["git", "-c", "core.quotePath=false",
                 "rm", "--cached", "-q", "--", old_path],
                cwd=str(PROJECT_DIR),
                capture_output=True,
            )
            moved += 1

    return moved


def commit_and_push(msg: str) -> bool:
    ok, out = _git("commit", "-m", msg)
    if not ok and "nothing to commit" not in out.lower():
        print(f"  commit 失敗: {out[:150]}")
        return False
    for attempt in range(3):
        ok, out = _git("push", REMOTE_URL, "master", timeout=360)
        if ok:
            return True
        print(f"  push 失敗 {attempt+1}/3: {out[:100]}")
        time.sleep(20)
    return False


def delete_local_jpgs(directory: Path) -> int:
    count = 0
    if not directory.exists():
        return 0
    for f in directory.rglob("*.jpg"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count


def main():
    # git で追跡されている photos/ 直下のディレクトリ一覧を取得 (quotePath=false)
    r = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-tree", "--name-only", "HEAD", "photos/"],
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    all_dirs = sorted([
        d.strip().removeprefix("photos/")
        for d in r.stdout.splitlines() if d.strip()
    ])
    print(f"photos/ 内のディレクトリ数 (HEAD): {len(all_dirs)}")

    # すでに photos_1/ or photos_2/ に移動済みのものをスキップ
    todo = []
    for d in all_dirs:
        dest = target_subdir(d)
        # インデックスまたは HEAD に新パスが存在するか確認
        r2 = subprocess.run(
            ["git", "-c", "core.quotePath=false", "ls-files", "-s", "--", f"{dest}/{d}/"],
            cwd=str(PROJECT_DIR),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        if r2.stdout.strip():
            continue  # 移動済み
        todo.append(d)

    print(f"移動残り: {len(todo)} ディレクトリ")

    batch_moved = 0
    batch_num   = 2   # batch 1 は前回実行済み
    batch_dirs  = []
    total_moved = 0

    for i, dir_name in enumerate(todo, 1):
        dest_sub = target_subdir(dir_name)
        old_prefix = f"photos/{dir_name}"
        new_prefix = f"{dest_sub}/{dir_name}"

        n = index_move_dir(old_prefix, new_prefix)
        if n:
            print(f"  [{i}/{len(todo)}] {dir_name} -> {dest_sub}/ ({n} files)")
            batch_moved += n
            batch_dirs.append(dir_name)
            total_moved += n
        else:
            print(f"  [{i}/{len(todo)}] スキップ（追跡ファイルなし）: {dir_name}")

        is_batch_end = (len(batch_dirs) >= BATCH_SIZE) or (i == len(todo))
        if is_batch_end and batch_dirs:
            msg = (
                f"refactor: photos/->photos_1,2 batch {batch_num} ({len(batch_dirs)}社, {batch_moved}files)\n\n"
                "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
            )
            pushed = commit_and_push(msg)
            if pushed:
                print(f"  push 完了 batch {batch_num} ({len(batch_dirs)}社)")
                # ローカルjpg削除（すでに削除済みの場合は0枚）
                deleted = sum(
                    delete_local_jpgs(PROJECT_DIR / target_subdir(d) / d)
                    for d in batch_dirs
                ) + sum(
                    delete_local_jpgs(PROJECT_DIR / "photos" / d)
                    for d in batch_dirs
                )
                print(f"  ローカルjpg削除: {deleted} 枚")
            batch_num  += 1
            batch_moved = 0
            batch_dirs  = []
            time.sleep(3)

    print(f"\n合計移動ファイル数: {total_moved}")

    # photos/ が git インデックスから空になったか確認
    r = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files", "-s", "--", "photos/"],
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    remaining_files = r.stdout.strip()
    if not remaining_files:
        # ローカルの空ディレクトリを削除
        photos_local = PROJECT_DIR / "photos"
        if photos_local.exists():
            for sub in list(photos_local.rglob("*")):
                if sub.is_file():
                    try: sub.unlink()
                    except: pass
            for sub in sorted(photos_local.rglob("*"), reverse=True):
                if sub.is_dir():
                    try: sub.rmdir()
                    except: pass
            try: photos_local.rmdir()
            except: pass
        print("photos/ ディレクトリを削除しました")
        ok, _ = _git("commit", "-m",
            "chore: 空の photos/ を削除\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>")
        if ok:
            _git("push", REMOTE_URL, "master")
    else:
        remaining_count = len(remaining_files.splitlines())
        print(f"photos/ に残り {remaining_count} ファイルあり（要手動確認）")

    print("=== reorganize_photos.py 完了 ===")


if __name__ == "__main__":
    main()
