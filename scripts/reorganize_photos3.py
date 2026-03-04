#!/usr/bin/env python3
"""
reorganize_photos3.py – photos_1/ photos_2/ を 5分割に再編成

新構造:
  photos_1/: ticker < 3500
  photos_2/: 3500 <= ticker < 5000
  photos_3/: 5000 <= ticker < 7000
  photos_4/: 7000 <= ticker < 9000
  photos_5/: ticker >= 9000 または非数値

git plumbing のみ使用（ファイルシステム操作なし）。
"""

import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
GH_USER   = "yukizi1113"
GH_TOKEN  = __import__("os").environ.get("GH_TOKEN", "")
REPO_NAME = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = (
    f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git"
    if GH_TOKEN else f"https://github.com/{GH_USER}/{REPO_NAME}.git"
)


def photos_dir_for(ticker: str) -> str:
    try:
        t = int(ticker)
        if t < 3500: return "photos_1"
        if t < 5000: return "photos_2"
        if t < 7000: return "photos_3"
        if t < 9000: return "photos_4"
        return "photos_5"
    except ValueError:
        return "photos_5"


def _git(*args, check=False, timeout=300):
    r = subprocess.run(
        ["git", "-c", "core.quotePath=false"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    if check and r.returncode != 0:
        print(f"ERROR: git {' '.join(args[:3])}: {r.stderr[:200]}", file=sys.stderr)
    return r.returncode == 0, r.stdout.strip(), r.stderr.strip()


def main():
    print("=== photos 5分割再編成 開始 ===")

    # 現在HEADにある全ファイルを取得
    ok, out, _ = _git("ls-tree", "-r", "--name-only", "HEAD")
    if not ok:
        print("git ls-tree 失敗", file=sys.stderr)
        sys.exit(1)

    all_files = out.splitlines()

    # photos_1/ と photos_2/ のファイルだけ抽出
    photo_files = [f for f in all_files if f.startswith("photos_1/") or f.startswith("photos_2/")]
    print(f"対象ファイル数: {len(photo_files)}")

    # 各ファイルを正しいディレクトリに分類
    moves: list[tuple[str, str]] = []  # (old_path, new_path)
    for path in photo_files:
        # photos_1/6612_○○/current/photo.jpg → ticker=6612, dir=photos_1
        m = re.match(r"^(photos_[12])/(\d+[A-Za-z]?_[^/]+)(/.*)?$", path)
        if not m:
            continue
        old_dir, company, rest = m.group(1), m.group(2), m.group(3) or ""
        # tickerをcompanyから取得
        ticker_m = re.match(r"^(\d+)", company)
        if not ticker_m:
            # 非数値ticker → photos_5
            new_dir = "photos_5"
        else:
            new_dir = photos_dir_for(ticker_m.group(1))

        if old_dir != new_dir:
            new_path = f"{new_dir}/{company}{rest}"
            moves.append((path, new_path))

    print(f"移動対象: {len(moves)} ファイル")
    if not moves:
        print("移動不要（すでに正しい構造）")
        return

    # ファイルごとのSHAを取得
    print("git ls-tree で SHA 取得中...")
    ok, out, _ = _git("ls-tree", "-r", "--long", "HEAD",
                      "photos_1/", "photos_2/")
    sha_map: dict[str, tuple[str, str]] = {}  # path → (mode, sha)
    for line in out.splitlines():
        # "100644 blob abc123 -\tpath/to/file"
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        meta, fpath = parts
        fields = meta.split()
        if len(fields) >= 3:
            sha_map[fpath] = (fields[0], fields[2])

    # update-index で移動（削除 + 追加）
    print("git update-index で再編成中...")
    batch_size = 500
    batches = [moves[i:i+batch_size] for i in range(0, len(moves), batch_size)]

    for bi, batch in enumerate(batches):
        print(f"  バッチ {bi+1}/{len(batches)}: {len(batch)} ファイル処理中...")

        # 削除
        remove_args = ["update-index", "--remove"] + [p for p, _ in batch]
        _git(*remove_args, check=True)

        # 追加
        for old_path, new_path in batch:
            if old_path not in sha_map:
                print(f"  WARNING: {old_path} のSHA不明、スキップ", file=sys.stderr)
                continue
            mode, sha = sha_map[old_path]
            _git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{new_path}")

    print("git write-tree でツリー作成...")
    ok, tree_sha, _ = _git("write-tree")
    if not ok:
        print("write-tree 失敗", file=sys.stderr)
        sys.exit(1)

    # コミット
    print("git commit-tree でコミット作成...")
    ok, head_sha, _ = _git("rev-parse", "HEAD")
    ok, commit_sha, _ = _git(
        "commit-tree", tree_sha.strip(),
        "-p", head_sha.strip(),
        "-m", "refactor: photos ディレクトリを5分割に再編成\n\n"
              "photos_1(ticker<3500) / photos_2(3500-4999) / photos_3(5000-6999)\n"
              "photos_4(7000-8999) / photos_5(9000+/非数値)\n\n"
              "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>",
    )
    if not ok:
        print("commit-tree 失敗", file=sys.stderr)
        sys.exit(1)

    # HEADを更新
    commit_sha = commit_sha.strip()
    ok, _, err = _git("update-ref", "refs/heads/master", commit_sha)
    if not ok:
        print(f"update-ref 失敗: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"コミット作成: {commit_sha[:8]}")

    # push
    if GH_TOKEN:
        print("GitHub push 中...")
        ok, out, err = _git("push", REMOTE_URL, "master", timeout=600)
        if ok:
            print("push 成功!")
        else:
            print(f"push 失敗: {err[:200]}", file=sys.stderr)
    else:
        print("GH_TOKEN が未設定のため push をスキップ")

    print("=== 完了 ===")


if __name__ == "__main__":
    main()
