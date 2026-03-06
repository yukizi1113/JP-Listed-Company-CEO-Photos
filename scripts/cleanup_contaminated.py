"""
汚染された歴代CEOデータのクリーンアップ

問題: collect_history.py の Wikipedia Strategy 2 (検索API) でページ検証なしに
      有名企業の歴代社長テーブルを別会社に誤帰属させた。

対策:
  1. 3社以上に出現する名前 = 汚染データ → 削除
  2. 2社出現でどちらも日付なし = 汚染データ → 削除
  3. クリーンなデータのみ残す
  4. 汚染済み写真パスを一覧化 → Gitで削除
"""
import json, sys, subprocess, re
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR = PROJECT_DIR / "data"

# ── データ読み込み ──────────────────────────────────────────────────────
hs = json.loads((DATA_DIR / "history_supplement.json").read_text(encoding="utf-8"))

# ── 名前→出現会社リスト ─────────────────────────────────────────────────
name_to_tickers = defaultdict(list)
for company in hs:
    ticker = str(company["ticker"])
    for ceo in company.get("previous_ceos", []):
        name = ceo.get("name", "")
        if name:
            name_to_tickers[name].append(ticker)

# ── 汚染判定 ───────────────────────────────────────────────────────────
def is_contaminated(name: str, ceo_entry: dict) -> bool:
    tickers = name_to_tickers[name]
    count = len(tickers)
    if count >= 3:
        return True  # 3社以上 → 汚染確定
    if count == 2:
        # 両エントリとも日付なし → 汚染
        appt = ceo_entry.get("appointment_date")
        rsgn = ceo_entry.get("resignation_date")
        if not appt and not rsgn:
            return True
    return False

# ── クリーンアップ実行 ────────────────────────────────────────────────
contaminated_photos = []  # 削除すべき写真パス
total_before = 0
total_after = 0
companies_emptied = 0
companies_reduced = 0

new_hs = []
for company in hs:
    ticker = str(company["ticker"])
    prev = company.get("previous_ceos", [])
    total_before += len(prev)

    clean_ceos = []
    removed_ceos = []
    for ceo in prev:
        name = ceo.get("name", "")
        if not name or is_contaminated(name, ceo):
            removed_ceos.append(ceo)
            photo = ceo.get("photo_path", "")
            if photo:
                contaminated_photos.append(photo)
        else:
            clean_ceos.append(ceo)

    total_after += len(clean_ceos)

    if removed_ceos and len(clean_ceos) == 0:
        companies_emptied += 1
    elif removed_ceos:
        companies_reduced += 1

    # 更新
    new_company = dict(company)
    new_company["previous_ceos"] = clean_ceos
    new_hs.append(new_company)

print(f"=== クリーンアップ結果 ===")
print(f"クリーンアップ前 CEO 数: {total_before}")
print(f"クリーンアップ後 CEO 数: {total_after}")
print(f"削除された CEO 数: {total_before - total_after}")
print(f"  完全に空になった会社: {companies_emptied}社")
print(f"  一部削除された会社: {companies_reduced}社")
print(f"削除対象写真: {len(contaminated_photos)}件")

# ── バックアップ作成 ───────────────────────────────────────────────────
backup = DATA_DIR / "history_supplement_backup.json"
backup.write_text(
    json.dumps(hs, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"\nバックアップ保存: {backup}")

# ── クリーン版保存 ────────────────────────────────────────────────────
(DATA_DIR / "history_supplement.json").write_text(
    json.dumps(new_hs, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"クリーン版保存: {DATA_DIR / 'history_supplement.json'}")

# ── 汚染写真パスを保存 ────────────────────────────────────────────────
contaminated_path = DATA_DIR / "contaminated_photos.json"
contaminated_path.write_text(
    json.dumps(contaminated_photos, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"汚染写真リスト保存: {contaminated_path} ({len(contaminated_photos)}件)")

# ── 空になった会社リスト（再収集対象）────────────────────────────────
emptied_companies = []
for company in new_hs:
    if not company.get("previous_ceos"):
        emptied_companies.append({
            "ticker": company["ticker"],
            "company_name": company.get("company_name", "")
        })

recollect_path = DATA_DIR / "recollect_targets.json"
recollect_path.write_text(
    json.dumps(emptied_companies, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"再収集対象リスト保存: {recollect_path} ({len(emptied_companies)}社)")

# ── 汚染写真をGitから削除する準備 ─────────────────────────────────────
print(f"\n=== 汚染写真の削除 ===")
print("汚染写真をGitリポジトリから削除します...")

# パスをUnix形式に変換し、存在確認
unix_paths = []
for p in contaminated_photos:
    # バックスラッシュをスラッシュに変換
    p_unix = p.replace("\\", "/")
    unix_paths.append(p_unix)

if unix_paths:
    # git rm --cached でインデックスから削除（ファイルが存在しない場合も対応）
    batch_size = 200
    removed_count = 0
    for i in range(0, len(unix_paths), batch_size):
        batch = unix_paths[i:i+batch_size]
        r = subprocess.run(
            ["git", "-c", "core.quotePath=false", "rm", "--cached", "--ignore-unmatch", "-r"] + batch,
            capture_output=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_DIR)
        )
        if r.returncode == 0:
            # 削除された行数をカウント
            lines = [l for l in r.stdout.splitlines() if l.startswith("rm '")]
            removed_count += len(lines)
            if i == 0:
                print(f"  バッチ {i//batch_size+1}: {len(lines)}件削除")

    print(f"Gitインデックスから削除: {removed_count}件")
else:
    print("削除対象なし")

print("\n完了。次のステップ:")
print("1. git commit してGitHubに汚染削除をプッシュ")
print("2. collect_history2.py で正しいデータを再収集")
