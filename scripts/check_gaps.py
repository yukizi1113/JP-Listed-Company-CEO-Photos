"""写真なし企業の詳細分析"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR = PROJECT_DIR / "data"

no_photo = json.loads((DATA_DIR / "no_photo_tickers.json").read_text(encoding="utf-8"))
hs = json.loads((DATA_DIR / "history_supplement.json").read_text(encoding="utf-8"))
hs_tickers = set(str(x["ticker"]) for x in hs)

has_history_no_photo = set(t for t, n in no_photo if t in hs_tickers)
no_history_no_photo = set(t for t, n in no_photo if t not in hs_tickers)

print(f"=== 歴代データあり・写真なし: {len(has_history_no_photo)}社 ===")
print("サンプル10社:")
count = 0
total_ceos_in_group = 0
ceos_with_valid_name = 0
for item in hs:
    if item["ticker"] in has_history_no_photo:
        ceos = item.get("previous_ceos", [])
        total_ceos_in_group += len(ceos)
        for c in ceos:
            if c.get("name"):
                ceos_with_valid_name += 1
        if count < 10:
            ticker = item["ticker"]
            name = item.get("company_name", "")
            ceos_with_photo = [c for c in ceos if c.get("photo_path")]
            print(f"  {ticker} {name} - CEO数:{len(ceos)}, 写真:{len(ceos_with_photo)}")
            for c in ceos[:3]:
                print(f"    - {c.get('name')} | photo:{bool(c.get('photo_path'))}")
        count += 1

print(f"合計{count}社, 総CEO名: {total_ceos_in_group}, 有効名: {ceos_with_valid_name}")

print(f"\n=== 歴代データなし・写真なし: {len(no_history_no_photo)}社 ===")
ceo_data = json.loads((DATA_DIR / "ceo_data.json").read_text(encoding="utf-8"))
ceo_by_ticker = {str(c["ticker"]): c for c in ceo_data}

# Check current CEO data for no-history companies
has_current = sum(1 for t in no_history_no_photo if ceo_by_ticker.get(t, {}).get("current_ceo"))
print(f"  現任CEOデータあり: {has_current}社")
print(f"  現任CEOデータなし: {len(no_history_no_photo) - has_current}社")

# Sample
print("サンプル15社:")
for t, n in sorted(no_photo)[:15]:
    if t in no_history_no_photo:
        curr = ceo_by_ticker.get(t, {}).get("current_ceo")
        curr_name = curr.get("name", "") if curr else "なし"
        print(f"  {t} {n} | 現任CEO: {curr_name}")
