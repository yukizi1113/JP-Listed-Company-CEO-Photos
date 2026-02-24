"""
サマリーCSV生成スクリプト
ceo_data.json から分析用CSVを生成します
"""
import json
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
CEO_DATA_FILE = PROJECT_DIR / "data" / "ceo_data.json"
OUTPUT_CSV = PROJECT_DIR / "data" / "ceo_summary.csv"

def main():
    if not CEO_DATA_FILE.exists():
        print("ceo_data.json が見つかりません")
        sys.exit(1)

    with open(CEO_DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for company in data:
        ticker = company.get("ticker", "")
        name = company.get("company_name", "")
        url = company.get("url", "")
        ceo = company.get("current_ceo") or {}
        price = (ceo.get("stock_price_at_appointment") or {})

        base_row = {
            "ticker": ticker,
            "company_name": name,
            "url": url,
            "ceo_name": ceo.get("name", ""),
            "ceo_title": ceo.get("title", ""),
            "appointment_date": ceo.get("appointment_info", ""),
            "photo_saved": ceo.get("photo_saved", False),
            "source_url": ceo.get("source_url", ""),
            "stock_price_at_appointment": price.get("close_on_date", ""),
            "stock_currency": price.get("currency", ""),
            "previous_ceo_count": len(company.get("previous_ceos", [])),
            "error": company.get("error", ""),
        }
        rows.append(base_row)

        # Add history rows
        for i, prev in enumerate(company.get("previous_ceos", [])[:3], 1):
            res_price = (prev.get("stock_price_at_resignation") or {})
            hist_row = {
                "ticker": ticker,
                "company_name": name,
                "url": url,
                "ceo_name": prev.get("name", ""),
                "ceo_title": f"前任({i})",
                "appointment_date": prev.get("appointment_date", ""),
                "resignation_date": prev.get("resignation_date", ""),
                "photo_saved": False,
                "source_url": prev.get("source_url", ""),
                "stock_price_at_resignation": res_price.get("close_on_date", ""),
                "stock_currency": res_price.get("currency", ""),
                "previous_ceo_count": "",
                "error": "",
            }
            rows.append(hist_row)

    # Write CSV
    if not rows:
        print("データがありません")
        return

    fieldnames = list(rows[0].keys())
    # Add resignation fields that may exist
    for extra in ["resignation_date", "stock_price_at_resignation"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved: {OUTPUT_CSV}")
    print(f"Total rows: {len(rows)}")
    with_photos = sum(1 for r in rows if r.get("photo_saved"))
    print(f"Rows with photo: {with_photos}")

if __name__ == "__main__":
    main()
