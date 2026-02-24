"""
ML Dataset Generator
GitHubリポジトリ上のデータから機械学習用インデックスCSVを生成します。

生成ファイル:
  data/ml_dataset.csv         - 全データ統合インデックス (ticker + CEO + 写真パス + 株価)
  data/ml_dataset_photos.csv  - 写真ファイルのみの索引 (1行=1写真)
  data/stats.json             - データセット統計

使用例:
  import pandas as pd
  df = pd.read_csv("data/ml_dataset.csv")
  df_photos = df[df["photo_exists"]]
  # -> ticker, company, ceo_name, appointment_date, photo_path, open_at_appointment, ...
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR = PROJECT_DIR / "data"
PHOTOS_DIR = PROJECT_DIR / "photos"
CEO_DATA_FILE = DATA_DIR / "ceo_data.json"
ML_CSV = DATA_DIR / "ml_dataset.csv"
PHOTOS_CSV = DATA_DIR / "ml_dataset_photos.csv"
STATS_JSON = DATA_DIR / "stats.json"


MAIN_COLUMNS = [
    "ticker",
    "company_name",
    "url",
    # Current CEO
    "current_ceo_name",
    "current_ceo_title",
    "appointment_date",
    "open_at_appointment",          # 就任時始値 (JPY)
    "close_at_appointment",         # 就任時終値 (JPY)
    "stock_date_at_appointment",    # 株価取得日
    "photo_count",                  # 取得写真枚数
    "photo_dir",                    # 写真ディレクトリ (GitHub相対パス)
    "photo_paths",                  # カンマ区切り写真パス一覧
    "ceo_source_url",               # CEO情報取得元URL
    # Previous CEO counts
    "prev_ceo_count",
    "processed_at",
    "error",
]

PHOTO_COLUMNS = [
    "ticker",
    "company_name",
    "ceo_name",
    "ceo_role",                     # "current" or "prev_1", "prev_2", ...
    "appointment_date",
    "resignation_date",
    "open_at_appointment",
    "close_at_resignation",
    "photo_path",                   # GitHub相対パス (photos/{ticker}_{name}/current/photo_01.jpg)
    "photo_exists",                 # ファイルが実際に存在するか
]


def make_safe_name(s: str) -> str:
    import re
    return re.sub(r'[^\w\u3040-\u30ff\u4e00-\u9fff\-]', '_', str(s))[:50]


def main():
    if not CEO_DATA_FILE.exists():
        print("ceo_data.json が見つかりません。先に collect_ceo.py を実行してください。")
        sys.exit(1)

    with open(CEO_DATA_FILE, encoding="utf-8") as f:
        companies = json.load(f)

    print(f"Loading {len(companies)} companies...")

    main_rows = []
    photo_rows = []

    for company in companies:
        ticker = company.get("ticker", "")
        cname = company.get("company_name", "")
        url = company.get("url", "")
        ceo = company.get("current_ceo") or {}
        prev_ceos = company.get("previous_ceos") or []
        sn = make_safe_name(cname)
        comp_dir = f"{ticker}_{sn}"

        # ── Main row (current CEO) ──
        price = ceo.get("stock_price_at_appointment") or {}
        photos_saved = ceo.get("photos_saved") or (["photo_01.jpg"] if ceo.get("photo_saved") else [])
        photo_dir = f"photos/{comp_dir}/current"
        photo_paths = ",".join(f"{photo_dir}/{p}" for p in photos_saved) if photos_saved else ""

        main_row = {
            "ticker": ticker,
            "company_name": cname,
            "url": url,
            "current_ceo_name": ceo.get("name", ""),
            "current_ceo_title": ceo.get("title", ""),
            "appointment_date": ceo.get("appointment_info", ""),
            "open_at_appointment": ceo.get("open_at_appointment") or price.get("open_on_date", ""),
            "close_at_appointment": price.get("close_on_date", ""),
            "stock_date_at_appointment": price.get("trading_date", price.get("date", "")),
            "photo_count": ceo.get("photo_count", len(photos_saved)),
            "photo_dir": photo_dir if photos_saved else "",
            "photo_paths": photo_paths,
            "ceo_source_url": ceo.get("source_url", ""),
            "prev_ceo_count": len(prev_ceos),
            "processed_at": company.get("processed_at", ""),
            "error": company.get("error", ""),
        }
        main_rows.append(main_row)

        # ── Photo rows (current CEO photos) ──
        for photo_name in photos_saved:
            ppath = f"photos/{comp_dir}/current/{photo_name}"
            pexists = (PHOTOS_DIR / f"{comp_dir}" / "current" / photo_name).exists()
            photo_rows.append({
                "ticker": ticker,
                "company_name": cname,
                "ceo_name": ceo.get("name", ""),
                "ceo_role": "current",
                "appointment_date": ceo.get("appointment_info", ""),
                "resignation_date": "",
                "open_at_appointment": main_row["open_at_appointment"],
                "close_at_resignation": "",
                "photo_path": ppath,
                "photo_exists": pexists,
            })

        # ── Photo rows (previous CEOs) ──
        for i, prev in enumerate(prev_ceos, 1):
            pn = make_safe_name(prev.get("name", "unknown"))
            hist_dir = f"photos/{comp_dir}/history/{i:02d}_{pn}"
            res_price = prev.get("stock_price_at_resignation") or {}
            appt_price = prev.get("stock_price_at_appointment") or {}

            # Check for photos in hist_dir
            local_hist = PHOTOS_DIR / f"{comp_dir}" / "history" / f"{i:02d}_{pn}"
            hist_photos = list(local_hist.glob("photo_*.jpg")) if local_hist.exists() else []
            if not hist_photos:
                # Still create a row for the CEO even without photo
                photo_rows.append({
                    "ticker": ticker,
                    "company_name": cname,
                    "ceo_name": prev.get("name", ""),
                    "ceo_role": f"prev_{i}",
                    "appointment_date": prev.get("appointment_date", ""),
                    "resignation_date": prev.get("resignation_date", ""),
                    "open_at_appointment": prev.get("open_at_appointment") or appt_price.get("open_on_date", ""),
                    "close_at_resignation": prev.get("close_at_resignation") or res_price.get("close_on_date", ""),
                    "photo_path": "",
                    "photo_exists": False,
                })
            else:
                for hp in hist_photos:
                    ppath = f"{hist_dir}/{hp.name}"
                    photo_rows.append({
                        "ticker": ticker,
                        "company_name": cname,
                        "ceo_name": prev.get("name", ""),
                        "ceo_role": f"prev_{i}",
                        "appointment_date": prev.get("appointment_date", ""),
                        "resignation_date": prev.get("resignation_date", ""),
                        "open_at_appointment": prev.get("open_at_appointment") or appt_price.get("open_on_date", ""),
                        "close_at_resignation": prev.get("close_at_resignation") or res_price.get("close_on_date", ""),
                        "photo_path": ppath,
                        "photo_exists": True,
                    })

    # Write main CSV
    with open(ML_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MAIN_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(main_rows)
    print(f"ML dataset CSV: {ML_CSV} ({len(main_rows)} rows)")

    # Write photos CSV
    with open(PHOTOS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PHOTO_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(photo_rows)
    print(f"Photos CSV: {PHOTOS_CSV} ({len(photo_rows)} rows)")

    # Stats
    with_ceo = sum(1 for r in main_rows if r["current_ceo_name"])
    with_photo = sum(1 for r in main_rows if r["photo_count"] and int(r["photo_count"] or 0) > 0)
    with_appt_price = sum(1 for r in main_rows if r["open_at_appointment"])
    prev_total = sum(int(r["prev_ceo_count"] or 0) for r in main_rows)
    photo_total = sum(1 for r in photo_rows if r["photo_exists"])

    stats = {
        "generated_at": datetime.now().isoformat(),
        "total_companies": len(companies),
        "ceo_identified": with_ceo,
        "ceo_with_photo": with_photo,
        "ceo_with_appointment_price": with_appt_price,
        "previous_ceos_total": prev_total,
        "photos_on_disk": photo_total,
        "photo_rows_total": len(photo_rows),
        "coverage_rate_ceo": f"{with_ceo/max(len(companies),1)*100:.1f}%",
        "coverage_rate_photo": f"{with_photo/max(len(companies),1)*100:.1f}%",
    }
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Dataset Stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


if __name__ == "__main__":
    main()
