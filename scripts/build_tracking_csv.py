"""
build_tracking_csv.py  –  全社長追跡CSVの生成

会社四季報Excelを企業マスターとし、history_supplement.json + ceo_data.json の
既存データを統合して「各社長の在任期間・写真取得状況・GitHub登録状況」を
一覧する data/ceo_tracking.csv を生成する。

Usage:
  python scripts/build_tracking_csv.py

出力列:
  ticker, company_name, sector, ceo_name, appointment_date, resignation_date,
  tenure_years, photo_path, photo_acquired, github_uploaded, source, notes
"""

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR    = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR       = PROJECT_DIR / "data"
SHIKIHO_XLSX   = Path(r"C:\Users\hp\Documents\JP_Valuation_EDINET\docs\source\会社四季報2026年1集 新春号_定性情報.xlsx")
CEO_DATA_FILE  = DATA_DIR / "ceo_data.json"
HISTORY_FILE   = DATA_DIR / "history_supplement.json"
TRACKING_CSV   = DATA_DIR / "ceo_tracking.csv"

import os
GH_TOKEN   = os.environ.get("GH_TOKEN", "")
GH_USER    = "yukizi1113"
REPO_NAME  = "JP-Listed-Company-CEO-Photos"
REMOTE_URL = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{REPO_NAME}.git" if GH_TOKEN else ""


def _git(*args, timeout=60):
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(PROJECT_DIR),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def tenure_years(appt: str, res: str) -> float | None:
    try:
        d1 = datetime.fromisoformat(str(appt)[:10])
        d2 = datetime.fromisoformat(str(res)[:10])
        return round((d2 - d1).days / 365.25, 2)
    except Exception:
        return None


def github_file_exists_in_index(path: str) -> bool:
    """git ls-files でファイルがインデックスに存在するか確認。"""
    ok, out = _git("-c", "core.quotePath=false", "ls-files", "--", path)
    return bool(out.strip())


def main():
    # ── 四季報 Excel 読み込み (企業マスター) ──
    print("四季報Excel読み込み中...")
    shikiho = pd.read_excel(SHIKIHO_XLSX, dtype={"ticker": str})
    shikiho["ticker"] = shikiho["ticker"].str.strip()
    # URLが列にある場合は活用
    url_map = {}
    if "ＵＲＬ" in shikiho.columns:
        url_map = dict(zip(shikiho["ticker"], shikiho["ＵＲＬ"].fillna("")))
    sector_map = {}
    if "業種名" in shikiho.columns:
        sector_map = dict(zip(shikiho["ticker"], shikiho["業種名"].fillna("")))
    elif "業種" in shikiho.columns:
        sector_map = dict(zip(shikiho["ticker"], shikiho["業種"].fillna("")))
    name_map = {}
    if "社名" in shikiho.columns:
        name_map = dict(zip(shikiho["ticker"], shikiho["社名"].fillna("")))

    all_tickers = list(shikiho["ticker"].dropna().unique())
    print(f"  四季報企業数: {len(all_tickers)}")

    # ── ceo_data.json 読み込み (現役社長) ──
    ceo_data_raw = json.loads(CEO_DATA_FILE.read_text(encoding="utf-8"))
    ceo_by_ticker = {c["ticker"]: c for c in ceo_data_raw}

    # ── history_supplement.json 読み込み (歴代社長) ──
    history_raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    history_by_ticker = {c["ticker"]: c for c in history_raw}

    # ── GitHub インデックス内ファイル一覧を取得 ──
    print("GitHub インデックス確認中...")
    ok, all_files = _git("-c", "core.quotePath=false", "ls-files", "--", "photos_1/", "photos_2/")
    github_files = set(all_files.splitlines()) if ok else set()
    print(f"  GitHubインデックスファイル数: {len(github_files)}")

    rows = []

    for ticker in all_tickers:
        comp_name = name_map.get(ticker, ceo_by_ticker.get(ticker, {}).get("company_name", ""))
        sector    = sector_map.get(ticker, "")

        ceo_entry    = ceo_by_ticker.get(ticker, {})
        history_entry = history_by_ticker.get(ticker, {})

        def _photo_github(photo_path: str) -> bool:
            if not photo_path:
                return False
            p = str(photo_path).replace("\\", "/")
            return p in github_files

        # ── 現役社長 ──
        current = ceo_entry.get("current_ceo") or {}
        if current.get("name"):
            photos_saved = current.get("photos_saved") or (
                ["photo_01.jpg"] if current.get("photo_saved") else []
            )
            # photos_1 or photos_2 判定
            try:
                pdir = "photos_1" if int(ticker) < 5500 else "photos_2"
            except ValueError:
                pdir = "photos_2"
            safe = re.sub(r'[\\/:*?"<>|\s\u3000]', "_", comp_name).strip("_")
            comp_dir = f"{ticker}_{safe}"
            photo_paths = [f"{pdir}/{comp_dir}/current/{p}" for p in photos_saved]
            github_ok = any(_photo_github(p) for p in photo_paths)

            rows.append({
                "ticker":           ticker,
                "company_name":     comp_name,
                "sector":           sector,
                "ceo_name":         current.get("name", ""),
                "appointment_date": current.get("appointment_info") or current.get("appointment_date", ""),
                "resignation_date": "（現在）",
                "tenure_years":     "",
                "photo_path":       photo_paths[0] if photo_paths else "",
                "photo_acquired":   "✓" if github_ok or bool(photos_saved) else "",
                "github_uploaded":  "✓" if github_ok else "",
                "source":           "current",
                "notes":            "",
            })

        # ── 歴代社長 ──
        for i, prev in enumerate(history_entry.get("previous_ceos", []), 1):
            name = prev.get("name", "")
            if not name:
                continue
            photo_path = prev.get("photo_path", "")
            github_ok  = _photo_github(photo_path)
            appt = prev.get("appointment_date", "")
            res  = prev.get("resignation_date", "")
            ty   = tenure_years(appt, res)

            rows.append({
                "ticker":           ticker,
                "company_name":     comp_name,
                "sector":           sector,
                "ceo_name":         name,
                "appointment_date": appt,
                "resignation_date": res,
                "tenure_years":     ty if ty else "",
                "photo_path":       str(photo_path).replace("\\", "/") if photo_path else "",
                "photo_acquired":   "✓" if photo_path else "",
                "github_uploaded":  "✓" if github_ok else "",
                "source":           "history",
                "notes":            "",
            })

        # ── データなし企業 (四季報にはあるが CEO データなし) ──
        if not current.get("name") and not history_entry.get("previous_ceos"):
            rows.append({
                "ticker":           ticker,
                "company_name":     comp_name,
                "sector":           sector,
                "ceo_name":         "",
                "appointment_date": "",
                "resignation_date": "",
                "tenure_years":     "",
                "photo_path":       "",
                "photo_acquired":   "",
                "github_uploaded":  "",
                "source":           "missing",
                "notes":            "要調査",
            })

    # ── CSV 保存 ──
    columns = [
        "ticker", "company_name", "sector", "ceo_name",
        "appointment_date", "resignation_date", "tenure_years",
        "photo_path", "photo_acquired", "github_uploaded",
        "source", "notes",
    ]
    with open(TRACKING_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ── サマリー ──
    total     = len(rows)
    with_data = sum(1 for r in rows if r["ceo_name"])
    acquired  = sum(1 for r in rows if r["photo_acquired"])
    uploaded  = sum(1 for r in rows if r["github_uploaded"])
    missing   = sum(1 for r in rows if r["source"] == "missing")
    hist_rows = sum(1 for r in rows if r["source"] == "history")

    print(f"\n=== 追跡CSV完成: {TRACKING_CSV} ===")
    print(f"  総行数:           {total}")
    print(f"  CEO情報あり:      {with_data}")
    print(f"  歴代社長行:       {hist_rows}")
    print(f"  写真取得済み:     {acquired} ({acquired/max(with_data,1)*100:.1f}%)")
    print(f"  GitHub登録済み:   {uploaded} ({uploaded/max(with_data,1)*100:.1f}%)")
    print(f"  データなし企業:   {missing} (要調査)")

    # ── sector別サマリー ──
    df = pd.DataFrame(rows)
    if "sector" in df.columns and df["sector"].notna().any():
        sector_summary = (
            df[df["source"] == "history"]
            .groupby("sector")
            .agg(
                CEO数=("ceo_name", "count"),
                写真取得=("photo_acquired", lambda x: (x == "✓").sum()),
                GitHub登録=("github_uploaded", lambda x: (x == "✓").sum()),
            )
            .reset_index()
            .sort_values("CEO数", ascending=False)
        )
        print("\n業種別サマリー (歴代社長のみ、上位10):")
        print(sector_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
