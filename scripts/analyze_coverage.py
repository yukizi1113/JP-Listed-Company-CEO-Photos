"""写真カバレッジ分析 & 写真なし企業リスト生成"""
import json, re, sys, subprocess
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"

def main():
    # git ls-filesから写真ありticker抽出
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-files",
         "photos_1/", "photos_2/", "photos_3/", "photos_4/", "photos_5/"],
        capture_output=True, cwd=str(PROJECT_DIR)
    )
    lines = result.stdout.decode("utf-8", errors="replace").splitlines()
    photo_lines = [l for l in lines if l.endswith(".jpg")]

    tickers_with_photos = set()
    for line in photo_lines:
        m = re.match(r"photos_\d+/([^/]+)/", line)
        if m:
            ticker = m.group(1).split("_")[0]
            tickers_with_photos.add(ticker)

    # ceo_data.jsonから全企業ロード
    ceo_data = json.loads((DATA_DIR / "ceo_data.json").read_text(encoding="utf-8"))
    all_tickers = set(str(c["ticker"]) for c in ceo_data)
    no_photo_tickers = all_tickers - tickers_with_photos

    print(f"全企業: {len(all_tickers)}社")
    print(f"写真あり: {len(tickers_with_photos)}社 ({100*len(tickers_with_photos)/len(all_tickers):.1f}%)")
    print(f"写真なし: {len(no_photo_tickers)}社 ({100*len(no_photo_tickers)/len(all_tickers):.1f}%)")
    print(f"合計写真ファイル: {len(photo_lines)}枚")

    # 写真なし企業リスト
    no_photo_companies = sorted(
        [(c["ticker"], c.get("company_name", "")) for c in ceo_data if str(c["ticker"]) in no_photo_tickers]
    )
    out_path = DATA_DIR / "no_photo_tickers.json"
    out_path.write_text(json.dumps(no_photo_companies, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写真なし企業リスト保存: {out_path} ({len(no_photo_companies)}社)")

    # historyあり/なし確認
    hs = json.loads((DATA_DIR / "history_supplement.json").read_text(encoding="utf-8"))
    hs_tickers = set(str(x["ticker"]) for x in hs)
    no_history = no_photo_tickers - hs_tickers
    has_history_no_photo = no_photo_tickers & hs_tickers
    print(f"\n写真なし企業の内訳:")
    print(f"  歴代データもなし: {len(no_history)}社")
    print(f"  歴代データあり→写真取得失敗: {len(has_history_no_photo)}社")

    return no_photo_companies

if __name__ == "__main__":
    main()
