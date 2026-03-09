# 日本上場企業 CEO 顔写真・経歴・株価データセット

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Companies](https://img.shields.io/badge/対象企業-3%2C727社-blue)]()
[![Source](https://img.shields.io/badge/出典-会社四季報2026年1集-orange)]()

## 概要

会社四季報2026年1集（新春号）掲載の全上場企業 **3,727社** を対象に、
代表取締役社長（CEO）の以下データを収集した機械学習用データセットです。

| データ種別 | 内容 |
|------------|------|
| 顔写真 | 現・歴代CEO（2000年以降）の顔写真 |
| 就任年月日 | 現CEO・歴代CEO（2000年以降）全員分 |
| 退任年月日 | 歴代CEO（2000年以降）全員分 |
| 就任時株価 | 就任日の始値 (Open) |
| 退任時株価 | 退任日の終値 (Close) |

---

## 収集統計

| 項目 | 数値 |
|------|------|
| 対象企業数 | 3,727社 |
| 歴代データあり企業 | 3,166社 (84.9%) |
| 歴代CEO総数 | 11,579名 |
| 写真取得済み | 11,194名 **(96.7%)** |
| データ収集期間 | 2026年2月〜3月 |

---

## クイックスタート (Python)

```python
import pandas as pd
from pathlib import Path

# ① 全社CEO統合データ（1行=1社）
df = pd.read_csv("data/ml_dataset.csv", encoding="utf-8-sig")

# ② 写真索引（1行=1写真ファイル）- ML学習に最適
photos = pd.read_csv("data/ml_dataset_photos.csv", encoding="utf-8-sig")
photos_exist = photos[photos["photo_exists"]]

# 例: 就任時株価と写真を持つ現CEO一覧
current = photos_exist[photos_exist["ceo_role"] == "current"]
print(current[["ticker", "company_name", "ceo_name", "appointment_date",
               "open_at_appointment", "photo_path"]].head())
```

```python
# ③ 顔写真を PIL で読み込む
from PIL import Image

def load_ceo_image(photo_path: str) -> Image.Image | None:
    """photo_path は ml_dataset_photos.csv の photo_path 列の値"""
    p = Path(photo_path)
    return Image.open(p) if p.exists() else None
```

---

## ディレクトリ構成

```
data/
  companies.json              全3,727社リスト (ticker/社名/URL)
  ceo_data.json               現CEO情報 JSON (メインDB)
  history_supplement.json     歴代CEO情報 JSON (11,579名)
  edinet_mapping.json         EDINET有価証券報告書インデックス (3,598社・28,768件)
  ml_dataset.csv              ML用統合インデックス (1行=1社)
  ml_dataset_photos.csv       写真索引 (1行=1写真, ML訓練に最適)

photos_1/   ticker < 3500
photos_2/   3500 <= ticker < 5000
photos_3/   5000 <= ticker < 7000
photos_4/   7000 <= ticker < 9000
photos_5/   ticker >= 9000 または非数値
  各ディレクトリ内:
  {ticker}_{社名}/
    current/
      photo_01.jpg            現CEO 写真
      info.json               現CEO メタデータ
    history/
      01_{氏名}/
        photo_01.jpg          前任CEO写真 (取得できた場合)
      02_{氏名}/              2代前
      ...                     2000年以降の全歴代

scripts/
  collect_ceo.py              現CEO情報収集
  collect_history.py          歴代CEO収集 (Wikipedia)
  collect_history2.py         歴代CEO収集 改良版 (汚染防止付き)
  collect_history_edinet.py   歴代CEO収集 (EDINET有価証券報告書)
  collect_history_photos.py   歴代CEO写真収集
  cleanup_contaminated.py     汚染データ除去
  merge_history.py            歴代データをメインDBへマージ
  make_ml_dataset.py          ML用CSV生成
  build_tracking_csv.py       CEO追跡CSV生成
```

---

## データスキーマ

### `data/history_supplement.json`

```json
{
  "ticker": 1234,
  "company_name": "サンプル株式会社",
  "previous_ceos": [
    {
      "name": "田中 次郎",
      "appointment_date": "2015-06-01",
      "resignation_date": "2021-06-01",
      "photo_path": "photos_1/1234_サンプル株式会社/history/01_田中_次郎/photo_01.jpg",
      "source": "edinet"
    }
  ]
}
```

### `data/ml_dataset_photos.csv` (1行=1写真)

| 列名 | 型 | 説明 |
|------|----|------|
| ticker | str | 証券コード |
| company_name | str | 社名 |
| ceo_name | str | CEO氏名 |
| ceo_role | str | `current` / `prev_1` / `prev_2` ... |
| appointment_date | str | 就任年月日 |
| resignation_date | str | 退任年月日 (歴代のみ) |
| open_at_appointment | float | **就任時始値** (JPY) |
| close_at_resignation | float | **退任時終値** (JPY) |
| photo_path | str | 写真ファイルの相対パス |
| photo_exists | bool | ファイル存在フラグ |

---

## 想定ML用途

| 用途 | 使用データ |
|------|-----------|
| CEO在任期間と株価リターンの相関分析 | `ml_dataset.csv`: open_at_appointment, close_at_resignation |
| CEO交代が株価に与える影響の予測 | `ml_dataset_photos.csv` + yfinance追加取得 |
| 経営者顔写真とパフォーマンスの関係 | `photo_path` + `open_at_appointment` |
| 歴代CEO在任期間の統計分析 | `history_supplement.json` |
| 業種別CEO交代パターン分析 | `ml_dataset.csv` + `companies.json` |

---

## データソース

| データ | 出典 |
|--------|------|
| 企業一覧 | 会社四季報2026年1集（新春号）|
| 現CEO情報・写真 | 各社公式IR・コーポレートサイト |
| 歴代CEO情報 | Wikipedia / 各社プレスリリース |
| 歴代CEO情報 (補完) | EDINET 有価証券報告書（金融庁）|
| 写真 | Wikipedia / Bing画像検索 |
| 株価データ | Yahoo Finance Japan (yfinance) |

**注意**: 顔写真の著作権は各社・各権利者に帰属します。研究・教育目的での利用に限定してください。
株価データは参考値であり、投資判断に使用しないでください。

---
*収集期間: 2026年2月〜3月 | Python: requests, BeautifulSoup, yfinance, Pillow*
