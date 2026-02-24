# 日本上場企業 CEO 顔写真・経歴・株価データセット

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Companies](https://img.shields.io/badge/対象企業-3%2C727社-blue)]()
[![Source](https://img.shields.io/badge/出典-会社四季報2026年1集-orange)]()

## 概要

会社四季報2026年1集（新春号）掲載の全上場企業 **3,727社** を対象に、
代表取締役社長（CEO）の以下データを収集した機械学習用データセットです。

| データ種別 | 内容 |
|------------|------|
| 顔写真 | 現・歴代CEO（2000年以降）の顔写真（複数枚/人） |
| 就任年月日 | 現CEO・歴代CEO（2000年以降）全員分 |
| 退任年月日 | 歴代CEO（2000年以降）全員分 |
| 就任時株価 | 就任月の始値 (Open) |
| 退任時株価 | 退任月の終値 (Close) |

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

# 例
for _, row in photos_exist.iterrows():
    img = load_ceo_image(row["photo_path"])
    ticker = row["ticker"]
    open_price = row["open_at_appointment"]
    appt_date = row["appointment_date"]
    # -> img, ticker, open_price, appt_date が揃った ML 学習データ
```

---

## ディレクトリ構成

```
data/
  companies.json          全3,727社リスト (ticker/社名/URL)
  ceo_data.json           全CEO情報 JSON (メインDB)
  ml_dataset.csv          ML用統合インデックス (1行=1社)
  ml_dataset_photos.csv   写真索引 (1行=1写真, ML訓練に最適)
  stats.json              データセット統計
  progress.json           収集進捗

photos/
  {ticker}_{社名}/
    current/
      photo_01.jpg        現CEO 写真1枚目
      photo_02.jpg        現CEO 写真2枚目 (あれば)
      info.json           現CEO メタデータ
    history/
      01_{氏名}/
        photo_01.jpg      前任CEO写真 (あれば)
        info.json         前任CEO メタデータ
      02_{氏名}/          2代前
      03_{氏名}/          3代前
      ...                 2000年以降の全歴代

scripts/
  collect_ceo.py          データ収集スクリプト
  make_ml_dataset.py      ML用CSV生成スクリプト
  run_all.py              全件一括収集・push・CSV自動生成
```

---

## データスキーマ

### `data/ml_dataset.csv` (1行=1社)

| 列名 | 型 | 説明 |
|------|----|------|
| ticker | str | 証券コード |
| company_name | str | 社名 |
| url | str | 企業URL |
| current_ceo_name | str | 現代表取締役社長名 |
| current_ceo_title | str | 役職名 |
| appointment_date | str | 就任年月（例: 2022年6月） |
| open_at_appointment | float | 就任月の始値 (JPY) |
| close_at_appointment | float | 就任月の終値 (JPY) |
| stock_date_at_appointment | str | 株価取得日 |
| photo_count | int | 取得写真枚数 |
| photo_dir | str | 写真ディレクトリパス |
| photo_paths | str | カンマ区切り写真パス一覧 |
| prev_ceo_count | int | 前任社長データ数（2000年以降） |

### `data/ml_dataset_photos.csv` (1行=1写真)

| 列名 | 型 | 説明 |
|------|----|------|
| ticker | str | 証券コード |
| company_name | str | 社名 |
| ceo_name | str | CEO氏名 |
| ceo_role | str | `current` / `prev_1` / `prev_2` ... |
| appointment_date | str | 就任年月 |
| resignation_date | str | 退任年月 (前任者のみ) |
| open_at_appointment | float | **就任時始値** (JPY) |
| close_at_resignation | float | **退任時終値** (JPY) |
| photo_path | str | 写真ファイルの相対パス |
| photo_exists | bool | ファイル存在フラグ |

### `photos/{ticker}_{社名}/current/info.json`

```json
{
  "name": "山田 太郎",
  "title": "代表取締役社長",
  "appointment_info": "2021年6月",
  "photos_saved": ["photo_01.jpg", "photo_02.jpg"],
  "photo_count": 2,
  "open_at_appointment": 2345.0,
  "stock_price_at_appointment": {
    "date": "2021-06-01",
    "open_on_date": 2345.0,
    "close_on_date": 2380.0,
    "trading_date": "2021-06-01",
    "currency": "JPY"
  },
  "source_url": "https://example.co.jp/company/officer/"
}
```

### `photos/{ticker}_{社名}/history/01_{氏名}/info.json`

```json
{
  "name": "田中 次郎",
  "appointment_date": "2015年6月",
  "resignation_date": "2021年6月",
  "open_at_appointment": 1200.0,
  "close_at_resignation": 2300.0,
  "stock_price_at_appointment": { ... },
  "stock_price_at_resignation": { ... },
  "source_url": "https://example.co.jp/news/..."
}
```

---

## 想定ML用途

| 用途 | 使用データ |
|------|-----------|
| CEO在任期間と株価リターンの相関分析 | `ml_dataset.csv`: open_at_appointment, close_at_resignation |
| CEO交代が株価に与える影響の予測 | `ml_dataset_photos.csv` + yfinance追加取得 |
| 経営者顔写真とパフォーマンスの関係 | `photo_path` + `open_at_appointment` |
| 歴代CEO在任期間の統計分析 | `ceo_data.json`: previous_ceos |
| 業種別CEO交代パターン分析 | `ml_dataset.csv` + `companies.json` |

---

## 収集統計

> `data/stats.json` を参照（収集完了後に自動更新）

---

## データソース・免責

| データ | 出典 |
|--------|------|
| 企業一覧 | 会社四季報2026年1集（新春号）|
| CEO情報・写真 | 各社公式IR・コーポレートサイト |
| 歴代CEO情報 | 各社プレスリリース |
| 株価データ | Yahoo Finance Japan (yfinance) |

**注意**: 顔写真の著作権は各社に帰属します。研究・教育目的での利用に限定してください。
株価データは参考値であり、投資判断に使用しないでください。

---
*収集期間: 2026年2月〜 | Python: yfinance, requests, BeautifulSoup, Pillow*
