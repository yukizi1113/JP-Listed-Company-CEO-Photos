# 日本上場企業 CEO 顔写真・経歴・株価データセット

[![Dataset](https://img.shields.io/badge/Dataset-JP%20CEO%20Photos-blue)](https://github.com/yukizi1113/JP-Listed-Company-CEO-Photos)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 概要

会社四季報2026年1集（新春号）掲載の全上場企業（3,727社）を対象に収集した、
機械学習用の日本企業CEO（代表取締役社長）マルチモーダルデータセットです。

**機械学習用途例：**
- CEO就任・退任と株価変動の相関分析
- 経営者の在任期間予測モデル
- CEO交代が企業パフォーマンスに与える影響分析
- 経営者顔写真とパフォーマンスの関係研究

---

## データ構成

```
data/
  companies.json       全3,727社リスト（ticker/社名/URL）
  ceo_data.json        全CEO情報（メインデータセット）
  ceo_summary.csv      CSV形式サマリー（ML用途向け）
  progress.json        収集進捗

photos/
  {ticker}_{社名}/
    current/
      photo.jpg        現代表取締役社長 顔写真 (JPEG)
      info.json        現社長メタデータ
    history/
      01_{氏名}/
        photo.jpg      前任社長 顔写真
        info.json      前任社長メタデータ
      02_{氏名}/       2代前の社長
      03_{氏名}/       3代前の社長

scripts/
  collect_ceo.py       データ収集スクリプト
  make_summary.py      CSV生成スクリプト
  run_all.py           全件一括収集スクリプト
```

---

## データスキーマ

### `data/ceo_data.json`

```json
[
  {
    "ticker": "1301",
    "company_name": "極洋",
    "url": "https://www.kyokuyo.co.jp/",
    "current_ceo": {
      "name": "山田 太郎",
      "title": "代表取締役社長",
      "photo_url": "https://...",
      "appointment_info": "2021年6月",
      "photo_saved": true,
      "source_url": "https://...",
      "stock_price_at_appointment": {
        "date": "2021-06",
        "close_on_date": 2345.0,
        "close_min_period": 2100.0,
        "close_max_period": 2500.0,
        "currency": "JPY"
      }
    },
    "previous_ceos": [
      {
        "name": "田中 次郎",
        "appointment_date": "2015年6月",
        "resignation_date": "2021年6月",
        "source_url": "https://...",
        "stock_price_at_resignation": {
          "date": "2021-06",
          "close_on_date": 2345.0,
          "currency": "JPY"
        }
      }
    ]
  }
]
```

### `data/ceo_summary.csv` 列定義

| 列名 | 型 | 説明 |
|------|----|------|
| ticker | str | 証券コード |
| company_name | str | 社名 |
| url | str | 企業IR URL |
| ceo_name | str | 代表取締役社長氏名 |
| ceo_title | str | 役職名 |
| appointment_date | str | 就任年月 |
| photo_saved | bool | 顔写真取得済みフラグ |
| stock_price_at_appointment | float | 就任月の株価（JPY） |
| previous_ceo_count | int | 前任社長データ数 |

---

## Pythonでのデータ読み込み例

```python
import json
import pandas as pd
from pathlib import Path

# JSON読み込み
with open("data/ceo_data.json", encoding="utf-8") as f:
    ceo_data = json.load(f)

# DataFrameに変換
rows = []
for company in ceo_data:
    ceo = company.get("current_ceo") or {}
    price = (ceo.get("stock_price_at_appointment") or {})
    rows.append({
        "ticker": company["ticker"],
        "company": company["company_name"],
        "ceo_name": ceo.get("name"),
        "appointment": ceo.get("appointment_info"),
        "photo_saved": ceo.get("photo_saved", False),
        "stock_price": price.get("close_on_date"),
        "photo_path": f"photos/{company['ticker']}_{company['company_name']}/current/photo.jpg"
            if ceo.get("photo_saved") else None,
    })

df = pd.DataFrame(rows)
print(df[df["photo_saved"]].head())
```

```python
# 写真とメタデータを組み合わせて読み込む
from PIL import Image

def load_ceo_sample(ticker: str, company_name: str) -> dict:
    photo_path = Path(f"photos/{ticker}_{company_name}/current/photo.jpg")
    info_path = photo_path.parent / "info.json"

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    img = Image.open(photo_path) if photo_path.exists() else None
    return {"info": info, "image": img}
```

---

## 収集統計

| 項目 | 数値 |
|------|------|
| 対象企業総数 | 3,727社 |
| CEO顔写真取得数 | 収集中 |
| CEO就任日取得率 | 収集中 |
| 前任社長情報あり | 収集中 |
| 株価データ対応 | yfinance (.T suffix) |

---

## データソース

| データ種別 | 出典 |
|-----------|------|
| 企業一覧 | 会社四季報2026年1集（新春号） |
| CEO情報・写真 | 各社公式IR・コーポレートサイト |
| 前任社長情報 | 各社プレスリリース |
| 株価データ | Yahoo Finance Japan (yfinance) |

---

## 注意事項

- 顔写真の著作権は各社に帰属します。研究・教育目的での利用に限定してください。
- 株価データはYahoo Finance Japan経由で取得しており、正確性を保証するものではありません。
- CEO就任日は公開情報から取得していますが、網羅的でない場合があります。

---

*収集期間: 2026年2月〜 | Python: yfinance, requests, BeautifulSoup, Pillow*
