# 日本上場企業 代表取締役社長 顔写真・経歴データ集

会社四季報2026年1集（新春号）掲載の全上場企業（3,727社）を対象に、
代表取締役社長の顔写真・就任退任日・株価データを収集したリポジトリです。

## データ構成

```
photos/
  {ticker}_{社名}/
    current/
      photo.jpg        # 現社長の顔写真
      info.json        # 氏名・役職・就任日・出典URL・就任時株価
    history/
      01_{氏名}/
        photo.jpg      # 前社長の顔写真（取得できた場合）
        info.json      # 氏名・就任日・退任日・退任時株価
      02_{氏名}/       # 更に前の社長
      03_{氏名}/       # 更に更に前の社長
data/
  companies.json       # 全対象企業リスト（ticker, 社名, URL）
  ceo_data.json        # 全CEO情報JSON
  progress.json        # 収集進捗
scripts/
  collect_ceo.py       # メイン収集スクリプト
  make_summary.py      # サマリーCSV生成スクリプト
```

## info.json フォーマット

### current/info.json
```json
{
  "name": "山田 太郎",
  "title": "代表取締役社長",
  "photo_url": "https://example.co.jp/...jpg",
  "appointment_info": "2021年6月",
  "source_url": "https://example.co.jp/company/officer/",
  "photo_saved": true,
  "stock_price_at_appointment": {
    "date": "2021-06",
    "close_on_date": 1234.0,
    "close_min": 1100.0,
    "close_max": 1350.0,
    "currency": "JPY"
  }
}
```

### history/XX_氏名/info.json
```json
{
  "name": "田中 次郎",
  "appointment_date": "2015年6月",
  "resignation_date": "2021年6月",
  "source_url": "https://...",
  "stock_price_at_resignation": {
    "date": "2021-06",
    "close_on_date": 1234.0,
    "currency": "JPY"
  }
}
```

## データ出典

- 企業一覧：会社四季報2026年1集（新春号）
- 役員情報：各社公式IRサイト / プレスリリース
- 株価データ：yfinance (Yahoo Finance Japan)

## 収集対象企業数

| 項目 | 数 |
|------|-----|
| 対象企業総数 | 3,727社 |
| CEO顔写真取得 | 集計中 |
| CEO就任日把握 | 集計中 |
| 前任社長情報 | 集計中 |

---
*収集日: 2026年2月*
