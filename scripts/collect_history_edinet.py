#!/usr/bin/env python3
"""
collect_history_edinet.py  –  EDINET有価証券報告書から歴代社長を収集

戦略:
  1. 2003-2024年の各期末(6月末・3月末・9月末・12月末)に提出された有価証券報告書を収集
  2. 各報告書から TitleAndNameOfRepresentativeCoverPage (代表者の役職氏名) を抽出
  3. 年次変化から社長就任・退任タイミングを特定
  4. 結果を history_supplement.json に統合

特徴:
  - Wikipedia未収録企業を優先
  - レート制限対策（1秒間隔）
  - 50社ごとにGitHubにコミット
"""

import json
import logging
import os
import re
import subprocess
import time
import zipfile
import io
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

import requests

# ── パス設定 ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
DATA_DIR    = PROJECT_DIR / "data"
LOG_FILE    = PROJECT_DIR / "collect_history_edinet.log"

EDINET_KEY = "a571566d8d174e52974a00c86f946dc6"
EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"

GH_TOKEN  = os.environ.get("GH_TOKEN", "")
GH_USER   = "yukizi1113"
REPO_NAME = "JP-Listed-Company-CEO-Photos"

# ── ログ設定 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── セッション ────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CEO-Research-Bot/1.0 (academic; github.com/yukizi1113)"
})


# ── 名前バリデーション ────────────────────────────────────────────────
_NAME_BAD = [
    "代表", "取締役", "社長", "会社", "株式", "在任", "就任",
    "会長", "業務", "管理", "担当", "執行", "監査", "常務", "専務",
]
_JP_CHARS = re.compile(r'^[\u4e00-\u9fff\u3040-\u30ff]+$')


def _is_valid_jp_name(text: str) -> bool:
    text = text.strip()
    if not text or len(text) < 2:
        return False
    if any(b in text for b in _NAME_BAD):
        return False
    if re.search(r'[\s\u3000]', text):
        parts = re.split(r'[\s\u3000]+', text.strip())
        if len(parts) != 2:
            return False
        family, given = parts
        if not (2 <= len(family) <= 5 and 1 <= len(given) <= 5):
            return False
        return bool(_JP_CHARS.match(family) and _JP_CHARS.match(given))
    if not (2 <= len(text) <= 7):
        return False
    return bool(_JP_CHARS.match(text))


def extract_name_from_title(title_str: str) -> str:
    """
    '代表取締役社長  池見  賢' → '池見 賢'
    '代表取締役社長（CEO）  伊藤  滋' → '伊藤 滋'
    """
    # 役職部分を除去
    s = re.sub(r'^[\u3000-\u30ff\u4e00-\u9fff（）()（）A-Za-z\s　]+(?=\s{2,}|\u3000{2,})', '', title_str)
    s = s.strip()
    # 複数スペースを1スペースに
    s = re.sub(r'[\s\u3000]{2,}', ' ', s)
    s = s.strip()
    if _is_valid_jp_name(s):
        return s
    # 代替: 最後のスペース区切りトークンが名前かもしれない
    # 全角スペースで分割して末尾から名前を探す
    parts = re.split(r'[\s\u3000]+', title_str.strip())
    name_parts = []
    for p in reversed(parts):
        p = p.strip()
        if _JP_CHARS.match(p) and 2 <= len(p) <= 5 and not any(b in p for b in _NAME_BAD):
            name_parts.insert(0, p)
        elif name_parts:
            break  # 名前部分が終わった
    name = ' '.join(name_parts)
    if _is_valid_jp_name(name):
        return name
    return ""


# ── EDINET API 呼び出し ───────────────────────────────────────────────
def edinet_documents_for_date(target_date: str) -> list[dict]:
    """指定日に提出された有価証券報告書(docTypeCode=120)を取得。"""
    try:
        r = SESSION.get(
            f"{EDINET_BASE}/documents.json",
            params={"date": target_date, "type": 2, "Subscription-Key": EDINET_KEY},
            timeout=20
        )
        if r.status_code != 200:
            return []
        data = r.json()
        results = data.get("results") or []
        # docTypeCode=120 (有価証券報告書) のみ
        return [d for d in results if d and d.get("docTypeCode") == "120" and d.get("secCode")]
    except Exception as e:
        log.debug(f"EDINET date {target_date}: {e}")
        return []


def build_seccode_to_edinet_map(years: list[int]) -> dict[str, list[dict]]:
    """
    各年の主要提出日を走査して secCode → [{year, docID, periodEnd, ...}] マッピングを構築。
    secCode は ticker4桁 + '0' の5桁。
    """
    log.info("EDINETコードマッピングを構築中...")
    mapping: dict[str, list[dict]] = defaultdict(list)

    # 各年の主な提出日(3月決算:6月末、12月決算:3月末、9月決算:12月末)
    # 提出期限の前後1週間をカバー
    key_dates = []
    for year in years:
        # 3月決算 → 6月中旬〜末
        for day in range(20, 31):
            d = f"{year}-06-{day:02d}"
            if date(year, 6, day).weekday() < 5:  # 平日のみ
                key_dates.append(d)
        # 12月決算 → 3月中旬〜末
        for day in range(20, 32):
            try:
                d = f"{year}-03-{day:02d}"
                if date(year, 3, day).weekday() < 5:
                    key_dates.append(d)
            except ValueError:
                pass
        # 9月決算 → 12月中旬〜末
        for day in range(20, 31):
            d = f"{year}-12-{day:02d}"
            if date(year, 12, day).weekday() < 5:
                key_dates.append(d)

    # 重複除去
    key_dates = sorted(set(key_dates))
    log.info(f"走査する日付数: {len(key_dates)}")

    for i, d in enumerate(key_dates):
        docs = edinet_documents_for_date(d)
        for doc in docs:
            sec = doc.get("secCode", "")
            ticker = sec[:4] if len(sec) == 5 else sec  # 5桁→4桁
            mapping[ticker].append({
                "year": d[:4],
                "docID": doc.get("docID", ""),
                "periodEnd": doc.get("periodEnd", ""),
                "edinetCode": doc.get("edinetCode", ""),
                "filerName": doc.get("filerName", ""),
                "submitDate": d,
            })
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{len(key_dates)} 日付処理済み, マッピング数: {len(mapping)}")
        time.sleep(0.3)  # レート制限

    return dict(mapping)


def get_ceo_from_doc(doc_id: str) -> str:
    """docIDから有価証券報告書CSVをダウンロードして社長名を取得。"""
    try:
        r = SESSION.get(
            f"{EDINET_BASE}/documents/{doc_id}",
            params={"type": 5, "Subscription-Key": EDINET_KEY},
            timeout=30
        )
        if r.status_code != 200:
            return ""
        z = zipfile.ZipFile(io.BytesIO(r.content))
        for name in z.namelist():
            if "jpcrp030000" in name and name.endswith(".csv"):
                raw = z.read(name)
                # UTF-16-LE デコード
                content = raw.decode("utf-16-le", errors="replace")
                for line in content.split("\r\n"):
                    if "TitleAndNameOfRepresentativeCoverPage" in line:
                        # 最後のフィールドが役職名氏名
                        parts = line.split("\t")
                        if parts:
                            title_str = parts[-1].strip().strip('"')
                            name = extract_name_from_title(title_str)
                            if name:
                                return name
        return ""
    except Exception as e:
        log.debug(f"  docID {doc_id} エラー: {e}")
        return ""


def collect_ceo_timeline(ticker: str, docs: list[dict]) -> list[dict]:
    """
    複数年の書類からCEOタイムラインを構築。
    年次変化を検出して就任・退任年を特定。
    """
    # 年→CEO名 マッピング
    year_ceo: dict[str, str] = {}
    docs_sorted = sorted(docs, key=lambda x: x.get("periodEnd", x.get("submitDate", "")))

    for doc in docs_sorted:
        doc_id = doc.get("docID", "")
        year = (doc.get("periodEnd") or doc.get("submitDate", ""))[:4]
        if not doc_id or not year:
            continue
        if year in year_ceo:
            continue  # 同年は1件のみ
        ceo = get_ceo_from_doc(doc_id)
        if ceo:
            year_ceo[year] = ceo
            log.debug(f"  {year}: {ceo}")
        time.sleep(0.5)

    if not year_ceo:
        return []

    # 年次変化を検出してCEO就任期間を計算
    timeline: list[dict] = []
    years_sorted = sorted(year_ceo.keys())
    prev_ceo = None
    prev_year = None

    for year in years_sorted:
        ceo = year_ceo[year]
        if ceo != prev_ceo:
            # CEOが変わった → 就任記録
            # 退任年を前の期間の最終年として推定
            if prev_ceo and timeline:
                timeline[-1]["resignation_date"] = f"{year}-06-30"  # 概算

            timeline.append({
                "name": ceo,
                "appointment_date": f"{year}-01-01",  # 概算（年初）
                "resignation_date": None,
                "source": f"edinet_annual_report",
            })
        prev_ceo = ceo
        prev_year = year

    return timeline


# ── 写真取得（Wikipedia） ────────────────────────────────────────────
def search_photo_wikipedia(name: str) -> str | None:
    try:
        from urllib.parse import quote
        r = SESSION.get(f"https://ja.wikipedia.org/wiki/{quote(name)}", timeout=10)
        if r.status_code != 200:
            return None
        from bs4 import BeautifulSoup
        sp = BeautifulSoup(r.text, "lxml")
        infobox = sp.find("table", {"class": lambda c: c and "infobox" in " ".join(c)})
        img = infobox.find("img") if infobox else None
        if not img:
            img = sp.find("img", src=re.compile(r"upload\.wikimedia"))
        if img and img.get("src"):
            src = img["src"]
            if src.startswith("//"):
                src = "https:" + src
            if not any(b in src.lower() for b in ["svg", "disambig", "question", "noimage"]):
                return src
    except Exception:
        pass
    return None


def download_photo(url: str, dest: Path) -> bool:
    try:
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200 or len(r.content) < 2000:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s　]', "_", s).strip("_")


def photos_dir(ticker: str) -> Path:
    try:
        t = int(ticker)
        if t < 3500: return PROJECT_DIR / "photos_1"
        if t < 5000: return PROJECT_DIR / "photos_2"
        if t < 7000: return PROJECT_DIR / "photos_3"
        if t < 9000: return PROJECT_DIR / "photos_4"
        return PROJECT_DIR / "photos_5"
    except ValueError:
        return PROJECT_DIR / "photos_5"


# ── コミット&プッシュ ──────────────────────────────────────────────────
def commit_and_push(batch_num):
    add_paths = ["data/history_supplement.json"]
    for d in ["photos_1", "photos_2", "photos_3", "photos_4", "photos_5"]:
        if (PROJECT_DIR / d).exists():
            add_paths.append(f"{d}/")
    subprocess.run(
        ["git", "-c", "core.quotePath=false", "add", "--ignore-removal"] + add_paths,
        cwd=str(PROJECT_DIR), capture_output=True
    )
    msg = f"[edinet] batch {batch_num}: EDINET歴代CEO追加"
    r = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=str(PROJECT_DIR), capture_output=True
    )
    if r.returncode != 0:
        log.warning("コミット対象なし")
        return
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(PROJECT_DIR), capture_output=True
    ).stdout.decode().strip()
    subprocess.run(
        ["git", "push", remote, "master"],
        cwd=str(PROJECT_DIR), capture_output=True
    )
    log.info(f"プッシュ完了 batch {batch_num}")
    for d in ["photos_1", "photos_2", "photos_3", "photos_4", "photos_5"]:
        p = PROJECT_DIR / d
        if p.exists():
            import shutil
            shutil.rmtree(p)


# ── メイン ────────────────────────────────────────────────────────────
def main():
    # データ読み込み
    ceo_data = json.loads((DATA_DIR / "ceo_data.json").read_text(encoding="utf-8"))
    ceo_by_ticker = {str(c["ticker"]): c for c in ceo_data}

    hs = json.loads((DATA_DIR / "history_supplement.json").read_text(encoding="utf-8"))
    hs_by_ticker = {str(x["ticker"]): x for x in hs}

    # 歴代データが少ない会社をターゲット (Wikipedia で見つからなかった会社)
    targets = []
    for c in ceo_data:
        ticker = str(c["ticker"])
        entry = hs_by_ticker.get(ticker)
        n_prev = len(entry.get("previous_ceos", [])) if entry else 0
        if n_prev == 0:  # 歴代データなし
            targets.append(c)

    log.info(f"EDINET対象: {len(targets)}社")

    # EDINET年次マッピングを構築（全3727社に使えるように一括構築）
    years = list(range(2003, datetime.now().year + 1))

    # キャッシュファイルがあれば読み込む
    cache_path = DATA_DIR / "edinet_mapping.json"
    if cache_path.exists():
        log.info("EDINETマッピングキャッシュを読み込み")
        edinet_map = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        edinet_map = build_seccode_to_edinet_map(years)
        cache_path.write_text(
            json.dumps(edinet_map, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info(f"EDINETマッピング保存: {len(edinet_map)}社")

    # 処理ループ
    BATCH_SIZE = 50
    batch_count = 0
    total_added = 0

    for ci, company in enumerate(targets, 1):
        ticker = str(company["ticker"])
        cname = company.get("company_name", "")

        docs = edinet_map.get(ticker, [])
        if not docs:
            log.debug(f"[{ci}/{len(targets)}] {ticker} {cname}: EDINETなし")
            continue

        log.info(f"[{ci}/{len(targets)}] {ticker} {cname}: {len(docs)}年分の報告書")

        # タイムライン構築
        timeline = collect_ceo_timeline(ticker, docs)
        if not timeline:
            continue

        # 既存データとマージ
        existing = hs_by_ticker.get(ticker, {})
        existing_names = {c.get("name", "") for c in existing.get("previous_ceos", [])}
        new_ceos = [c for c in timeline if c.get("name") and c["name"] not in existing_names]

        if not new_ceos:
            continue

        # 写真取得
        pdir = photos_dir(ticker)
        folder = pdir / f"{ticker}_{safe_name(cname)}" / "history"
        for i, ceo in enumerate(new_ceos):
            name = ceo.get("name", "")
            ceo_folder = folder / f"{i+1:02d}_{safe_name(name)}"
            photo_url = search_photo_wikipedia(name)
            if photo_url:
                ceo_folder.mkdir(parents=True, exist_ok=True)
                dest = ceo_folder / "photo_01.jpg"
                if download_photo(photo_url, dest):
                    ceo["photo_path"] = str(dest.relative_to(PROJECT_DIR))
                    log.info(f"  写真取得: {name}")
            time.sleep(0.3)

        # history_supplement 更新
        if ticker in hs_by_ticker:
            hs_by_ticker[ticker]["previous_ceos"].extend(new_ceos)
            for x in hs:
                if str(x["ticker"]) == ticker:
                    x["previous_ceos"].extend(new_ceos)
                    break
        else:
            entry = {
                "ticker": ticker,
                "company_name": cname,
                "previous_ceos": new_ceos,
            }
            hs.append(entry)
            hs_by_ticker[ticker] = entry

        (DATA_DIR / "history_supplement.json").write_text(
            json.dumps(hs, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        total_added += len(new_ceos)
        batch_count += 1
        log.info(f"  {ticker} {cname}: {len(new_ceos)}名追加")

        if batch_count % BATCH_SIZE == 0:
            commit_and_push(batch_count // BATCH_SIZE)

    # 最終コミット
    if batch_count % BATCH_SIZE != 0:
        commit_and_push("final")

    log.info(f"完了: {total_added}名追加")


if __name__ == "__main__":
    main()
