"""
CEO Photo & History Collector v4
会社四季報2026年1集掲載企業の代表取締役社長情報収集スクリプト

Features:
- 複数写真取得 (photo_01.jpg, photo_02.jpg, ...)
- 2000年以降の全歴代社長データ収集
- 就任時始値 (open_at_appointment) / 退任時終値 (close_at_resignation)
- 短いタイムアウト・少ない再試行で高速化
"""

import sys
import json
import re
import time
import random
import logging
from pathlib import Path
from io import BytesIO
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup
from PIL import Image
import yfinance as yf

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Users\hp\Documents\CEO_Photos_Project")
PHOTOS_DIR = PROJECT_DIR / "photos"
DATA_DIR = PROJECT_DIR / "data"
COMPANIES_FILE = DATA_DIR / "companies.json"
CEO_DATA_FILE = DATA_DIR / "ceo_data.json"
PROGRESS_FILE = DATA_DIR / "progress.json"
LOG_FILE = PROJECT_DIR / "collect.log"

PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
CEO_TITLES_JP = [
    "代表取締役社長", "代表取締役 社長", "取締役社長",
    "代表執行役社長", "代表執行役員社長", "社長執行役員",
    "代表取締役兼社長", "代表取締役CEO", "最高経営責任者",
    "代表取締役会長兼社長", "代表取締役社長執行役員",
    "代表取締役・社長", "取締役 社長",
]

TOP_MGMT_PATHS = [
    "/company/topmessage/",
    "/company/topmessage/index.html",
    "/ir/topmessage/",
    "/company/officer/",
    "/company/officer/index.html",
    "/ir/company/officer/",
    "/ir/officer/",
    "/company/management/",
    "/corporate/officer/",
    "/ir/governance/",
    "/ir/corp_gov/",
    "/company/profile/",
]

LINK_KW = ["役員", "経営陣", "取締役", "社長", "management", "officer",
           "topmessage", "governance", "ガバナンス", "トップメッセージ"]

ICON_PATTERNS = [
    "logo", "icon", "banner", "button", "arrow", "sprite", "bg_", "bg-",
    "back", "mark", "symbol", "badge", "nav", "menu", "footer", "header",
    ".gif", "blank", "noimage", "no-image", "placeholder", "default",
    "bullet", "check", "close", "next", "prev", "search", "home",
    "mail", "tel", "pdf", "xls", "doc", "sns", "social", "facebook",
    "twitter", "instagram", "youtube", "linkedin",
]

HTTP_TIMEOUT = 8
HTTP_RETRIES = 1

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
})
SESSION.max_redirects = 5


# ─── HTTP ─────────────────────────────────────────────────────────────────────

def safe_get(url: str, timeout: int = HTTP_TIMEOUT) -> requests.Response | None:
    for attempt in range(HTTP_RETRIES + 1):
        try:
            r = SESSION.get(url, timeout=timeout, allow_redirects=True, verify=False)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404, 410, 429):
                return None
        except Exception:
            if attempt < HTTP_RETRIES:
                time.sleep(0.5)
    return None


def decode_resp(resp: requests.Response) -> str:
    for enc in ("utf-8", resp.apparent_encoding or "shift_jis", "euc-jp", "shift_jis"):
        try:
            return resp.content.decode(enc)
        except Exception:
            continue
    return resp.content.decode("utf-8", errors="replace")


def get_soup(url: str) -> tuple[BeautifulSoup | None, str]:
    r = safe_get(url)
    if not r:
        return None, url
    text = decode_resp(r)
    return BeautifulSoup(text, "lxml"), r.url


# ─── Image helpers ────────────────────────────────────────────────────────────

def is_icon(src: str) -> bool:
    sl = src.lower()
    return any(p in sl for p in ICON_PATTERNS)


def score_img(img) -> int:
    src = (img.get("src") or img.get("data-src") or "").lower()
    alt = (img.get("alt") or "").lower()
    if is_icon(src):
        return -100
    score = 0
    if any(k in alt for k in ["社長", "取締役", "会長", "president", "ceo", "officer", "代表"]):
        score += 60
    if any(k in src for k in ["photo", "img", "face", "portrait", "president", "officer", "person"]):
        score += 20
    if src.endswith((".jpg", ".jpeg", ".png", ".webp")):
        score += 10
    try:
        w, h = int(img.get("width", 0)), int(img.get("height", 0))
        if 60 <= w <= 600 and 60 <= h <= 800:
            score += 25
        elif w < 40 or h < 40:
            score -= 60
    except Exception:
        pass
    return score


def download_image(url: str, path: Path) -> bool:
    """Download and save a single image. Returns True on success."""
    try:
        r = SESSION.get(url, timeout=HTTP_TIMEOUT * 2, verify=False)
        if r.status_code != 200 or len(r.content) < 3000:
            return False
        img = Image.open(BytesIO(r.content))
        if img.width < 60 or img.height < 60:
            return False
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def download_multiple_photos(photo_urls: list[str], photo_dir: Path) -> list[str]:
    """
    Download multiple photos for a CEO.
    Saves as photo_01.jpg, photo_02.jpg, ...
    Returns list of saved filenames.
    """
    saved = []
    idx = 1
    seen_urls = set()
    for url in photo_urls:
        if url in seen_urls or not url:
            continue
        seen_urls.add(url)
        path = photo_dir / f"photo_{idx:02d}.jpg"
        if path.exists():
            saved.append(path.name)
            idx += 1
            continue
        if download_image(url, path):
            saved.append(path.name)
            idx += 1
        if idx > 5:  # Max 5 photos per CEO
            break
    return saved


# ─── CEO Detection ─────────────────────────────────────────────────────────────

def is_ceo_title(text: str) -> bool:
    return any(t in text for t in CEO_TITLES_JP)


def extract_person_name(text: str) -> str:
    for t in CEO_TITLES_JP:
        text = text.replace(t, " ")
    bad = ["取締役", "社長", "会社", "株式", "本社", "設立", "概要", "情報", "事業",
           "製品", "採用", "管理", "技術", "経営", "商号", "業務", "代表", "役員"]
    for pat in [
        r"[\u4e00-\u9fff]{1,2}[\s\u3000]{1,2}[\u4e00-\u9fff]{1,3}(?!\s*[社会株])",
        r"[\u4e00-\u9fff]{2,4}(?![年月日社会株])",
    ]:
        m = re.search(pat, text)
        if m:
            candidate = m.group().strip()
            if not any(b in candidate for b in bad) and len(candidate.replace(" ", "").replace("\u3000", "")) >= 2:
                return candidate
    return ""


def extract_appt_date(text: str) -> str | None:
    for pat in [
        r"(20\d{2}年\d{1,2}月)",
        r"(令和\d{1,2}年\d{1,2}月)",
        r"(平成\d{1,2}年\d{1,2}月)",
        r"(\d{4}/\d{1,2}(?:/\d{1,2})?)",
    ]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def find_ceo_photos_in_soup(soup: BeautifulSoup, page_url: str) -> dict | None:
    """
    Find CEO info + ALL related photos in a page.
    Returns dict with: name, title, photo_urls (list), appointment_info, source_url
    """
    if not soup:
        return None
    full_text = soup.get_text()
    if not is_ceo_title(full_text):
        return None

    best = None
    best_score = -999

    for title in CEO_TITLES_JP:
        for elem in soup.find_all(string=lambda s: s and title in s):
            container = elem.parent
            for _ in range(8):
                if container is None or container.name in ("html", "body"):
                    break
                ctxt = container.get_text()
                if len(ctxt) > 3000:
                    container = container.parent
                    continue
                imgs = container.find_all("img")
                valid_imgs = [(img, score_img(img)) for img in imgs
                              if (img.get("src") or img.get("data-src")) and score_img(img) > -10]
                if valid_imgs:
                    top_score = max(s for _, s in valid_imgs)
                    if top_score > best_score:
                        best_score = top_score
                        photo_urls = []
                        for img, s in sorted(valid_imgs, key=lambda x: -x[1]):
                            src = img.get("src") or img.get("data-src") or ""
                            if src:
                                photo_urls.append(urljoin(page_url, src))
                        name = extract_person_name(ctxt)
                        appt = extract_appt_date(ctxt)
                        best = {
                            "name": name,
                            "title": title,
                            "photo_urls": photo_urls,
                            "appointment_info": appt,
                            "source_url": page_url,
                        }
                container = container.parent

    if best and best_score > -10:
        return best

    # Fallback: search in sections
    for tag in ("section", "article", "div", "table"):
        for el in soup.find_all(tag, limit=150):
            txt = el.get_text()
            if len(txt) > 2500 or not is_ceo_title(txt):
                continue
            valid_imgs = []
            for img in el.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                s = score_img(img) if src else -999
                if src and s > -50:
                    valid_imgs.append((img, s, src))
            if valid_imgs:
                top_score = max(s for _, s, _ in valid_imgs)
                if top_score > best_score:
                    best_score = top_score
                    title = next((t for t in CEO_TITLES_JP if t in txt), CEO_TITLES_JP[0])
                    photo_urls = [urljoin(page_url, src)
                                  for img, s, src in sorted(valid_imgs, key=lambda x: -x[1])]
                    best = {
                        "name": extract_person_name(txt),
                        "title": title,
                        "photo_urls": photo_urls,
                        "appointment_info": extract_appt_date(txt),
                        "source_url": page_url,
                    }

    return best if (best and best_score > -50) else None


# ─── Management Page Finder ────────────────────────────────────────────────────

def find_mgmt_pages(base_url: str) -> list[str]:
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    found = []

    for path in TOP_MGMT_PATHS:
        r = safe_get(base + path)
        if r and r.url:
            try:
                text = decode_resp(r)
            except Exception:
                continue
            if is_ceo_title(text):
                found.append(r.url)
                if len(found) >= 2:
                    return found

    soup, _ = get_soup(base_url)
    if soup:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            txt = a.get_text()
            if any(kw in href.lower() or kw in txt for kw in LINK_KW):
                full = urljoin(base_url, href)
                if urlparse(full).netloc == parsed.netloc and full not in found:
                    r2 = safe_get(full)
                    if r2:
                        try:
                            t2 = decode_resp(r2)
                        except Exception:
                            continue
                        if is_ceo_title(t2):
                            found.append(r2.url)
                            if len(found) >= 2:
                                return found
    return found


# ─── Stock Price ──────────────────────────────────────────────────────────────

def parse_date(date_str: str) -> datetime | None:
    """Parse Japanese date string to datetime."""
    # Wareki conversion
    m = re.search(r"令和(\d+)年(\d{1,2})月(?:(\d{1,2})日)?", date_str)
    if m:
        y = 2018 + int(m.group(1))
        mo = int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 1
        return datetime(y, mo, d)
    m = re.search(r"平成(\d+)年(\d{1,2})月(?:(\d{1,2})日)?", date_str)
    if m:
        y = 1988 + int(m.group(1))
        mo = int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 1
        return datetime(y, mo, d)
    m = re.search(r"(\d{4})年(\d{1,2})月(?:(\d{1,2})日)?", date_str)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 1
        # Only accept 2000 onwards
        if y < 2000:
            return None
        return datetime(y, mo, d)
    m = re.search(r"(\d{4})/(\d{1,2})(?:/(\d{1,2}))?", date_str)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        d = int(m.group(3)) if m.group(3) else 1
        if y < 2000:
            return None
        return datetime(y, mo, d)
    return None


def get_stock_prices(ticker: str, date_str: str) -> dict | None:
    """
    Get stock prices around a CEO transition date.
    Returns:
      - open_on_date: Opening price on or near the date (for appointment)
      - close_on_date: Closing price on or near the date (for resignation)
    """
    try:
        dt = parse_date(date_str)
        if not dt:
            return None
        # Skip future dates
        if dt > datetime.now():
            return None

        start = (dt - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=45)).strftime("%Y-%m-%d")

        hist = yf.Ticker(f"{ticker}.T").history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None

        # Find closest trading day
        try:
            idx = hist.index.get_indexer([dt], method="nearest")[0]
            idx = max(0, min(idx, len(hist) - 1))
        except Exception:
            idx = 0

        row = hist.iloc[idx]
        return {
            "date": dt.strftime("%Y-%m-%d"),
            "open_on_date": round(float(row["Open"]), 2),
            "close_on_date": round(float(row["Close"]), 2),
            "high_on_date": round(float(row["High"]), 2),
            "low_on_date": round(float(row["Low"]), 2),
            "trading_date": str(hist.index[idx].date()),
            "currency": "JPY",
        }
    except Exception as e:
        log.debug(f"Stock price error {ticker}: {e}")
        return None


# ─── Historical CEO Search ─────────────────────────────────────────────────────

def search_historical_ceos(base_url: str, ticker: str) -> list[dict]:
    """
    Search for all CEOs since 2000 from:
    1. Company press releases / news
    2. IR pages
    Returns list sorted by appointment_date desc (newest first).
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    results = []
    seen_names = set()

    news_paths = [
        "/news/", "/ir/news/", "/press/", "/ir/release/",
        "/newsrelease/", "/ir/press/", "/news/topics/",
        "/ir/news/release/", "/news/release/",
    ]
    ceo_kw = ["社長交代", "社長就任", "代表取締役社長", "退任", "人事異動", "役員改選", "社長に就任", "新社長"]

    for path in news_paths[:5]:
        url = base + path
        soup, _ = get_soup(url)
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            link_text = a.get_text()
            if any(k in link_text for k in ceo_kw):
                news_url = urljoin(url, a["href"])
                nsoup, nurl = get_soup(news_url)
                if nsoup:
                    entries = _parse_historical_ceo_news(nsoup.get_text(), nurl, ticker)
                    for e in entries:
                        key = e.get("name", "")
                        if key and key not in seen_names:
                            seen_names.add(key)
                            results.append(e)
        if len(results) >= 10:
            break

    # Sort by appointment_date descending
    def sort_key(e):
        d = e.get("appointment_date") or ""
        dt = parse_date(d)
        return dt or datetime(2000, 1, 1)

    results.sort(key=sort_key, reverse=True)
    return results[:6]  # Up to 6 previous CEOs (since 2000)


def _parse_historical_ceo_news(text: str, url: str, ticker: str) -> list[dict]:
    """Parse a news article for CEO change info. Returns list of entries."""
    if not (is_ceo_title(text) and any(k in text for k in ["就任", "退任"])):
        return []

    results = []
    # Find all mentions of CEO changes
    patterns = [
        r"([\u4e00-\u9fff]{1,2}[\s　][\u4e00-\u9fff]{1,3}).*?(?:が|は)?(?:代表取締役社長|社長).*?就任",
        r"新社長.*?([\u4e00-\u9fff]{1,2}[\s　]?[\u4e00-\u9fff]{1,3})氏",
        r"([\u4e00-\u9fff]{1,2}[\s　]?[\u4e00-\u9fff]{1,3}).*?新社長",
        r"([\u4e00-\u9fff]{1,2}[\s　]?[\u4e00-\u9fff]{1,3})氏.*?社長",
    ]

    found_names = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if name and name not in found_names:
                found_names.append(name)

    dates = re.findall(r"(?:20\d{2}|平成\d+|令和\d+)年\d{1,2}月(?:\d{1,2}日)?", text)

    for i, name in enumerate(found_names[:3]):
        appt = dates[i * 2] if i * 2 < len(dates) else (dates[0] if dates else None)
        res = dates[i * 2 + 1] if i * 2 + 1 < len(dates) else None

        # Only include 2000 onwards
        if appt:
            dt = parse_date(appt)
            if dt and dt.year < 2000:
                continue

        entry = {
            "name": name,
            "appointment_date": appt,
            "resignation_date": res,
            "source_url": url,
        }

        # Get stock prices
        if appt:
            price = get_stock_prices(ticker, appt)
            if price:
                entry["stock_price_at_appointment"] = price
                entry["open_at_appointment"] = price.get("open_on_date")
        if res:
            price = get_stock_prices(ticker, res)
            if price:
                entry["stock_price_at_resignation"] = price
                entry["close_at_resignation"] = price.get("close_on_date")

        results.append(entry)

    return results


# ─── Safe dir name ────────────────────────────────────────────────────────────

def safe_name(s: str) -> str:
    return re.sub(r'[^\w\u3040-\u30ff\u4e00-\u9fff\-]', '_', str(s))[:50]


# ─── Main company processor ───────────────────────────────────────────────────

def process_company(company: dict) -> dict:
    ticker = company["ticker"]
    name = company["name"]
    url = company.get("url", "")

    result = {
        "ticker": ticker,
        "company_name": name,
        "url": url,
        "current_ceo": None,
        "previous_ceos": [],
        "error": None,
        "processed_at": datetime.now().isoformat(),
    }

    if not url or url in ("None", "nan", ""):
        result["error"] = "no_url"
        return result

    log.info(f"[{ticker}] {name}")

    try:
        # ── Find management page(s) ──
        mgmt_urls = find_mgmt_pages(url)
        ceo = None
        for mu in mgmt_urls:
            soup, final_url = get_soup(mu)
            ceo = find_ceo_photos_in_soup(soup, final_url or mu)
            if ceo:
                break

        # Fallback: company top page
        if not ceo:
            soup, final_url = get_soup(url)
            ceo = find_ceo_photos_in_soup(soup, final_url or url)

        if ceo:
            sn = safe_name(name)
            photo_dir = PHOTOS_DIR / f"{ticker}_{sn}" / "current"
            photo_dir.mkdir(parents=True, exist_ok=True)

            # Download multiple photos
            photo_urls = ceo.pop("photo_urls", [])
            saved_files = download_multiple_photos(photo_urls, photo_dir)
            ceo["photos_saved"] = saved_files
            ceo["photo_count"] = len(saved_files)
            # Backward compat: primary photo
            ceo["photo_saved"] = len(saved_files) > 0

            # Stock prices at appointment (open price)
            if ceo.get("appointment_info"):
                price = get_stock_prices(ticker, ceo["appointment_info"])
                if price:
                    ceo["stock_price_at_appointment"] = price
                    ceo["open_at_appointment"] = price.get("open_on_date")

            (photo_dir / "info.json").write_text(
                json.dumps(ceo, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["current_ceo"] = ceo
        else:
            result["error"] = "ceo_not_found"

        # ── Historical CEOs (since 2000) ──
        try:
            prev_ceos = search_historical_ceos(url, ticker)
            for i, prev in enumerate(prev_ceos, 1):
                sn = safe_name(name)
                pn = safe_name(prev.get("name", "unknown"))
                hist_dir = PHOTOS_DIR / f"{ticker}_{sn}" / "history" / f"{i:02d}_{pn}"
                hist_dir.mkdir(parents=True, exist_ok=True)
                (hist_dir / "info.json").write_text(
                    json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            result["previous_ceos"] = prev_ceos
            if prev_ceos:
                log.info(f"  歴代社長: {len(prev_ceos)}名")
        except Exception as e:
            log.debug(f"  前任社長検索エラー: {e}")

    except Exception as e:
        result["error"] = str(e)
        log.warning(f"  Error [{ticker}]: {e}")

    time.sleep(random.uniform(0.2, 0.5))
    return result
