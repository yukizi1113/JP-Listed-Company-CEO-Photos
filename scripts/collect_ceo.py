"""
CEO Photo & History Collector for Japanese Listed Companies
会社四季報2026年1集掲載企業の代表取締役社長情報収集スクリプト

Collects:
- Current CEO photo
- CEO appointment date
- Previous CEOs (up to 3 predecessors) with photos and dates
- Stock prices at CEO transitions (yfinance)
"""

import sys
import json
import os
import re
import time
import random
import logging
from pathlib import Path
from io import BytesIO
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
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
    "代表取締役社長",
    "代表取締役 社長",
    "取締役社長",
    "代表執行役社長",
    "代表執行役員社長",
    "社長執行役員",
    "代表取締役兼社長",
    "代表取締役CEO",
    "最高経営責任者",
    "代表取締役会長兼社長",
    "代表取締役社長執行役員",
    "代表取締役・社長",
    "取締役 社長",
]

# Ordered by priority - more specific patterns first
MGMT_PATH_PATTERNS = [
    # Top message pages (most likely to have CEO photo)
    "/company/topmessage/index.html",
    "/company/topmessage/",
    "/company/topmessage",
    "/ir/topmessage/index.html",
    "/ir/topmessage/",
    "/about/topmessage/",
    "/topmessage/",
    "/company/message/",
    "/company/message/index.html",
    "/ir/message/",
    # Officer/management pages
    "/company/officer/index.html",
    "/company/officer/",
    "/company/officers/",
    "/company/officer",
    "/ir/company/officer/",
    "/ir/company/officer",
    "/ir/officer/",
    "/company/management/",
    "/company/management",
    "/company/management/index.html",
    "/about/management/",
    "/about/management",
    "/about/officers/",
    "/corporate/officer/",
    "/corporate/officer",
    "/corporate/management/",
    "/corporate/officers/",
    "/company/directors/",
    "/company/director/",
    "/company/board/",
    "/company/profile/officer/",
    "/company/info/officer/",
    "/ir/corporate/officer/",
    "/aboutus/officer/",
    "/aboutus/management/",
    # Governance pages
    "/ir/governance/",
    "/ir/corp_gov/",
    "/ir/corp_gov/index.html",
    "/ir/governance/officer/",
    "/governance/",
    "/about/governance/",
    "/corporate/governance/",
    "/company/governance/",
    # Profiles
    "/company/profile/",
    "/company/profile/index.html",
    "/company/profile",
    "/company/about/",
    "/company/about/index.html",
    "/company/outline/",
    "/company/overview/",
    # IR pages
    "/ir/company/",
    "/ir/company/index.html",
    # General about pages
    "/about/",
    "/about/index.html",
    "/aboutus/",
    "/company/",
    "/company/index.html",
]

MGMT_LINK_KEYWORDS = [
    "役員", "経営陣", "取締役", "代表取締役", "社長", "management", "officer", "board",
    "governance", "ガバナンス", "コーポレート", "メッセージ", "topmessage",
    "トップメッセージ", "会社情報", "会社概要", "経営",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.max_redirects = 5

# Known icon/non-photo image patterns
ICON_PATTERNS = [
    "logo", "icon", "banner", "button", "arrow", "bg_", "bg-",
    "back", "sprite", "mark", "symbol", "badge", "nav", "menu",
    "footer", "header", ".gif", "blank", "noimage", "no-image",
    "placeholder", "default", "bullet", "check", "close", "open",
    "next", "prev", "search", "home", "mail", "tel", "pdf", "xls",
    "doc", "link", "new", "top", "bottom", "left", "right",
    "facebook", "twitter", "instagram", "youtube", "linkedin",
    "twitter", "x-icon", "sns", "social",
]


# ─── HTTP ─────────────────────────────────────────────────────────────────────

def safe_get(url: str, timeout: int = 15) -> requests.Response | None:
    """HTTP GET with retry logic and error handling."""
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                return resp
            elif resp.status_code in (403, 404, 410, 429):
                if resp.status_code == 429:
                    time.sleep(5)
                return None
            elif resp.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.SSLError:
            # Try without SSL verification as fallback
            try:
                resp = SESSION.get(url, timeout=timeout, allow_redirects=True, verify=False)
                if resp.status_code == 200:
                    return resp
            except Exception:
                pass
            return None
        except requests.exceptions.TooManyRedirects:
            return None
        except requests.exceptions.ConnectionError:
            time.sleep(1 + attempt)
        except Exception as e:
            if attempt == 2:
                log.debug(f"GET failed {url}: {type(e).__name__}: {e}")
            time.sleep(1 + attempt)
    return None


def get_soup(url: str) -> tuple[BeautifulSoup | None, str | None]:
    """Get BeautifulSoup and final URL after redirects."""
    resp = safe_get(url)
    if not resp:
        return None, None
    try:
        # Try utf-8 first, then apparent encoding
        try:
            text = resp.content.decode("utf-8")
        except UnicodeDecodeError:
            text = resp.content.decode(resp.apparent_encoding or "utf-8", errors="replace")
        return BeautifulSoup(text, "lxml"), resp.url
    except Exception:
        return None, None


# ─── CEO Detection ─────────────────────────────────────────────────────────────

def is_ceo_title(text: str) -> bool:
    return any(t in text for t in CEO_TITLES_JP)


def is_icon_image(src: str) -> bool:
    src_lower = src.lower()
    return any(p in src_lower for p in ICON_PATTERNS)


def score_img(img_tag) -> int:
    """Score an image tag for likelihood of being a person photo."""
    score = 0
    src = img_tag.get("src", "") + img_tag.get("data-src", "")
    alt = img_tag.get("alt", "").lower()
    width = img_tag.get("width", "")
    height = img_tag.get("height", "")

    # Negative: icon patterns
    if is_icon_image(src):
        return -100

    # Positive: photo-like filenames
    if any(kw in src.lower() for kw in ["photo", "img", "picture", "face", "portrait", "person", "staff"]):
        score += 20
    if any(kw in alt for kw in ["社長", "取締役", "会長", "president", "ceo", "officer"]):
        score += 50

    # Size hints
    try:
        w = int(width)
        h = int(height)
        if 80 <= w <= 400 and 80 <= h <= 600:
            score += 30
        elif w < 50 or h < 50:
            score -= 50
    except (ValueError, TypeError):
        pass

    # JPEG/PNG likely a photo
    if src.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        score += 10

    return score


def find_ceo_in_soup(soup: BeautifulSoup, page_url: str) -> dict | None:
    """
    Find CEO information in a BeautifulSoup object.
    Returns dict: {name, title, photo_url, appointment_info, source_url}
    """
    if not soup:
        return None

    text = soup.get_text()
    if not is_ceo_title(text):
        return None

    best_result = None
    best_score = -999

    for title in CEO_TITLES_JP:
        # Find all elements containing this title
        for elem in soup.find_all(string=lambda s: s and title in s):
            container = elem.parent

            # Search upward for a photo container
            for depth in range(10):
                if container is None:
                    break
                imgs = container.find_all("img")
                for img in imgs:
                    src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
                    if not src:
                        continue
                    photo_url = urljoin(page_url, src)

                    score = score_img(img)
                    if score > best_score:
                        # Try to find name
                        name = _extract_name_from_container(container)
                        appt = _extract_appointment_date(container.get_text())
                        best_score = score
                        best_result = {
                            "name": name,
                            "title": title,
                            "photo_url": photo_url,
                            "appointment_info": appt,
                            "source_url": page_url,
                        }
                container = container.parent

    # If score too low, reject
    if best_result and best_score > -10:
        return best_result

    # Fallback: look for common page structures
    return _fallback_ceo_search(soup, page_url)


def _fallback_ceo_search(soup: BeautifulSoup, page_url: str) -> dict | None:
    """Fallback: look for CEO in structured table/list formats."""
    # Look for tables with role and name
    for table in soup.find_all(["table", "dl", "ul"]):
        text = table.get_text()
        if not is_ceo_title(text):
            continue
        imgs = table.find_all("img")
        for img in imgs:
            src = img.get("src") or img.get("data-src") or ""
            if src and not is_icon_image(src):
                photo_url = urljoin(page_url, src)
                name = _extract_name_from_container(table)
                appt = _extract_appointment_date(text)
                # Find which title matched
                title = next((t for t in CEO_TITLES_JP if t in text), CEO_TITLES_JP[0])
                return {"name": name, "title": title, "photo_url": photo_url,
                        "appointment_info": appt, "source_url": page_url}

    # Look for section/article with CEO title
    for section in soup.find_all(["section", "article", "div"], limit=200):
        text = section.get_text()
        if len(text) > 2000:  # Skip large containers
            continue
        if not is_ceo_title(text):
            continue
        imgs = section.find_all("img")
        for img in imgs:
            src = img.get("src") or img.get("data-src") or ""
            if src and not is_icon_image(src) and score_img(img) > -50:
                photo_url = urljoin(page_url, src)
                name = _extract_name_from_container(section)
                appt = _extract_appointment_date(text)
                title = next((t for t in CEO_TITLES_JP if t in text), CEO_TITLES_JP[0])
                return {"name": name, "title": title, "photo_url": photo_url,
                        "appointment_info": appt, "source_url": page_url}
    return None


def _extract_name_from_container(container) -> str:
    """Extract Japanese person name from a container element."""
    text = container.get_text(" ", strip=True)
    # Remove title keywords to isolate the name
    for t in CEO_TITLES_JP + ["氏", "様", "さん", "（", "）", "(", ")", "【", "】", "■", "●"]:
        text = text.replace(t, " ")
    # Japanese name: family name (1-2 kanji) + space + given name (1-3 kanji) or just 2-4 kanji together
    patterns = [
        r"[\u4e00-\u9fff]{1,2}[\s\u3000]+[\u4e00-\u9fff]{1,3}",  # kanji space kanji
        r"[\u4e00-\u9fff]{2,4}",  # 2-4 consecutive kanji
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            name = m.group().strip()
            # Filter out company-like names or short single kanji
            if len(name.replace(" ", "").replace("\u3000", "")) >= 2:
                return name
    return ""


def _extract_appointment_date(text: str) -> str | None:
    """Extract appointment date from text."""
    patterns = [
        r"(20\d{2}年\d{1,2}月)",
        r"(平成\d{1,2}年\d{1,2}月)",
        r"(令和\d{1,2}年\d{1,2}月)",
        r"(\d{4}/\d{1,2}(?:/\d{1,2})?)",
        r"(就任[：:\s]*20\d{2}年\d{1,2}月)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


# ─── Management Page Finder ────────────────────────────────────────────────────

def find_management_page(base_url: str) -> list[str]:
    """
    Find potential management/officer pages for a company.
    Returns list of URLs sorted by relevance.
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    found_urls = []

    # 1. Try known path patterns
    for path in MGMT_PATH_PATTERNS:
        url = base + path
        resp = safe_get(url)
        if resp and resp.url and len(resp.content) > 500:
            try:
                content = resp.content.decode("utf-8", errors="replace")
            except Exception:
                content = ""
            if is_ceo_title(content):
                found_urls.append(resp.url)
                if len(found_urls) >= 3:
                    break

    # 2. Parse top page links
    if not found_urls:
        soup, final_url = get_soup(base_url)
        if soup:
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                link_text = a.get_text(" ", strip=True)
                if any(kw in href.lower() or kw in link_text
                       for kw in MGMT_LINK_KEYWORDS):
                    full_url = urljoin(base_url, href)
                    fp = urlparse(full_url)
                    if fp.netloc == parsed.netloc:
                        sub_soup, sub_url = get_soup(full_url)
                        if sub_soup and is_ceo_title(sub_soup.get_text()):
                            found_urls.append(sub_url)
                if len(found_urls) >= 3:
                    break

    return found_urls


# ─── Photo Download ─────────────────────────────────────────────────────────────

def download_photo(photo_url: str, save_path: Path) -> bool:
    """Download and validate a photo, save as JPEG."""
    try:
        resp = SESSION.get(photo_url, timeout=20, stream=True)
        if resp.status_code != 200:
            return False
        img_data = resp.content
        try:
            img = Image.open(BytesIO(img_data))
            # Require minimum size
            if img.width < 60 or img.height < 60:
                return False
            # Must be reasonable aspect ratio for portrait
            ratio = img.height / max(img.width, 1)
            if ratio < 0.5 or ratio > 4.0:
                return False
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path, "JPEG", quality=90)
            return True
        except Exception:
            ct = resp.headers.get("content-type", "")
            if "image" in ct and len(img_data) > 5000:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(img_data)
                return True
    except Exception as e:
        log.debug(f"Photo download error {photo_url}: {e}")
    return False


# ─── Stock Price ──────────────────────────────────────────────────────────────

def get_stock_price(ticker: str, date_str: str) -> dict | None:
    """Get stock price around a given date using yfinance."""
    try:
        m = re.search(r"(\d{4})[年/](\d{1,2})", date_str)
        if not m:
            return None
        year, month = int(m.group(1)), int(m.group(2))
        dt = datetime(year, month, 1)
        start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=40)).strftime("%Y-%m-%d")

        stock = yf.Ticker(f"{ticker}.T")
        hist = stock.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None

        prices = hist["Close"]
        idx = prices.index.get_indexer([dt], method="nearest")[0]
        close_on_date = float(prices.iloc[idx]) if idx >= 0 else float(prices.iloc[0])

        return {
            "date": dt.strftime("%Y-%m"),
            "close_on_date": round(close_on_date, 2),
            "close_min_period": round(float(prices.min()), 2),
            "close_max_period": round(float(prices.max()), 2),
            "currency": "JPY",
        }
    except Exception as e:
        log.debug(f"Stock price error {ticker}: {e}")
        return None


# ─── Previous CEO Search ───────────────────────────────────────────────────────

def search_previous_ceos(base_url: str) -> list[dict]:
    """Search for historical CEO information from press releases."""
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    prev_ceos = []

    news_paths = [
        "/news/", "/ir/news/", "/news/release/", "/press/",
        "/ir/release/", "/release/", "/ir/press/", "/newsrelease/",
        "/ir/newsrelease/", "/news/topics/",
    ]
    ceo_change_kw = ["社長交代", "代表取締役社長", "社長就任", "退任", "人事異動", "役員改選"]

    for path in news_paths[:4]:
        url = base + path
        soup, _ = get_soup(url)
        if not soup:
            continue
        for a in soup.find_all("a", href=True):
            link_text = a.get_text()
            if any(kw in link_text for kw in ceo_change_kw):
                news_url = urljoin(url, a["href"])
                news_soup, _ = get_soup(news_url)
                if news_soup:
                    entry = _parse_ceo_change_news(news_soup.get_text(), news_url)
                    if entry:
                        prev_ceos.append(entry)
        if len(prev_ceos) >= 3:
            break

    return prev_ceos[:3]


def _parse_ceo_change_news(text: str, url: str) -> dict | None:
    if not (is_ceo_title(text) and any(kw in text for kw in ["就任", "退任", "交代"])):
        return None

    # Extract name (person becoming CEO)
    patterns = [
        r"([\u4e00-\u9fff]{1,2}[\s　][\u4e00-\u9fff]{1,3}).*?(?:が|は)?(?:代表取締役社長|社長).*?就任",
        r"新社長.*?([\u4e00-\u9fff]{1,2}[\s　]?[\u4e00-\u9fff]{1,3})氏",
        r"([\u4e00-\u9fff]{1,2}[\s　]?[\u4e00-\u9fff]{1,3}).*?新.*?社長",
    ]
    name = ""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip()
            break

    dates = re.findall(r"\d{4}年\d{1,2}月(?:\d{1,2}日)?", text)
    appt_date = dates[0] if dates else None
    res_date = dates[-1] if len(dates) > 1 else None

    if not name:
        return None
    return {"name": name, "appointment_date": appt_date, "resignation_date": res_date, "source_url": url}


# ─── Main Processing ──────────────────────────────────────────────────────────

def safe_dir_name(s: str) -> str:
    return re.sub(r'[^\w\u3040-\u30ff\u4e00-\u9fff\-]', '_', str(s))[:50]


def process_company(company: dict) -> dict:
    ticker = company["ticker"]
    name = company["name"]
    base_url = company.get("url", "")

    result = {
        "ticker": ticker,
        "company_name": name,
        "url": base_url,
        "current_ceo": None,
        "previous_ceos": [],
        "error": None,
        "processed_at": datetime.now().isoformat(),
    }

    if not base_url or base_url in ("None", "nan", ""):
        result["error"] = "No URL"
        return result

    log.info(f"[{ticker}] {name}")

    try:
        # Find management pages
        mgmt_urls = find_management_page(base_url)

        ceo_info = None

        # Try each management page
        for mgmt_url in mgmt_urls:
            soup, final_url = get_soup(mgmt_url)
            ceo_info = find_ceo_in_soup(soup, final_url or mgmt_url)
            if ceo_info:
                break

        # Fallback: try base URL
        if not ceo_info:
            soup, final_url = get_soup(base_url)
            ceo_info = find_ceo_in_soup(soup, final_url or base_url)

        if ceo_info:
            # Download photo
            safe_name = safe_dir_name(name)
            photo_dir = PHOTOS_DIR / f"{ticker}_{safe_name}" / "current"
            photo_path = photo_dir / "photo.jpg"

            photo_saved = False
            if ceo_info.get("photo_url") and not photo_path.exists():
                photo_saved = download_photo(ceo_info["photo_url"], photo_path)
            elif photo_path.exists():
                photo_saved = True

            ceo_info["photo_saved"] = photo_saved

            # Stock price at appointment
            if ceo_info.get("appointment_info"):
                price = get_stock_price(ticker, ceo_info["appointment_info"])
                if price:
                    ceo_info["stock_price_at_appointment"] = price

            # Save info.json
            photo_dir.mkdir(parents=True, exist_ok=True)
            (photo_dir / "info.json").write_text(
                json.dumps(ceo_info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result["current_ceo"] = ceo_info
        else:
            result["error"] = (result.get("error") or "") + "|ceo_not_found"

        # Find previous CEOs
        prev_ceos = search_previous_ceos(base_url)
        for i, prev in enumerate(prev_ceos[:3], 1):
            if prev.get("resignation_date"):
                price = get_stock_price(ticker, prev["resignation_date"])
                if price:
                    prev["stock_price_at_resignation"] = price

            safe_name = safe_dir_name(name)
            prev_name = safe_dir_name(prev.get("name", "unknown"))
            hist_dir = PHOTOS_DIR / f"{ticker}_{safe_name}" / "history" / f"{i:02d}_{prev_name}"
            hist_dir.mkdir(parents=True, exist_ok=True)
            (hist_dir / "info.json").write_text(
                json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        result["previous_ceos"] = prev_ceos

    except Exception as e:
        result["error"] = str(e)
        log.warning(f"Error [{ticker}] {name}: {e}")

    time.sleep(random.uniform(0.3, 1.0))
    return result


# ─── Progress ─────────────────────────────────────────────────────────────────

def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_state(done: set, results: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(done), f)
    with open(CEO_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(list(results.values()), f, ensure_ascii=False, indent=2)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("CEO Photo & History Collector v2 - Starting")
    log.info(f"Project dir: {PROJECT_DIR}")
    log.info("=" * 60)

    with open(COMPANIES_FILE, encoding="utf-8") as f:
        companies = json.load(f)
    log.info(f"Total companies: {len(companies)}")

    # Load existing results
    results = {}
    if CEO_DATA_FILE.exists():
        with open(CEO_DATA_FILE, encoding="utf-8") as f:
            for r in json.load(f):
                results[r["ticker"]] = r

    done_tickers = load_progress()
    log.info(f"Already done: {len(done_tickers)}")

    todo = [c for c in companies if c["ticker"] not in done_tickers]
    log.info(f"Remaining: {len(todo)}")

    BATCH = 50
    WORKERS = 3

    for batch_i, start in enumerate(range(0, len(todo), BATCH)):
        batch = todo[start:start + BATCH]
        log.info(f"\nBatch {batch_i + 1}/{-(-len(todo)//BATCH)}: {len(batch)} companies")

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(process_company, c): c for c in batch}
            for future in as_completed(futures):
                c = futures[future]
                try:
                    r = future.result(timeout=120)
                    results[r["ticker"]] = r
                    done_tickers.add(r["ticker"])
                    if r.get("current_ceo") and r["current_ceo"].get("photo_saved"):
                        log.info(f"  ✓ Photo saved: [{r['ticker']}] {r['company_name']}")
                except Exception as e:
                    log.error(f"Future error {c['ticker']}: {e}")
                    done_tickers.add(c["ticker"])

        save_state(done_tickers, results)
        with_photos = sum(1 for r in results.values()
                          if r.get("current_ceo") and r["current_ceo"].get("photo_saved"))
        log.info(f"Progress: {len(done_tickers)}/{len(companies)} | Photos: {with_photos}")

    log.info("\n" + "=" * 60)
    log.info("Collection complete!")
    with_ceo = sum(1 for r in results.values() if r.get("current_ceo"))
    with_photos = sum(1 for r in results.values()
                      if r.get("current_ceo") and r["current_ceo"].get("photo_saved"))
    log.info(f"CEO info found: {with_ceo}/{len(results)}")
    log.info(f"Photos saved: {with_photos}/{len(results)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
