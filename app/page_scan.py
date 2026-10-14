"""
Sarwsh selidwn

To reverse search vriskei eikones pou moiazoun optika me tin eikona eisodou.
Ena deepfake me to idio proswpo omws einai alli eikona (allo soma, allo
fonto), opote den epistrefetai. Edw anoigoume tis selides opou vrethike
epivevaiomeno match kai mazevoume oles tis eikones tous, wste na elegxthoun
kai aftes apo to filtro prosopou kai to DeepSafe.
"""

import asyncio
import logging
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from . import config
from .reverse_search import SearchHit

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/126.0 Safari/537.36"}

# Eikones pou sxedon panta einai logotypa, eikonidia klp.
SKIP_EXTENSIONS = (".svg", ".ico", ".gif")


class _ImageCollector(HTMLParser):
    """Mazevei ta URL eikonwn apo ena HTML (img, srcset, og:image, lazy load)."""

    def __init__(self):
        super().__init__()
        self.urls: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img" or tag == "source":
            # Sto srcset pairnoume thn teleutaia (megalyterh) ekdosh
            srcset = attrs.get("srcset") or attrs.get("data-srcset")
            if srcset:
                last = srcset.split(",")[-1].strip().split(" ")[0]
                if last:
                    self.urls.append(last)
            for key in ("data-src", "data-original", "data-lazy-src", "src"):
                if attrs.get(key):
                    self.urls.append(attrs[key])
                    break
        elif tag == "meta":
            prop = attrs.get("property") or attrs.get("name") or ""
            if prop in ("og:image", "twitter:image") and attrs.get("content"):
                self.urls.append(attrs["content"])
        elif tag == "a":
            # Polles gallery selides deixnoun mikrografia pou syndeei sthn
            # megalh eikona.
            href = attrs.get("href", "")
            if href.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp")):
                self.urls.append(href)
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()


def _keep(url: str) -> bool:
    """Kratame mono http(s) eikones pou den einai eikonidia."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    return not parsed.path.lower().endswith(SKIP_EXTENSIONS)


async def _scan_one(client: httpx.AsyncClient, page_url: str) -> list[SearchHit]:
    """Katevazei mia selida kai epistrefei tis eikones tis ws SearchHit."""
    try:
        response = await client.get(page_url, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:
        logger.info("Apotyxia sarwshs selidas %s: %s", page_url[:60], exc)
        return []

    if "html" not in response.headers.get("content-type", ""):
        return []

    collector = _ImageCollector()
    try:
        collector.feed(response.text)
    except Exception as exc:
        logger.info("Sfalma sto HTML ths %s: %s", page_url[:60], exc)

    base = str(response.url)
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for raw in collector.urls:
        url = urljoin(base, raw.strip())
        if url in seen or not _keep(url):
            continue
        seen.add(url)
        hits.append(SearchHit(
            image_url=url,
            source_url=page_url,
            title=collector.title,
            engine="page_scan",
            match_type="same_page",
        ))

    logger.info("Selida %s: %d eikones", page_url[:60], len(hits))
    return hits[:config.PAGE_SCAN_MAX_IMAGES]


async def scan_pages(confirmed: list[SearchHit],
                     known_urls: set[str]) -> list[SearchHit]:
    """
    Sarwnei tis selides twn epivevaiomenwn matches.

    known_urls: eikones pou exoun idi elegxthei apo to reverse search,
    gia na mhn tis ksanaelegxoume.
    """
    pages: list[str] = []
    for hit in confirmed:
        if hit.source_url and hit.source_url not in pages:
            pages.append(hit.source_url)
    pages = pages[:config.PAGE_SCAN_MAX_PAGES]

    if not pages:
        return []

    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS) as client:
        results = await asyncio.gather(*(_scan_one(client, p) for p in pages))

    new_hits: list[SearchHit] = []
    seen = set(known_urls)
    for page_hits in results:
        for hit in page_hits:
            if hit.image_url in seen:
                continue
            seen.add(hit.image_url)
            new_hits.append(hit)

    logger.info("Sarwsh selidwn: %d selides, %d nees eikones",
               len(pages), len(new_hits))
    return new_hits
