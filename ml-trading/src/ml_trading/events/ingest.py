"""Near-real-time event ingestion: SEC EDGAR 8-K filings and news APIs.

EDGAR is free and authoritative for corporate events (8-K = material events);
Finnhub supplies headline news. Items are normalized to `NewsItem`, deduplicated
and entity-linked to tickers before classification.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from xml.etree import ElementTree

import httpx

from ml_trading.config import Secrets

EDGAR_BASE = "https://www.sec.gov"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass
class NewsItem:
    id: str
    ts: datetime
    source: str  # "edgar" | "finnhub" | ...
    headline: str
    body: str = ""
    tickers: list[str] = field(default_factory=list)
    url: str = ""
    form_type: str = ""  # for EDGAR filings


def _item_id(source: str, key: str) -> str:
    return hashlib.sha1(f"{source}:{key}".encode()).hexdigest()[:16]


def dedup_items(items: list[NewsItem], window_chars: int = 80) -> list[NewsItem]:
    """Drop exact-id duplicates and near-duplicate headlines (same normalized prefix)."""
    seen_ids: set[str] = set()
    seen_heads: set[str] = set()
    out: list[NewsItem] = []
    for it in sorted(items, key=lambda x: x.ts):
        head_key = re.sub(r"\W+", " ", it.headline.lower()).strip()[:window_chars]
        if it.id in seen_ids or head_key in seen_heads:
            continue
        seen_ids.add(it.id)
        seen_heads.add(head_key)
        out.append(it)
    return out


class EdgarClient:
    """Fetch recent 8-K filings from EDGAR full-text/atom feeds.

    SEC requires a descriptive User-Agent with contact info.
    """

    def __init__(self, user_agent: str = "ml-trading research agent@example.com") -> None:
        self._headers = {"User-Agent": user_agent}

    def recent_8k(self, count: int = 40) -> list[NewsItem]:
        url = (
            f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&type=8-K"
            f"&dateb=&owner=include&count={count}&output=atom"
        )
        resp = httpx.get(url, headers=self._headers, timeout=30.0)
        resp.raise_for_status()
        return self.parse_atom(resp.text)

    @staticmethod
    def parse_atom(xml_text: str) -> list[NewsItem]:
        root = ElementTree.fromstring(xml_text)
        items: list[NewsItem] = []
        for entry in root.iter(f"{_ATOM_NS}entry"):
            title = entry.findtext(f"{_ATOM_NS}title", default="")
            updated = entry.findtext(f"{_ATOM_NS}updated", default="")
            link_el = entry.find(f"{_ATOM_NS}link")
            href = link_el.get("href", "") if link_el is not None else ""
            try:
                ts = datetime.fromisoformat(updated)
            except ValueError:
                ts = datetime.now(timezone.utc)
            items.append(
                NewsItem(
                    id=_item_id("edgar", href or title),
                    ts=ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc),
                    source="edgar",
                    headline=title,
                    url=href,
                    form_type="8-K",
                )
            )
        return items


class FinnhubNewsClient:
    def __init__(self, secrets: Secrets | None = None) -> None:
        s = secrets or Secrets()
        if not s.finnhub_api_key:
            raise RuntimeError("FINNHUB_API_KEY not set in the environment")
        self._key = s.finnhub_api_key

    def company_news(self, ticker: str, start: str, end: str) -> list[NewsItem]:
        resp = httpx.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": start, "to": end, "token": self._key},
            timeout=30.0,
        )
        resp.raise_for_status()
        items = []
        for row in resp.json():
            items.append(
                NewsItem(
                    id=_item_id("finnhub", str(row.get("id", row.get("url", "")))),
                    ts=datetime.fromtimestamp(row["datetime"], tz=timezone.utc),
                    source="finnhub",
                    headline=row.get("headline", ""),
                    body=row.get("summary", ""),
                    tickers=[ticker],
                    url=row.get("url", ""),
                )
            )
        return items


def link_tickers(item: NewsItem, name_to_ticker: dict[str, str]) -> NewsItem:
    """Naive entity linking: match known company names in the headline."""
    text = item.headline.lower()
    for name, ticker in name_to_ticker.items():
        if name.lower() in text and ticker not in item.tickers:
            item.tickers.append(ticker)
    return item
