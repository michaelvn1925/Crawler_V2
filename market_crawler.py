#!/usr/bin/env python3
"""
Market Research Web Crawler
---------------------------
A polite, domain-limited crawler for collecting public website signals.

What it collects:
- URL, page title, meta description, canonical URL
- H1/H2 headings
- word count and visible page text
- internal links discovered (used for crawling, not exported as separate rows)

It does NOT bypass logins, paywalls, CAPTCHAs, or access controls.
It checks robots.txt, uses a delay between requests, and stays on the
seed domains you provide.

Install:
    python -m pip install requests beautifulsoup4

Example:
    python market_crawler.py https://example.com https://competitor.com \
        --max-pages 100 --delay 2 --output market_research.csv

Tip:
    Start with a small page limit and review each site's terms and robots.txt.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Iterable
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


USER_AGENT = "MarketResearchCrawler/1.0 (+respectful public-web research)"


@dataclass
class PageRecord:
    seed_domain: str
    url: str
    status_code: int
    title: str
    meta_description: str
    canonical_url: str
    h1: str
    h2: str
    word_count: int
    page_text: str
    crawled_at: str


def normalize_url(url: str) -> str:
    """Remove fragments and normalize a URL for deduplication."""
    url, _fragment = urldefrag(url.strip())
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    # Drop common tracking parameters while preserving meaningful query strings.
    tracking = {"fbclid", "gclid", "mc_cid", "mc_eid"}
    query_parts = [
        part for part in parsed.query.split("&")
        if part and part.split("=", 1)[0].lower() not in tracking
    ]
    clean = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        query="&".join(query_parts),
        fragment="",
    )
    normalized = urlunparse(clean)
    if normalized.endswith("/") and parsed.path not in {"", "/"}:
        normalized = normalized.rstrip("/")
    return normalized


def get_domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def is_html_response(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    return "text/html" in content_type or "application/xhtml+xml" in content_type


class PoliteCrawler:
    def __init__(
        self,
        seeds: Iterable[str],
        max_pages: int = 100,
        delay: float = 2.0,
        timeout: float = 15.0,
        max_text_chars: int = 12000,
    ):
        self.seeds = [normalize_url(s) for s in seeds]
        self.seeds = [s for s in self.seeds if s]
        if not self.seeds:
            raise ValueError("Provide at least one valid http(s) seed URL.")

        self.max_pages = max_pages
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.max_text_chars = max_text_chars
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.allowed_domains = {get_domain(s) for s in self.seeds}
        self.robots: dict[str, RobotFileParser] = {}
        self.last_request_at: dict[str, float] = {}
        self.visited: set[str] = set()

    def robots_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self.robots:
            robots_url = urljoin(origin, "/robots.txt")
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                response = self.session.get(robots_url, timeout=self.timeout)
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                elif response.status_code in {401, 403}:
                    # Treat an explicitly restricted robots endpoint conservatively.
                    parser.parse(["User-agent: *", "Disallow: /"])
                else:
                    # If no robots file is published, RobotFileParser's empty
                    # rules allow crawling; site terms still apply.
                    parser.parse([])
            except requests.RequestException:
                # If robots.txt cannot be checked, fail closed.
                parser.parse(["User-agent: *", "Disallow: /"])
            self.robots[origin] = parser
        return self.robots[origin]

    def allowed_by_robots(self, url: str) -> bool:
        return self.robots_for(url).can_fetch(USER_AGENT, url)

    def wait_for_domain(self, url: str) -> None:
        domain = get_domain(url)
        previous = self.last_request_at.get(domain, 0.0)
        wait = self.delay - (time.monotonic() - previous)
        if wait > 0:
            time.sleep(wait)
        self.last_request_at[domain] = time.monotonic()

    def extract_record(
        self, response: requests.Response, seed_domain: str
    ) -> tuple[PageRecord, list[str]]:
        soup = BeautifulSoup(response.text, "html.parser")

        for element in soup(["script", "style", "noscript", "svg", "iframe"]):
            element.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        description_tag = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
        description = (
            description_tag.get("content", "").strip()
            if description_tag else ""
        )

        canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
        canonical = urljoin(response.url, canonical_tag.get("href", "")) if canonical_tag else ""

        h1 = " | ".join(
            tag.get_text(" ", strip=True) for tag in soup.find_all("h1")
        )
        h2 = " | ".join(
            tag.get_text(" ", strip=True) for tag in soup.find_all("h2")
        )

        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)

        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            absolute = normalize_url(urljoin(response.url, tag["href"]))
            if absolute and get_domain(absolute) in self.allowed_domains:
                links.append(absolute)

        from datetime import datetime, timezone
        record = PageRecord(
            seed_domain=seed_domain,
            url=response.url,
            status_code=response.status_code,
            title=title,
            meta_description=description,
            canonical_url=canonical,
            h1=h1,
            h2=h2,
            word_count=len(words),
            page_text=text[: self.max_text_chars],
            crawled_at=datetime.now(timezone.utc).isoformat(),
        )
        return record, links

    def crawl(self) -> list[PageRecord]:
        queue = deque((seed, get_domain(seed)) for seed in self.seeds)
        records: list[PageRecord] = []

        while queue and len(records) < self.max_pages:
            url, seed_domain = queue.popleft()
            url = normalize_url(url)
            if not url or url in self.visited:
                continue
            self.visited.add(url)

            if get_domain(url) not in self.allowed_domains:
                continue
            if not self.allowed_by_robots(url):
                print(f"[ROBOTS BLOCKED] {url}")
                continue

            self.wait_for_domain(url)
            try:
                response = self.session.get(
                    url, timeout=self.timeout, allow_redirects=True
                )
                if not is_html_response(response):
                    print(f"[SKIP non-HTML] {url}")
                    continue
                if response.status_code >= 400:
                    print(f"[HTTP {response.status_code}] {url}")
                    continue

                record, links = self.extract_record(response, seed_domain)
                records.append(record)
                print(f"[{len(records)}/{self.max_pages}] {record.url}")

                for link in links:
                    if link not in self.visited:
                        queue.append((link, seed_domain))

            except requests.RequestException as exc:
                print(f"[ERROR] {url}: {exc}")

        return records


def write_csv(records: list[PageRecord], output_path: str) -> None:
    if not records:
        print("No pages collected; CSV was not created.")
        return
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    print(f"\nSaved {len(records)} pages to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polite, robots.txt-aware crawler for public market research."
    )
    parser.add_argument("urls", nargs="+", help="One or more seed URLs")
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Minimum seconds between requests to the same domain")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--output", default="market_research.csv")
    args = parser.parse_args()

    if args.max_pages < 1:
        parser.error("--max-pages must be at least 1")

    crawler = PoliteCrawler(
        seeds=args.urls,
        max_pages=args.max_pages,
        delay=args.delay,
        timeout=args.timeout,
        max_text_chars=args.max_text_chars,
    )
    records = crawler.crawl()
    write_csv(records, args.output)


if __name__ == "__main__":
    main()
