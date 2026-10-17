#!/usr/bin/env python3
"""Verify academic papers (arXiv XML API) and open-source repos (GitHub REST API).

Ensures that no audit report ever cites a hallucinated paper, wrong arXiv ID,
unverified star count, or missing repository license (following the strict
verification standard of docs/00-research-dossier.md and docs/reviews.md).

Usage:
    python3 verify_citations.py --arxiv 2407.17866 2510.02209 --github AI4Finance-Foundation/FinRobot NoFxAiOS/nofx
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def verify_arxiv(arxiv_ids: list[str]) -> list[dict[str, Any]]:
    """Query the official arXiv API and parse the Atom XML response."""
    if not arxiv_ids:
        return []
    clean_ids = [aid.replace("arXiv:", "").strip() for aid in arxiv_ids]
    query = urllib.parse.urlencode({"id_list": ",".join(clean_ids)})
    url = f"https://export.arxiv.org/api/query?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "alphalab-quant-auditor/1.0"})
    results: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)
        for entry in root.findall("atom:entry", ARXIV_NS):
            id_elem = entry.find("atom:id", ARXIV_NS)
            title_elem = entry.find("atom:title", ARXIV_NS)
            pub_elem = entry.find("atom:published", ARXIV_NS)
            authors = [
                a.find("atom:name", ARXIV_NS).text
                for a in entry.findall("atom:author", ARXIV_NS)
                if a.find("atom:name", ARXIV_NS) is not None
            ]
            raw_id = id_elem.text if id_elem is not None and id_elem.text else ""
            title = " ".join((title_elem.text or "").split()) if title_elem is not None else None
            results.append(
                {
                    "id": raw_id.rsplit("/", 1)[-1],
                    "verified": bool(title and title != "Error"),
                    "title": title,
                    "authors": authors[:5],
                    "published": pub_elem.text[:10] if pub_elem is not None and pub_elem.text else None,
                }
            )
    except Exception as exc:
        results.append({"error": f"arXiv API error: {type(exc).__name__}: {exc}", "verified": False})
    return results


def verify_github(repos: list[str]) -> list[dict[str, Any]]:
    """Query the GitHub REST API for repository metadata, license, and activity."""
    results: list[dict[str, Any]] = []
    for repo in repos:
        clean = repo.strip().strip("/")
        url = f"https://api.github.com/repos/{clean}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "alphalab-quant-auditor/1.0",
                "Accept": "application/vnd.github+json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            lic = data.get("license")
            results.append(
                {
                    "repo": clean,
                    "verified": True,
                    "stars": data.get("stargazers_count"),
                    "license": lic.get("spdx_id") if isinstance(lic, dict) else None,
                    "pushed_at": (data.get("pushed_at") or "")[:10],
                    "archived": bool(data.get("archived")),
                    "description": data.get("description"),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "repo": clean,
                    "verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv", nargs="*", default=[], help="arXiv IDs to verify")
    parser.add_argument("--github", nargs="*", default=[], help="GitHub owner/repo names to verify")
    args = parser.parse_args()

    payload = {
        "arxiv": verify_arxiv(args.arxiv),
        "github": verify_github(args.github),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
