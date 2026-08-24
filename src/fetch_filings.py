"""
Download recent 10-K filings from SEC EDGAR.

EDGAR is built to be fetched: documented endpoints, stable URLs, and a
published fair-access policy that asks for a declared User-Agent and modest
request rates. We honour both.
"""
from __future__ import annotations

import gzip
import json
import ssl
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

UA = ("TheGenAcademy-Week2-FinRAG/1.0 (student coursework; "
      "archana.jalamadugu12@gmail.com)")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}

# The corpus, defined by TICKER rather than by CIK.
#
# EDGAR identifies companies by CIK -- a permanent number the SEC assigns, which
# never changes even when a company renames or re-lists. But nobody knows CIKs
# by heart, and a wrong one fails in a confusing way (you get somebody else's
# filings, not an error). So the tickers below are resolved to CIKs at runtime
# against the SEC's own published mapping. One extra request, no numbers for us
# to get wrong, and it is the same lookup an "add any company" feature needs.
#
# Eight semiconductor companies: one sector, so comparison questions are
# meaningful rather than apples-to-oranges.
TICKERS = ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "AMAT"]

# Display names, so citations read "NVIDIA" rather than "NVIDIA CORP".
NICE_NAMES = {
    "NVDA": "NVIDIA",   "AMD":  "AMD",       "INTC": "Intel",
    "AVGO": "Broadcom", "QCOM": "Qualcomm",  "MU":   "Micron",
    "TXN":  "Texas Instruments",             "AMAT": "Applied Materials",
}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def ticker_to_cik(tickers=None) -> dict:
    """
    Resolve stock tickers to SEC CIK numbers.

    The SEC publishes one JSON file mapping every listed ticker to its CIK.
    It arrives keyed by meaningless row numbers:

        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}

    so we flip it into {ticker: cik}. CIKs come back as plain integers and
    EDGAR wants them zero-padded to ten digits, which `latest_10k` handles.

    Raises on an unknown ticker rather than skipping it silently -- a typo
    should stop you, not quietly shrink your corpus.
    """
    raw = json.loads(get(TICKER_MAP_URL))
    lookup = {row["ticker"].upper(): str(row["cik_str"]) for row in raw.values()}
    if tickers is None:
        return lookup
    out = {}
    for t in tickers:
        t = t.upper()
        if t not in lookup:
            raise ValueError(
                f"ticker {t!r} not found in the SEC's mapping. Check the symbol -- "
                f"it must be a US-listed company that files with the SEC.")
        out[t] = lookup[t]
    return out


def resolve_companies(tickers=None, names=None) -> dict:
    """Build the {display_name: cik} mapping the fetcher works from."""
    tickers = tickers or TICKERS
    names = names or NICE_NAMES
    ciks = ticker_to_cik(tickers)
    return {names.get(t, t): cik for t, cik in ciks.items()}


def _ssl_context() -> ssl.SSLContext:
    """
    An SSL context that can actually verify certificates on macOS.

    Python installed from python.org does NOT use the macOS system keychain to
    verify HTTPS certificates -- it looks for a certificate bundle it expects
    someone to have installed for it. On a fresh Mac nobody has, so every
    `urllib` request to an https:// address fails with

        CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate

    which reads like the remote server is broken when the truth is that our
    side has no list of trusted authorities to check it against.

    The usual advice is to run `Install Certificates.command` from the Python
    folder in Applications. That works, but it fixes one machine -- anybody
    cloning this repo onto a fresh Mac hits the same wall.

    So instead we point at the bundle from `certifi`, a package whose entire
    job is to ship Mozilla's list of trusted certificate authorities. It is
    already installed (requests depends on it), it travels with the project,
    and it works identically on macOS, Linux and Windows.

    Note what this is NOT: disabling verification. Turning certificate checks
    off is the other answer you will find online and it means anyone between
    you and the SEC can impersonate them. We are giving verification the list
    it needs, not switching it off.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL = _ssl_context()


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    return raw


def latest_10k(cik: str) -> List[Tuple[str, str, str]]:
    """Return [(filing_date, accession_no_dashes, primary_doc)] newest first."""
    data = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"))
    rec = data["filings"]["recent"]
    out = []
    for form, acc, doc, date in zip(rec["form"], rec["accessionNumber"],
                                    rec["primaryDocument"], rec["filingDate"]):
        if form == "10-K":
            out.append((date, acc.replace("-", ""), doc))
    return sorted(out, reverse=True)


def fetch_corpus(dest: Path, companies: Optional[dict] = None,
                 pause: float = 0.5) -> List[dict]:
    """
    Download the latest 10-K for each company, caching to `dest`.

    Pass `companies` as {display_name: cik} to override the default corpus --
    that is the hook an "add any company" feature would use. Left empty, the
    tickers in `TICKERS` are resolved against the SEC's mapping.
    """
    if companies is None:
        print(f"resolving {len(TICKERS)} tickers to CIKs ...")
        companies = resolve_companies()
        for n, c in companies.items():
            print(f"  {n:20s} CIK {c}")
        print()
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []

    for name, cik in companies.items():
        try:
            hits = latest_10k(cik)
        except Exception as e:
            print(f"  FAIL   {name:20s} {type(e).__name__}: {str(e)[:60]}")
            continue
        if not hits:
            print(f"  FAIL   {name:20s} no 10-K found")
            continue

        date, acc, doc = hits[0]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
        safe = name.replace(" ", "_")
        path = dest / f"{safe}_10K_{date}.html"
        if path.exists():
            print(f"  cached {name:20s} {date}  {path.stat().st_size/1e6:.2f} MB")
        else:
            path.write_bytes(get(url))
            print(f"  got    {name:20s} {date}  {path.stat().st_size/1e6:.2f} MB")
            time.sleep(pause)

        manifest.append({"company": name, "cik": cik, "filing_date": date,
                         "form": "10-K", "url": url, "path": str(path),
                         "scheme": "item", "bytes": path.stat().st_size})

    (dest.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(companies)} filings ready")
    return manifest
