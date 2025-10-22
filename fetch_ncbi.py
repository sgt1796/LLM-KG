#!/usr/bin/env python3
"""
fetch_ncbi.py — Download random N papers from a CSV/TSV list.

Robust PMC handling:
- Some PMC /pdf endpoints return HTML instead of redirecting.
- We GET the page; if not a PDF, we parse HTML for a *.pdf link and follow it.

Also supports DOI via content negotiation and HTML sniffing for <meta name="citation_pdf_url">.

Usage examples:
  python fetch_ncbi.py --csv csv-adhd-set.csv --out papers --n 5 --seed 7 --only-pmcid --resume
  python fetch_ncbi.py --out papers --validate-only
"""

from __future__ import annotations
import argparse
import pathlib
import random
import re
import sys
import time
from typing import Optional, Tuple
from urllib.parse import urljoin
import os, tempfile
import pandas as pd
import requests
from requests.exceptions import RequestException,ChunkedEncodingError
from urllib3.exceptions import ProtocolError


# -------------------- utilities --------------------
def stream_save(resp: requests.Response, out_path: pathlib.Path, first_bytes: Optional[bytes] = None) -> None:
    """Write response to a temp file, then atomically move into place."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".part_", suffix=".pdf", dir=str(out_path.parent))
    os.close(fd)
    ok = False
    try:
        with open(tmp_path, "wb") as f:
            if first_bytes:
                f.write(first_bytes)
            for chunk in resp.iter_content(chunk_size=1024 * 64, decode_unicode=False):
                if chunk:
                    f.write(chunk)
        ok = True
    finally:
        if ok:
            os.replace(tmp_path, out_path)
        else:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180]


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def get_cell(row: pd.Series, *keys) -> Optional[str]:
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]).strip():
            return str(row[k]).strip()
    return None


def is_probably_pdf_bytes(b: bytes) -> bool:
    return len(b) >= 5 and b[:5] == b"%PDF-"


def is_probably_pdf_path(p: pathlib.Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(5)
        return is_probably_pdf_bytes(head)
    except Exception:
        return False


def peek_first_bytes(resp: requests.Response, n: int = 5) -> bytes:
    # Use resp.raw to avoid exhausting iter_content
    return resp.raw.read(n, decode_content=True)


def stream_save(resp: requests.Response, out_path: pathlib.Path, first_bytes: Optional[bytes] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        if first_bytes:
            f.write(first_bytes)
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)


def pmcid_to_pdf_candidates(pmcid: str) -> list[str]:
    pmcid = pmcid.strip().upper()
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + re.sub(r"[^0-9]", "", pmcid)
    base = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}"
    return [
        f"{base}/pdf",                 # often an HTML page with a link to the real PDF
        f"{base}/pdf/{pmcid}.pdf",     # explicit filename (sometimes present)
    ]


def doi_to_request(doi: str) -> Tuple[str, dict]:
    url = f"https://doi.org/{doi.strip()}"
    headers = {
        "Accept": "application/pdf, text/html;q=0.9, */*;q=0.8",
        "User-Agent": "paper-downloader/1.4 (+https://example.org)"
    }
    return url, headers


def find_pdf_link_in_html(html: str, base_url: str) -> Optional[str]:
    """
    Heuristically find a .pdf link in HTML. Works for PMC /pdf landing pages and many publisher pages.
    Strategies:
      - <meta name="citation_pdf_url" content="...">
      - href="...pdf"
      - Link text containing 'PDF'
    """
    # meta tag
    m = re.search(r'<meta[^>]+name=[\'"]citation_pdf_url[\'"][^>]+content=[\'"]([^\'"]+)[\'"]', html, re.IGNORECASE)
    if m:
        return urljoin(base_url, m.group(1).strip())

    # any href ending with .pdf (relative or absolute)
    for m in re.finditer(r'href=[\'"]([^\'"]+\.pdf)(\?[^\'"]*)?[\'"]', html, re.IGNORECASE):
        href = m.group(1)
        return urljoin(base_url, href)

    # anchors with text PDF
    for m in re.finditer(r'<a[^>]+href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
        href = m.group(1)
        text = re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", m.group(2))).strip()
        if "pdf" in text.lower():
            return urljoin(base_url, href)

    return None


def get_with_fallback(url: str, headers: Optional[dict], timeout: int = 30) -> Optional[requests.Response]:
    """
    Try a normal streamed GET. If the server uses broken chunked encoding,
    the caller should catch and retry non-streamed.
    """
    try:
        r = requests.get(url, headers=headers or {}, stream=True,
                         timeout=(10, timeout), allow_redirects=True)
        if r.status_code == 200:
            return r
    except RequestException:
        return None
    return None


def download_pdf_bytes(url: str, headers: Optional[dict], timeout: int = 30) -> Optional[bytes]:
    """
    Non-streamed fallback: disable compression & keep-alive to avoid chunking issues.
    """
    hdrs = dict(headers or {})
    # Force identity (no gzip/deflate/br) and close connection
    hdrs.setdefault("Accept-Encoding", "identity")
    hdrs.setdefault("Connection", "close")
    try:
        r = requests.get(url, headers=hdrs, stream=False,
                         timeout=(10, timeout), allow_redirects=True)
        if r.status_code != 200:
            return None
        data = r.content
        if not (len(data) >= 5 and data[:5] == b"%PDF-"):
            return None
        return data
    except RequestException:
        return None


def download_pdf_resolving_html(url: str, headers: Optional[dict], out_path: pathlib.Path, timeout: int = 30) -> bool:
    """
    GET url. If PDF -> save. If HTML -> parse for *.pdf and retry.
    Handles broken chunked transfers by falling back to non-streamed fetch.
    """
    # 1) First attempt: streamed GET
    r = get_with_fallback(url, headers, timeout)
    if r:
        ct = (r.headers.get("Content-Type") or "").lower()
        try:
            first = peek_first_bytes(r, 5)
        except Exception:
            first = b""

        if "pdf" in ct or is_probably_pdf_bytes(first) or r.url.lower().endswith(".pdf"):
            if not is_probably_pdf_bytes(first):
                # Fallback to non-streamed download (some servers mislabel ct)
                data = download_pdf_bytes(r.url, headers, timeout)
                if data:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    return True
                return False
            # Stream-save with protection against chunk errors
            try:
                stream_save(r, out_path, first_bytes=first)
                return True
            except (ChunkedEncodingError, ProtocolError):
                # Fallback: non-streamed, no compression
                data = download_pdf_bytes(r.url, headers, timeout)
                if data:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    return True
                return False

        # 2) Not a PDF → parse HTML for a *.pdf link (need text fetch)
        try:
            r2 = requests.get(r.url, headers=headers or {}, stream=False,
                              timeout=(10, timeout), allow_redirects=True)
            if r2.status_code != 200:
                return False
            html = r2.text
        except RequestException:
            return False

        pdf_url = find_pdf_link_in_html(html, r2.url)
        if not pdf_url:
            return False

        # 3) Try to download the discovered PDF (streamed, then fallback)
        r3 = get_with_fallback(pdf_url, headers, timeout)
        if r3:
            try:
                first3 = peek_first_bytes(r3, 5)
            except Exception:
                first3 = b""
            if is_probably_pdf_bytes(first3):
                try:
                    stream_save(r3, out_path, first_bytes=first3)
                    return True
                except (ChunkedEncodingError, ProtocolError):
                    data = download_pdf_bytes(pdf_url, headers, timeout)
                    if data:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_path, "wb") as f:
                            f.write(data)
                        return True
                    return False

        # Final fallback: non-streamed download of the resolved PDF URL
        data = download_pdf_bytes(pdf_url, headers, timeout)
        if data:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(data)
            return True
        return False

    # If first GET failed, try one-shot non-streamed
    data = download_pdf_bytes(url, headers, timeout)
    if data:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        return True
    return False


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV/TSV file containing columns like PMID, Title, PMCID, DOI")
    ap.add_argument("--out", default="downloads", help="Output directory")
    ap.add_argument("--n", type=int, default=-1, help="How many random items to download, default all.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--sleep", type=float, default=0.8, help="Seconds to sleep between downloads")
    ap.add_argument("--dry-run", action="store_true", help="Select and show targets, but do not download")
    ap.add_argument("--validate-only", action="store_true", help="Scan --out for real PDFs and report problems")
    ap.add_argument("--only-pmcid", action="store_true", help="Only sample rows that have a PMCID")
    ap.add_argument("--resume", action="store_true", help="Skip items if a valid PDF already exists")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)

    # Validation mode
    if args.validate_only:
        bad = []
        small = []
        for p in outdir.glob("*.pdf"):
            if not is_probably_pdf_path(p):
                bad.append(p)
            elif p.stat().st_size < 1024:
                small.append(p)
        if bad:
            print("Corrupted or non-PDF files:")
            for p in bad:
                print(" -", p)
        if small:
            print("Very small PDFs (possibly truncated):")
            for p in small:
                print(" -", p)
        if not bad and not small:
            print("All PDFs look valid.")
        return

    # Normal run requires csv + n
    if not args.csv or args.n is None:
        ap.error("the following arguments are required: --csv, --n")

    # Read CSV/TSV robustly
    try:
        try:
            df = pd.read_csv(args.csv, engine="python")
        except Exception:
            df = pd.read_csv(args.csv, sep="\t", engine="python")
    except Exception as e:
        print(f"Failed to read {args.csv}: {e}", file=sys.stderr)
        sys.exit(2)

    df = norm_cols(df)

    pmcid_cols = ["pmcid"]
    doi_cols = ["doi"]
    pmid_cols = ["pmid"]
    title_cols = ["title"]

    df["__pmcid"] = df.apply(lambda r: get_cell(r, *pmcid_cols), axis=1)
    df["__doi"] = df.apply(lambda r: get_cell(r, *doi_cols), axis=1)

    if args.only_pmcid:
        usable = df[df["__pmcid"].notna()].copy()
    else:
        usable = df[(df["__pmcid"].notna()) | (df["__doi"].notna())].copy()

    if usable.empty:
        print("No rows matching your criteria (check PMCID/DOI columns).", file=sys.stderr)
        sys.exit(1)

    if args.n >= 0:
        if args.seed is not None:
            random.seed(args.seed)

        n = min(args.n, len(usable))
        sampled = usable.sample(n=n, random_state=args.seed) if args.seed is not None else usable.sample(n=n)
    else:
        # Use all
        n = len(usable)
        sampled = usable

    successes = 0
    failures = 0

    for _, row in sampled.iterrows():
        pmid = get_cell(row, *pmid_cols) or "unknown"
        title = get_cell(row, *title_cols) or "untitled"
        base_name = sanitize_filename(f"{pmid} - {title}") or f"{pmid}"
        out_path = outdir / f"{base_name}.pdf"

        pmcid = row["__pmcid"]
        doi = row["__doi"]

        print(f"\n> {base_name}")

        if args.resume and out_path.exists() and is_probably_pdf_path(out_path) and out_path.stat().st_size > 1024:
            print(f"  Skipping (already exists and looks valid): {out_path}")
            successes += 1
            continue

        if args.dry_run:
            print(f"  (dry-run) source: {'PMCID' if pmcid else 'DOI'}")
            continue

        ok = False

        # 1) PMC path: try candidates, with HTML-sniff fallback
        if pmcid and not ok:
            for cand in pmcid_to_pdf_candidates(pmcid):
                print(f"  Trying PMCID: {pmcid} -> {cand}")
                ok = download_pdf_resolving_html(
                    cand, {"User-Agent": "paper-downloader/1.4"}, out_path
                )
                if ok:
                    break

        # 2) DOI path: try content-negotiation; if HTML, sniff for PDF link
        if doi and not ok and not args.only_pmcid:
            url, headers = doi_to_request(doi)
            print(f"  Trying DOI: {doi} -> {url}")
            ok = download_pdf_resolving_html(url, headers, out_path)
            if not ok:
                print("  DOI did not yield a usable PDF (publisher may block direct PDF).")

        if ok and not is_probably_pdf_path(out_path):
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            ok = False

        if ok:
            print(f"  Saved: {out_path}")
            successes += 1
        else:
            print("  ❌ Could not fetch a PDF via PMCID or DOI.")
            failures += 1

        time.sleep(args.sleep)

    print(f"\nDone. Success: {successes}, Failed: {failures}, Requested: {args.n}, Attempted: {n}")


if __name__ == "__main__":
    main()
