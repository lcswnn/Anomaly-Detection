"""
SDSS Bulk FITS Downloader v2 — Verified Fields
================================================
Step 1: Queries SDSS SkyServer to get REAL run/camcol/field combos
Step 2: Downloads only verified FITS frames (no more 404s)

Usage:
    pip install aiohttp requests
    python download_sdss_fits.py
"""

import asyncio
import aiohttp
import requests
import csv
import io
import os
import time
import random
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
DOWNLOAD_DIR = "../data/raw"
MAX_CONCURRENT = 10
TIMEOUT_SECONDS = 180
CHUNK_SIZE = 1024 * 1024

NUM_FILES = 250             # total FITS files to download
BANDS = ["r"]                # ["u", "g", "r", "i", "z"] for all bands
RERUN = 301

# SkyServer SQL API endpoint (DR17)
SKYSERVER_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
# ───────────────────────────────────────────────────────────────


def query_sdss_fields(n_fields):
    """
    Query SDSS SkyServer for verified (run, camcol, field) combinations
    from the Field table. We ask for DISTINCT combos with rerun=301,
    sampling broadly across the sky.
    """
    # Request more than we need since we'll sample randomly
    query_limit = min(n_fields * 3, 50000)

    sql = f"""
    SELECT DISTINCT TOP {query_limit}
        run, camcol, field
    FROM Field
    WHERE rerun = {RERUN}
    ORDER BY run, camcol, field
    """

    print("Querying SDSS SkyServer for verified fields...")
    print(f"  SQL: SELECT DISTINCT TOP {query_limit} run, camcol, field FROM Field WHERE rerun={RERUN}")

    params = {
        "cmd": sql,
        "format": "csv",
    }

    try:
        resp = requests.get(SKYSERVER_URL, params=params, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"\nError querying SkyServer: {e}")
        print("Make sure you have internet access and try again.")
        return []

    # Parse CSV response — skip SkyServer's "#Table1" header line
    text = resp.text.strip()
    lines = text.split("\n")
    # Remove comment/metadata lines that start with #
    lines = [l for l in lines if not l.startswith("#")]
    text = "\n".join(lines)

    if "error" in text.lower()[:200]:
        print(f"\nSkyServer returned an error:\n{text[:500]}")
        return []

    reader = csv.DictReader(io.StringIO(text))
    fields = []
    for row in reader:
        try:
            fields.append((
                int(row["run"]),
                int(row["camcol"]),
                int(row["field"]),
            ))
        except (KeyError, ValueError):
            continue

    print(f"  Got {len(fields)} verified fields from SkyServer")
    return fields


def generate_urls(verified_fields, n_files):
    """Generate download URLs from verified fields."""
    # Randomly sample to get diverse sky coverage
    if len(verified_fields) > n_files:
        selected = random.sample(verified_fields, n_files)
    else:
        selected = verified_fields

    urls = []
    for run, camcol, field in selected:
        run6 = f"{run:06d}"
        field4 = f"{field:04d}"
        for band in BANDS:
            url = (
                f"https://data.sdss.org/sas/dr17/eboss/photoObj/frames/"
                f"{RERUN}/{run}/{camcol}/frame-{band}-{run6}-{camcol}-{field4}.fits.bz2"
            )
            urls.append(url)

    return urls


async def download_one(session, semaphore, url, dest_dir, progress):
    """Download a single FITS file."""
    async with semaphore:
        filename = url.split("/")[-1]
        filepath = os.path.join(dest_dir, filename)

        if os.path.exists(filepath):
            progress["skipped"] += 1
            return

        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status == 404:
                    progress["missing"] += 1
                    return
                if resp.status != 200:
                    print(f"  ✗ HTTP {resp.status}: {filename}")
                    progress["failed"] += 1
                    return

                size = 0
                with open(filepath, "wb") as f:
                    async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)
                        size += len(chunk)

                size_mb = size / (1024 * 1024)
                progress["done"] += 1
                print(f"  ✓ [{progress['done']}/{progress['target']}] {filename} ({size_mb:.1f} MB)")

        except asyncio.TimeoutError:
            print(f"  ✗ Timeout: {filename}")
            progress["failed"] += 1
        except Exception as e:
            print(f"  ✗ Error: {filename} — {e}")
            progress["failed"] += 1


async def main():
    dest_dir = os.path.abspath(DOWNLOAD_DIR)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Get verified fields from SDSS database
    verified_fields = query_sdss_fields(NUM_FILES)
    if not verified_fields:
        print("Failed to get field list from SkyServer. Exiting.")
        return

    # Step 2: Generate URLs from verified fields
    file_urls = generate_urls(verified_fields, NUM_FILES)
    total_files = len(file_urls)

    print(f"\nSDSS Bulk FITS Downloader v2")
    print("============================")
    print(f"  Verified fields: {len(verified_fields)}")
    print(f"  Files to download: {total_files}")
    print(f"  Bands: {', '.join(BANDS)}")
    print(f"  Destination: {dest_dir}")
    print(f"  Concurrent downloads: {MAX_CONCURRENT}")
    print()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    progress = {"done": 0, "failed": 0, "missing": 0, "skipped": 0, "target": total_files}

    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [download_one(session, semaphore, url, dest_dir, progress) for url in file_urls]
        await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"  Downloaded: {progress['done']}")
    print(f"  Not found:  {progress['missing']}")
    print(f"  Failed:     {progress['failed']}")
    print(f"  Skipped:    {progress['skipped']} (already existed)")
    print(f"\nFiles saved to: {dest_dir}")

    if progress["done"] > 0:
        print(f"\nTo decompress: bunzip2 {dest_dir}/*.bz2")
        print("Or read directly: astropy.io.fits.open('file.fits.bz2')")


if __name__ == "__main__":
    asyncio.run(main())