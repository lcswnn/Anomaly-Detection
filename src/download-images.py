"""
SDSS Bulk FITS Downloader for Anomaly Detection Training Data
=============================================================
Downloads calibrated imaging frames from the Sloan Digital Sky Survey (DR17).
Each frame is a ~2048x1489 pixel image of a patch of sky.

Downloads across all 5 photometric bands (u, g, r, i, z) to give diverse data.

Usage:
    pip install aiohttp
    python download_sdss_fits.py

Configuration:
    - NUM_FIELDS: how many unique sky fields to download (each has 5 bands)
    - BANDS: which photometric bands to include
    - DOWNLOAD_DIR: where to save files
"""

import asyncio
import aiohttp
import os
import time
import random
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
DOWNLOAD_DIR = "../data/raw"
MAX_CONCURRENT = 10          # simultaneous downloads (be nice to SDSS servers)
TIMEOUT_SECONDS = 120        # per-file timeout
CHUNK_SIZE = 1024 * 1024     # 1 MB chunks

NUM_FIELDS = 500             # number of unique sky fields to download
BANDS = ["r"]                # bands to download per field: ["u", "g", "r", "i", "z"]
                             # Start with just "r" band for ~500 files
                             # Set to ["u", "g", "r", "i", "z"] for ~2500 files (5 per field)
RERUN = 301                  # standard rerun for SDSS imaging

# ── SDSS Run/Camcol/Field combinations ────────────────────────
# These are known valid SDSS imaging runs with many fields.
# Each (run, camcol) pair has hundreds of fields.
# We sample across many runs to get diverse sky coverage.
SDSS_RUNS = [
    # (run, camcol, max_field) - curated set of productive SDSS runs
    (752, 1, 500), (752, 2, 500), (752, 3, 500),
    (756, 1, 400), (756, 2, 400),
    (1033, 1, 500), (1033, 3, 500),
    (1140, 1, 250), (1140, 4, 250),
    (1231, 2, 300), (1231, 5, 300),
    (1336, 1, 400), (1336, 3, 400),
    (1339, 2, 350), (1339, 4, 350),
    (1356, 1, 300), (1356, 6, 300),
    (1359, 2, 250), (1359, 5, 250),
    (1740, 1, 400), (1740, 3, 400),
    (2078, 2, 300), (2078, 4, 300),
    (2190, 1, 250), (2190, 3, 250),
    (2505, 1, 350), (2505, 3, 350),
    (2570, 2, 300), (2570, 4, 300),
    (2583, 1, 250), (2583, 5, 250),
    (2662, 1, 300), (2662, 3, 300),
    (2700, 2, 250), (2700, 6, 250),
    (2728, 1, 300), (2728, 4, 300),
    (2820, 1, 250), (2820, 3, 250),
    (2873, 2, 300), (2873, 5, 300),
    (3225, 1, 200), (3225, 3, 200),
    (3325, 2, 250), (3325, 4, 250),
    (3388, 1, 300), (3388, 3, 300),
    (3434, 2, 200), (3434, 6, 200),
    (3437, 1, 250),
    (4128, 1, 250), (4128, 3, 250),
    (4145, 2, 200), (4145, 4, 200),
    (4157, 1, 200), (4157, 3, 200),
    (4184, 2, 250), (4184, 5, 250),
    (4188, 1, 200), (4188, 4, 200),
    (4198, 1, 200),
    (4207, 2, 200), (4207, 3, 200),
    (4263, 1, 200), (4263, 4, 200),
    (4288, 2, 200), (4288, 5, 200),
    (5314, 1, 200), (5314, 3, 200),
    (5566, 2, 200), (5566, 4, 200),
    (5590, 1, 200), (5590, 3, 200),
    (5597, 2, 150), (5597, 5, 150),
    (5610, 1, 200), (5610, 4, 200),
    (5633, 2, 150), (5633, 3, 150),
    (5642, 1, 150), (5642, 6, 150),
    (5646, 1, 150), (5646, 3, 150),
    (5709, 2, 200), (5709, 4, 200),
    (5744, 1, 200), (5744, 3, 200),
    (5759, 2, 150), (5759, 5, 150),
    (5770, 1, 150),
    (5776, 1, 200), (5776, 3, 200),
    (5781, 2, 200), (5781, 4, 200),
    (5792, 1, 150), (5792, 6, 150),
    (5800, 1, 200), (5800, 3, 200),
    (5813, 2, 150), (5813, 5, 150),
    (5823, 1, 150), (5823, 4, 150),
    (5836, 2, 200), (5836, 3, 200),
    (5853, 1, 150), (5853, 6, 150),
    (5866, 1, 200), (5866, 4, 200),
    (5878, 2, 150), (5878, 5, 150),
    (5898, 1, 150), (5898, 3, 150),
    (5905, 2, 200), (5905, 4, 200),
    (6074, 1, 200), (6074, 3, 200),
    (6162, 2, 150), (6162, 4, 150),
    (6283, 1, 150), (6283, 5, 150),
    (6314, 2, 200), (6314, 3, 200),
    (6383, 1, 150), (6383, 4, 150),
    (6442, 2, 150), (6442, 6, 150),
    (6462, 1, 200), (6462, 3, 200),
    (6476, 2, 150), (6476, 5, 150),
    (6501, 1, 150), (6501, 4, 150),
    (6513, 2, 200), (6513, 3, 200),
    (6537, 1, 150), (6537, 6, 150),
    (6548, 1, 200), (6548, 4, 200),
    (6564, 2, 150), (6564, 5, 150),
    (6580, 1, 150), (6580, 3, 150),
    (6596, 2, 200), (6596, 4, 200),
    (6614, 1, 150), (6614, 6, 150),
    (6625, 1, 200), (6625, 3, 200),
    (6640, 2, 150), (6640, 5, 150),
    (6648, 1, 150), (6648, 4, 150),
]


def generate_urls(n_fields):
    """Generate n random SDSS frame URLs across diverse sky regions."""
    urls = []
    fields_picked = set()

    while len(fields_picked) < n_fields:
        run, camcol, max_field = random.choice(SDSS_RUNS)
        field = random.randint(11, min(max_field, 500))  # fields start around 11
        key = (run, camcol, field)
        if key in fields_picked:
            continue
        fields_picked.add(key)

        for band in BANDS:
            run6 = f"{run:06d}"
            url = (
                f"https://data.sdss.org/sas/dr17/eboss/photoObj/frames/"
                f"{RERUN}/{run}/{camcol}/frame-{band}-{run6}-{camcol}-{field:04d}.fits.bz2"
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
                    # This run/field combo doesn't exist, skip silently
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
                total = progress["done"] + progress["failed"] + progress["missing"]
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

    print("Generating SDSS frame URLs...")
    file_urls = generate_urls(NUM_FIELDS)
    total_files = len(file_urls)

    print(f"\nSDSS Bulk FITS Downloader")
    print(f"========================")
    print(f"  Fields: {NUM_FIELDS}")
    print(f"  Bands:  {', '.join(BANDS)}")
    print(f"  Total files to attempt: {total_files}")
    print(f"  Destination: {dest_dir}")
    print(f"  Concurrent downloads: {MAX_CONCURRENT}")
    print(f"\n  Note: Some run/field combos may not exist (404s are normal).\n")

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
    print(f"  Not found:  {progress['missing']} (normal — not all fields exist)")
    print(f"  Failed:     {progress['failed']}")
    print(f"  Skipped:    {progress['skipped']} (already existed)")
    print(f"\nFiles saved to: {dest_dir}")

    if progress["done"] > 0:
        print(f"\nTip: Files are .fits.bz2 (bzip2 compressed).")
        print(f"     To decompress: bunzip2 {dest_dir}/*.bz2")
        print(f"     Or read directly with astropy:")
        print(f"       from astropy.io import fits")
        print(f"       hdul = fits.open('file.fits.bz2')")


if __name__ == "__main__":
    asyncio.run(main())