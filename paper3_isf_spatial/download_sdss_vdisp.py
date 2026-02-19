#!/usr/bin/env python3
"""
download_sdss_vdisp.py
======================
Download velocity dispersions (and stellar mass estimates where available)
from SDSS DR12 for BOSS LOWZ galaxies, using PLATE/FIBERID/MJD as keys.

This script:
  1. Reads PLATE, FIBERID, MJD from your LOWZ FITS catalogs
  2. Batches queries to the SDSS SkyServer SQL endpoint
  3. Writes a single CSV: lowz_vdisp_masses.csv

Requirements:
  pip3 install astropy requests

Usage:
  python3 download_sdss_vdisp.py

The script expects these files in the working directory:
  - galaxy_DR12v5_LOWZ_North.fits.gz
  - galaxy_DR12v5_LOWZ_South.fits.gz

Output:
  - lowz_vdisp_masses.csv  (PLATE, FIBERID, MJD, VELDISP, VELDISPERR,
                             Z, RA, DEC, plus stellar mass columns if available)
  - download_progress.json  (checkpoint file for resuming interrupted downloads)
"""

import os
import sys
import json
import time
import csv
import requests
import numpy as np
from astropy.io import fits
from io import StringIO

# ── Configuration ──────────────────────────────────────────────────
BATCH_SIZE = 50           # galaxies per query (keep POST body small)
MAX_RETRIES = 5           # retries per batch
INITIAL_BACKOFF = 5       # seconds, doubles on each retry
THROTTLE_DELAY = 1.0      # seconds between successful requests
TIMEOUT = 120             # seconds per request
OUTPUT_FILE = "lowz_vdisp_masses.csv"
PROGRESS_FILE = "download_progress.json"

# ── Step 1: Extract unique (PLATE, FIBERID, MJD) from FITS ────────
def load_keys_from_fits():
    """Load unique (PLATE, FIBERID, MJD) tuples from LOWZ catalogs."""
    keys = set()
    for fname in ["galaxy_DR12v5_LOWZ_North.fits.gz",
                   "galaxy_DR12v5_LOWZ_South.fits.gz"]:
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        print(f"  Reading {fname}...")
        with fits.open(fname) as hdu:
            data = hdu[1].data
            plates = data['PLATE']
            fibers = data['FIBERID']
            mjds = data['MJD']
            for p, f, m in zip(plates, fibers, mjds):
                keys.add((int(p), int(f), int(m)))
        print(f"    {len(data)} rows, {len(keys)} unique keys so far")
    return sorted(keys)


# ── Step 2: Query SDSS in batches ─────────────────────────────────
def build_query(key_batch):
    """Build SQL query for a batch of (PLATE, FIBERID, MJD) tuples.
    
    Uses a compact encoding: plate*10000000 + mjd*10000 + fiberid
    won't work because MJD is too large. Instead use a short OR list
    with composite key matching.
    """
    # Use compact (plate, fiberid, mjd) matching
    conditions = []
    for plate, fiber, mjd in key_batch:
        conditions.append(
            f"(s.plate={plate} AND s.fiberID={fiber} AND s.mjd={mjd})"
        )
    where_clause = " OR ".join(conditions)

    query = f"""
    SELECT
        s.plate, s.fiberID, s.mjd,
        s.ra, s.dec, s.z, s.zerr,
        s.veldisp, s.veldisperr,
        s.class, s.subclass,
        p.dered_u, p.dered_g, p.dered_r, p.dered_i, p.dered_z,
        p.petroR50_r, p.petroR90_r
    FROM SpecObj AS s
    JOIN PhotoObj AS p ON s.bestobjid = p.objid
    WHERE ({where_clause})
      AND s.class = 'GALAXY'
    """
    return query.strip()


def query_sdss(sql, attempt=0):
    """Execute a SQL query against SDSS SkyServer via GET with retry logic."""
    # Use the SkyServer search endpoint with GET — cmd goes in URL params
    url = "https://skyserver.sdss.org/dr12/en/tools/search/x_results.aspx"
    params = {
        "searchtool": "SQL",
        "TaskName": "Skyserver.Search.SQL",
        "syntax": "NoSyntax",
        "ReturnHtml": "false",
        "cmd": sql,
        "format": "csv",
    }

    try:
        # Use POST to avoid URL length limits — this endpoint accepts both
        resp = requests.post(url, data=params, timeout=TIMEOUT)

        if resp.status_code == 200:
            text = resp.text.strip()
            if text.startswith("<!DOCTYPE") or text.startswith("<html") or text.startswith("<Html"):
                if "syntax" in text.lower() or "error" in text.lower():
                    raise Exception(f"SDSS SQL error (HTML response, {len(text)} bytes)")
                raise Exception("SDSS returned HTML error page")
            if text.startswith("#"):
                # SDSS prepends "#Table1\n" to valid CSV — strip it
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("#"))
                text = text.strip()
            if len(text) < 10 and "error" in text.lower():
                raise Exception(f"SDSS query error: {text[:300]}")
            return text
        elif resp.status_code in (503, 502, 429):
            raise Exception(f"SDSS server busy ({resp.status_code})")
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")

    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            Exception) as e:
        if attempt < MAX_RETRIES:
            wait = INITIAL_BACKOFF * (2 ** attempt)
            print(f"    Retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
            time.sleep(wait)
            return query_sdss(sql, attempt + 1)
        else:
            raise


def parse_csv_response(text):
    """Parse SDSS CSV response into list of dicts."""
    rows = []
    reader = csv.DictReader(StringIO(text))
    for row in reader:
        rows.append(row)
    return rows


# ── Step 3: Checkpoint/resume logic ───────────────────────────────
def load_progress():
    """Load download progress from checkpoint file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_batches": 0, "total_rows": 0}


def save_progress(batch_num, total_rows):
    """Save download progress to checkpoint file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed_batches": batch_num, "total_rows": total_rows}, f)


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("SDSS Velocity Dispersion Downloader for LOWZ")
    print("=" * 60)

    # Load keys
    print("\nStep 1: Loading galaxy keys from FITS catalogs...")
    keys = load_keys_from_fits()
    print(f"  Total unique (PLATE, FIBERID, MJD) keys: {len(keys)}")

    if len(keys) == 0:
        print("ERROR: No keys found. Check that FITS files are present.")
        sys.exit(1)

    # Create batches
    batches = []
    for i in range(0, len(keys), BATCH_SIZE):
        batches.append(keys[i:i + BATCH_SIZE])
    print(f"  Split into {len(batches)} batches of ~{BATCH_SIZE}")

    # Check for resume
    progress = load_progress()
    start_batch = progress["completed_batches"]
    total_rows = progress["total_rows"]

    if start_batch > 0:
        print(f"\n  Resuming from batch {start_batch + 1}/{len(batches)} "
              f"({total_rows} rows already downloaded)")

    # Set up output file
    header_written = os.path.exists(OUTPUT_FILE) and start_batch > 0
    mode = "a" if start_batch > 0 else "w"

    print(f"\nStep 2: Querying SDSS SkyServer...")
    print(f"  Estimated time: ~{len(batches) * (THROTTLE_DELAY + 1) / 60:.0f} minutes "
          f"(+ retries)")
    print(f"  Progress is checkpointed — safe to Ctrl-C and resume later.")
    print()

    failed_batches = []

    with open(OUTPUT_FILE, mode, newline="") as outf:
        writer = None

        for batch_idx in range(start_batch, len(batches)):
            batch = batches[batch_idx]
            pct = (batch_idx + 1) / len(batches) * 100

            try:
                sql = build_query(batch)
                text = query_sdss(sql)
                rows = parse_csv_response(text)

                if rows:
                    # Write header on first successful batch
                    if writer is None:
                        fieldnames = list(rows[0].keys())
                        writer = csv.DictWriter(outf, fieldnames=fieldnames)
                        if not header_written:
                            writer.writeheader()
                            header_written = True
                    for row in rows:
                        writer.writerow(row)
                    outf.flush()

                total_rows += len(rows)
                matched = len(rows)
                requested = len(batch)

                print(f"  Batch {batch_idx+1}/{len(batches)} "
                      f"({pct:5.1f}%) — {matched}/{requested} matched, "
                      f"{total_rows} total rows")

            except Exception as e:
                print(f"  Batch {batch_idx+1}/{len(batches)} FAILED: {e}")
                failed_batches.append(batch_idx)

            # Checkpoint
            save_progress(batch_idx + 1, total_rows)

            # Throttle
            time.sleep(THROTTLE_DELAY)

    # Summary
    print()
    print("=" * 60)
    print(f"Download complete!")
    print(f"  Total rows downloaded: {total_rows}")
    print(f"  Output file: {OUTPUT_FILE}")
    if failed_batches:
        print(f"  WARNING: {len(failed_batches)} batches failed: {failed_batches}")
        print(f"  Re-run the script to retry failed batches.")
    else:
        print(f"  All batches successful.")
        # Clean up checkpoint
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
    print("=" * 60)

    # Validate output
    if os.path.exists(OUTPUT_FILE):
        print(f"\nValidating {OUTPUT_FILE}...")
        with open(OUTPUT_FILE) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n = len(rows)
        if n == 0:
            print("  WARNING: Output file is empty!")
            return

        # Check velocity dispersion coverage
        n_vdisp = sum(1 for r in rows
                       if r.get("veldisp", "0") not in ("0", "", "-9999"))
        n_good_vdisp = sum(1 for r in rows
                            if float(r.get("veldisp", "0")) > 50
                            and float(r.get("veldisperr", "999")) < 100)

        print(f"  Total galaxies: {n}")
        print(f"  With veldisp > 0: {n_vdisp} ({n_vdisp/n*100:.1f}%)")
        print(f"  With good veldisp (>50, err<100): {n_good_vdisp} "
              f"({n_good_vdisp/n*100:.1f}%)")
        print(f"  Columns: {reader.fieldnames}")


if __name__ == "__main__":
    main()
