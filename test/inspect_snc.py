"""
inspect_snc.py - Inspect the contents of an XLTEK .snc (sync) file

Loads the SNC file path from ../.env (variable: snc_file)
Uses the project's snc_reader module to parse and summarize the file.
"""

import sys
import os
import struct
from pathlib import Path
from datetime import timedelta

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

# Add parent directory to path so we can import snc_reader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from snc_reader import read_snc_file, GENERIC_HEADER_END, SNC_ENTRY_SIZE


def inspect_snc(snc_path: Path):
    """Print a human-readable summary of an SNC file."""
    
    # --- File-level info ---
    file_size = snc_path.stat().st_size
    data_size = file_size - GENERIC_HEADER_END
    expected_entries = data_size // SNC_ENTRY_SIZE
    remainder_bytes = data_size % SNC_ENTRY_SIZE

    print("=" * 65)
    print("SNC FILE INSPECTION")
    print("=" * 65)
    print(f"File:           {snc_path}")
    print(f"File size:      {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"Header size:    {GENERIC_HEADER_END} bytes")
    print(f"Data size:      {data_size:,} bytes")
    print(f"Entry size:     {SNC_ENTRY_SIZE} bytes each")
    print(f"Expected entries: {expected_entries}")
    if remainder_bytes:
        print(f"⚠ Remainder:    {remainder_bytes} bytes (not a clean multiple of entry size)")

    # --- Peek at raw header ---
    print(f"\n{'─' * 65}")
    print("RAW HEADER (first 352 bytes)")
    print("─" * 65)
    with open(snc_path, 'rb') as f:
        header_bytes = f.read(GENERIC_HEADER_END)
    
    # Check for readable strings in header
    # Try to find a null-terminated string at the start (often a format identifier)
    printable = []
    for b in header_bytes[:64]:
        if 32 <= b < 127:
            printable.append(chr(b))
        elif b == 0 and printable:
            break
        else:
            printable.append('.')
    print(f"First 64 bytes (text): {''.join(printable)}")
    
    # Show some key offsets as little-endian uint32
    print(f"Bytes  0-3  (uint32): {struct.unpack_from('<I', header_bytes, 0)[0]}")
    print(f"Bytes  4-7  (uint32): {struct.unpack_from('<I', header_bytes, 4)[0]}")
    print(f"Bytes  8-11 (uint32): {struct.unpack_from('<I', header_bytes, 8)[0]}")

    # --- Parse sync entries ---
    print(f"\n{'─' * 65}")
    print("SYNC ENTRIES")
    print("─" * 65)
    
    entries = read_snc_file(snc_path)
    print(f"Parsed entries: {len(entries)}")
    
    if not entries:
        print("No sync entries found!")
        return

    # --- Time range ---
    first = entries[0]
    last = entries[-1]
    duration_usec = last.unix_time_usec - first.unix_time_usec
    duration_sec = duration_usec / 1_000_000
    duration = timedelta(seconds=duration_sec)

    print(f"\n{'─' * 65}")
    print("TIME RANGE")
    print("─" * 65)
    print(f"First timestamp:  {first.datetime.strftime('%Y-%m-%d %H:%M:%S.%f %Z')}")
    print(f"Last timestamp:   {last.datetime.strftime('%Y-%m-%d %H:%M:%S.%f %Z')}")
    print(f"Duration:         {duration}  ({duration_sec:.3f} seconds)")
    print(f"First sample #:   {first.sample_stamp:,}")
    print(f"Last sample #:    {last.sample_stamp:,}")
    
    sample_span = last.sample_stamp - first.sample_stamp
    print(f"Sample span:      {sample_span:,} samples")
    
    if sample_span > 0 and duration_sec > 0:
        derived_rate = sample_span / duration_sec
        print(f"Derived sample rate: {derived_rate:.2f} Hz")

    # --- Entry spacing statistics ---
    if len(entries) >= 2:
        print(f"\n{'─' * 65}")
        print("ENTRY SPACING STATISTICS")
        print("─" * 65)
        
        sample_diffs = []
        time_diffs_sec = []
        
        for i in range(1, len(entries)):
            ds = entries[i].sample_stamp - entries[i-1].sample_stamp
            dt = (entries[i].unix_time_usec - entries[i-1].unix_time_usec) / 1_000_000
            sample_diffs.append(ds)
            time_diffs_sec.append(dt)
        
        print(f"{'':>25s}  {'Samples':>12s}  {'Time (sec)':>12s}")
        print(f"{'Min':>25s}  {min(sample_diffs):>12,}  {min(time_diffs_sec):>12.4f}")
        print(f"{'Max':>25s}  {max(sample_diffs):>12,}  {max(time_diffs_sec):>12.4f}")
        print(f"{'Mean':>25s}  {sum(sample_diffs)/len(sample_diffs):>12,.1f}  {sum(time_diffs_sec)/len(time_diffs_sec):>12.4f}")
        
        # Check for any anomalies (large gaps or zero-length intervals)
        anomalies = [(i, sample_diffs[i-1], time_diffs_sec[i-1]) 
                     for i in range(1, len(entries))
                     if time_diffs_sec[i-1] <= 0 or sample_diffs[i-1] <= 0]
        if anomalies:
            print(f"\n⚠ Found {len(anomalies)} anomalous intervals (zero or negative):")
            for idx, ds, dt in anomalies[:5]:
                print(f"  Entry {idx}: Δsamples={ds}, Δtime={dt:.6f}s")

    # --- First and last entries ---
    n_show = min(5, len(entries))
    
    print(f"\n{'─' * 65}")
    print(f"FIRST {n_show} ENTRIES")
    print("─" * 65)
    print(f"{'#':>5s}  {'Sample':>12s}  {'Timestamp':>28s}  {'Unix µs':>20s}")
    for i, e in enumerate(entries[:n_show]):
        print(f"{i:>5d}  {e.sample_stamp:>12,}  {e.datetime.strftime('%Y-%m-%d %H:%M:%S.%f'):>28s}  {e.unix_time_usec:>20,}")

    if len(entries) > 2 * n_show:
        print(f"  ... ({len(entries) - 2*n_show} entries omitted) ...")

    if len(entries) > n_show:
        print(f"\n{'─' * 65}")
        print(f"LAST {n_show} ENTRIES")
        print("─" * 65)
        print(f"{'#':>5s}  {'Sample':>12s}  {'Timestamp':>28s}  {'Unix µs':>20s}")
        for i, e in enumerate(entries[-n_show:], start=len(entries)-n_show):
            print(f"{i:>5d}  {e.sample_stamp:>12,}  {e.datetime.strftime('%Y-%m-%d %H:%M:%S.%f'):>28s}  {e.unix_time_usec:>20,}")

    # --- Drift / jitter analysis ---
    if len(entries) >= 3 and sample_span > 0 and duration_sec > 0:
        analyze_drift(entries, derived_rate, snc_path)

    print(f"\n{'=' * 65}")
    print("DONE")
    print("=" * 65)


def analyze_drift(entries, sample_rate, snc_path):
    """
    Compare actual SNC timestamps against ideal (first-sync + constant rate)
    as assumed by xltek_to_mat.py.  Print stats and save plots.
    """
    output_dir = Path(__file__).resolve().parent

    # Build arrays
    samples = np.array([e.sample_stamp for e in entries], dtype=np.float64)
    actual_usec = np.array([e.unix_time_usec for e in entries], dtype=np.float64)

    # Ideal timestamps: same formula as get_sample_timestamps()
    base_time = actual_usec[0]
    base_sample = samples[0]
    sample_interval_usec = 1_000_000.0 / sample_rate
    ideal_usec = base_time + (samples - base_sample) * sample_interval_usec

    # Residuals in milliseconds (actual - ideal)
    residual_ms = (actual_usec - ideal_usec) / 1_000.0

    # Elapsed time in hours (for x-axis and drift rate)
    elapsed_sec = (actual_usec - actual_usec[0]) / 1_000_000.0
    elapsed_hr = elapsed_sec / 3600.0

    # Linear fit to residuals vs elapsed time
    coeffs = np.polyfit(elapsed_hr, residual_ms, 1)  # [slope, intercept]
    drift_ms_per_hr = coeffs[0]
    fit_line = np.polyval(coeffs, elapsed_hr)

    # Detrended residuals (jitter)
    detrended_ms = residual_ms - fit_line
    jitter_std_ms = np.std(detrended_ms)

    # Drift in ppm:  drift_ms_per_hr / (3600 * 1000) * 1e6 = drift_ms_per_hr / 3.6
    drift_ppm = drift_ms_per_hr / 3.6

    # --- Print stats ---
    print(f"\n{'─' * 65}")
    print("DRIFT & JITTER ANALYSIS  (actual SNC vs ideal first-sync + Fs)")
    print("─" * 65)
    print(f"Sample rate assumed:          {sample_rate:.2f} Hz")
    print(f"Number of sync points:        {len(entries)}")
    print(f"")
    print(f"  Residual (actual − ideal):")
    print(f"    Min:                      {residual_ms.min():+.4f} ms")
    print(f"    Max:                      {residual_ms.max():+.4f} ms")
    print(f"    Mean:                     {residual_ms.mean():+.4f} ms")
    print(f"    Max |error|:              {np.max(np.abs(residual_ms)):.4f} ms")
    print(f"")
    print(f"  Linear drift:")
    print(f"    Rate:                     {drift_ms_per_hr:+.4f} ms/hour")
    print(f"    Rate:                     {drift_ppm:+.4f} ppm")
    print(f"    Accumulated over {elapsed_hr[-1]:.1f} hr:  {fit_line[-1] - fit_line[0]:+.4f} ms")
    print(f"")
    print(f"  Jitter (detrended std):     {jitter_std_ms:.4f} ms")
    print(f"  Jitter (detrended max |e|): {np.max(np.abs(detrended_ms)):.4f} ms")

    # --- Plot 1: Drift ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(elapsed_hr, residual_ms, 'o-', markersize=4, label='Residual (actual − ideal)')
    ax.plot(elapsed_hr, fit_line, '--', color='red', linewidth=1.5,
            label=f'Linear fit: {drift_ms_per_hr:+.4f} ms/hr ({drift_ppm:+.2f} ppm)')
    ax.set_xlabel('Elapsed time (hours)')
    ax.set_ylabel('Residual (ms)')
    ax.set_title('SNC Sync Drift vs. Ideal (first sync + constant Fs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    drift_path = output_dir / 'sync_drift.png'
    fig.savefig(drift_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {drift_path}")

    # --- Plot 2: Jitter (detrended) ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(elapsed_hr, detrended_ms, 'o-', markersize=4, color='C1')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axhline(+jitter_std_ms, color='red', linewidth=0.8, linestyle=':', label=f'±1σ = {jitter_std_ms:.4f} ms')
    ax.axhline(-jitter_std_ms, color='red', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Elapsed time (hours)')
    ax.set_ylabel('Detrended residual (ms)')
    ax.set_title('SNC Sync Jitter (linear drift removed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    jitter_path = output_dir / 'sync_jitter.png'
    fig.savefig(jitter_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {jitter_path}")


def main():
    # Load .env from project root (one level up from test/)
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)
    
    load_dotenv(env_path)
    
    snc_file = os.getenv('snc_file')
    if not snc_file:
        print("Error: 'snc_file' not set in .env")
        sys.exit(1)
    
    snc_path = Path(snc_file)
    if not snc_path.exists():
        print(f"Error: SNC file not found: {snc_path}")
        sys.exit(1)
    
    inspect_snc(snc_path)


if __name__ == "__main__":
    main()
