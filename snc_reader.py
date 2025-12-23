"""
snc_reader.py - Read XLTEK SNC (sync) files for timestamp information

SNC files provide the mapping between sample numbers and wall-clock time.
The format is:
- 352 bytes: Generic header
- Repeating entries (12 bytes each):
  - 4 bytes: sample_stamp (sample number)
  - 8 bytes: file_time (Windows FILETIME - 100ns intervals since 1601-01-01)
"""

import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import numpy as np


# Windows FILETIME epoch starts 1601-01-01, Unix epoch starts 1970-01-01
# Difference in seconds: ~369 years
FILETIME_UNIX_DIFF = 11644473600  # seconds from 1601 to 1970

GENERIC_HEADER_END = 352
SNC_ENTRY_SIZE = 12  # 4 bytes sample + 8 bytes time


@dataclass
class SyncEntry:
    """A single sync entry with sample number and timestamp."""
    sample_stamp: int
    file_time: int  # Raw Windows FILETIME
    unix_time_usec: int  # Microseconds since Unix epoch
    
    @property
    def datetime(self) -> datetime:
        """Convert to datetime object."""
        return datetime.fromtimestamp(self.unix_time_usec / 1_000_000, tz=timezone.utc)


def filetime_to_unix_usec(file_time: int) -> int:
    """
    Convert Windows FILETIME to Unix microseconds.
    
    FILETIME is 100-nanosecond intervals since January 1, 1601.
    We convert to microseconds since January 1, 1970.
    """
    # Convert from 100ns to microseconds (divide by 10)
    usec = file_time // 10
    # Subtract the difference between Windows and Unix epochs
    usec -= FILETIME_UNIX_DIFF * 1_000_000
    return usec


def read_snc_file(snc_path: str | Path) -> list[SyncEntry]:
    """
    Read SNC file and extract sync entries.
    
    Args:
        snc_path: Path to the .snc file
        
    Returns:
        List of SyncEntry objects
    """
    snc_path = Path(snc_path)
    
    with open(snc_path, 'rb') as f:
        # Get file size
        f.seek(0, 2)
        file_size = f.tell()
        
        # Read data portion (after header)
        f.seek(GENERIC_HEADER_END)
        data = f.read()
    
    entries = []
    pos = 0
    
    while pos + SNC_ENTRY_SIZE <= len(data):
        sample_stamp = struct.unpack_from('<I', data, pos)[0]
        file_time = struct.unpack_from('<Q', data, pos + 4)[0]
        
        unix_usec = filetime_to_unix_usec(file_time)
        
        entries.append(SyncEntry(
            sample_stamp=sample_stamp,
            file_time=file_time,
            unix_time_usec=unix_usec
        ))
        
        pos += SNC_ENTRY_SIZE
    
    return entries


def get_sample_timestamps(sync_entries: list[SyncEntry], 
                          sample_freq: float,
                          num_samples: int) -> np.ndarray:
    """
    Interpolate timestamps for all samples based on sync entries.
    
    Args:
        sync_entries: List of sync entries from SNC file
        sample_freq: Sample rate in Hz
        num_samples: Total number of samples
        
    Returns:
        Array of timestamps in microseconds since Unix epoch
    """
    if not sync_entries:
        raise ValueError("No sync entries provided")
    
    # Calculate sample interval in microseconds
    sample_interval_usec = 1_000_000 / sample_freq
    
    # Get first sync entry as reference
    first_entry = sync_entries[0]
    
    # Generate timestamps for all samples
    # timestamp[n] = first_timestamp + (n - first_sample) * sample_interval
    base_time = first_entry.unix_time_usec
    base_sample = first_entry.sample_stamp
    
    sample_nums = np.arange(num_samples)
    timestamps = base_time + (sample_nums - base_sample) * sample_interval_usec
    
    return timestamps.astype(np.int64)


def get_start_time(snc_path: str | Path) -> Optional[datetime]:
    """
    Get the recording start time from SNC file.
    
    Args:
        snc_path: Path to the .snc file
        
    Returns:
        datetime of recording start, or None if file can't be read
    """
    try:
        entries = read_snc_file(snc_path)
        if entries:
            return entries[0].datetime
    except Exception:
        pass
    return None


def get_snc_file_from_data_dir(data_dir: str | Path) -> Optional[Path]:
    """
    Find the .snc file in a data directory.
    
    Args:
        data_dir: Directory containing XLTEK data files
        
    Returns:
        Path to the .snc file, or None if not found
    """
    data_dir = Path(data_dir)
    snc_files = list(data_dir.glob("*.snc"))
    
    if not snc_files:
        return None
    
    return snc_files[0]


if __name__ == "__main__":
    import sys
    
    # Get data path
    datapath_file = Path(__file__).parent / "datapath.txt"
    if datapath_file.exists():
        with open(datapath_file) as f:
            data_dir = Path(f.readline().strip().strip("'\""))
    else:
        print("datapath.txt not found")
        sys.exit(1)
    
    # Find SNC file
    snc_file = get_snc_file_from_data_dir(data_dir)
    
    if snc_file is None:
        print("No SNC file found")
        sys.exit(1)
    
    print(f"SNC file: {snc_file.name}")
    
    # Read sync entries
    entries = read_snc_file(snc_file)
    print(f"Found {len(entries)} sync entries")
    
    if entries:
        print(f"\nFirst entry:")
        print(f"  Sample: {entries[0].sample_stamp}")
        print(f"  Time: {entries[0].datetime}")
        
        print(f"\nLast entry:")
        print(f"  Sample: {entries[-1].sample_stamp}")
        print(f"  Time: {entries[-1].datetime}")
        
        # Calculate recording duration
        duration_usec = entries[-1].unix_time_usec - entries[0].unix_time_usec
        duration_sec = duration_usec / 1_000_000
        duration_min = duration_sec / 60
        print(f"\nRecording duration: {duration_min:.2f} minutes ({duration_sec:.1f} seconds)")
        
        # Calculate sample rate from sync data
        if len(entries) >= 2:
            sample_diff = entries[-1].sample_stamp - entries[0].sample_stamp
            if sample_diff > 0:
                derived_freq = sample_diff / duration_sec
                print(f"Derived sample rate: {derived_freq:.2f} Hz")
