"""
erd_reader.py - Read XLTEK ERD (raw data) files

This module parses ERD file headers and extracts sample data using delta decompression,
based on the xltek2mef C implementation.

ERD File Structure:
- Bytes 0-351: Generic header
- Bytes 352-8655: File-specific header (sample freq, channels, headbox, etc.)
- Bytes 8656+: Sample data (delta-compressed)
"""

import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import math


# ERD File Offsets (from XLTek2Mayo.h)
GUID_OFFSET = 0
GUID_BYTES = 16
FILE_SCHEMA_OFFSET = 16
CREATION_TIME_OFFSET = 20
GENERIC_HEADER_END = 352
SAMP_FREQ_OFFSET = 352
NUM_CHANS_OFFSET = 360
DELTA_BITS_OFFSET = 364
PHYS_TO_STOR_CHAN_MAP_OFFSET = 368
HEADBOX_TYPE_ARRAY_OFFSET = 4464
HEADBOX_TYPE_ARRAY_SIZE = 4
DISCARD_BITS_OFFSET = 4556
SHORTED_CHANS_OFFSET = 4560
SKIP_FACT_OFFSET = 6608
SAMPLE_PACKET_OFFSET = 8656

# XLTEK constants
XLTEK_OVERFLOW_FLAG = 0xFFFF


@dataclass
class ERDHeader:
    """ERD file header information."""
    file_schema: int
    creation_time: Optional[datetime]
    sample_freq: float
    num_channels: int
    delta_bits: int
    discard_bits: int
    headbox_types: list[int]
    phys_chan_map: list[int]  # Physical channel mapping
    shorted_chans: list[int]  # Shorted (inactive) channels
    num_recorded_chans: int   # Channels actually recorded (non-shorted)
    rec_to_phys_map: list[int]  # Recorded to physical channel mapping


def read_erd_header(erd_path: str | Path) -> ERDHeader:
    """
    Read and parse ERD file header.
    
    Args:
        erd_path: Path to the .erd file
        
    Returns:
        ERDHeader with parsed header information
    """
    erd_path = Path(erd_path)
    
    with open(erd_path, 'rb') as f:
        # Read enough of the file for the header
        header_data = f.read(SAMPLE_PACKET_OFFSET)
    
    # File schema (2 bytes, little-endian)
    file_schema = struct.unpack_from('<H', header_data, FILE_SCHEMA_OFFSET)[0]
    
    # Creation time (4 bytes, little-endian, UTC time_t)
    creation_time_raw = struct.unpack_from('<I', header_data, CREATION_TIME_OFFSET)[0]
    try:
        creation_time = datetime.utcfromtimestamp(creation_time_raw)
    except (ValueError, OSError):
        creation_time = None
    
    # Sample frequency (8 bytes, little-endian double)
    sample_freq = struct.unpack_from('<d', header_data, SAMP_FREQ_OFFSET)[0]
    
    # Number of channels (4 bytes, little-endian)
    num_channels = struct.unpack_from('<i', header_data, NUM_CHANS_OFFSET)[0]
    
    # Delta bits (4 bytes, little-endian)
    delta_bits = struct.unpack_from('<i', header_data, DELTA_BITS_OFFSET)[0]
    
    # Discard bits (4 bytes, little-endian)
    discard_bits = struct.unpack_from('<i', header_data, DISCARD_BITS_OFFSET)[0]
    
    # Headbox types (4 x 4 bytes)
    headbox_types = []
    for i in range(HEADBOX_TYPE_ARRAY_SIZE):
        hbt = struct.unpack_from('<i', header_data, HEADBOX_TYPE_ARRAY_OFFSET + i * 4)[0]
        if hbt != 0:
            headbox_types.append(hbt)
    
    # Physical channel map (num_channels x 4 bytes)
    phys_chan_map = []
    for i in range(min(num_channels, 1024)):  # Max 1024 entries
        pcm = struct.unpack_from('<i', header_data, PHYS_TO_STOR_CHAN_MAP_OFFSET + i * 4)[0]
        phys_chan_map.append(pcm)
    
    # Shorted channels (1024 x 2 bytes, short-int flags)
    shorted_chans = []
    for i in range(min(num_channels, 1024)):
        sc = struct.unpack_from('<h', header_data, SHORTED_CHANS_OFFSET + i * 2)[0]
        shorted_chans.append(sc)
    
    # Build recorded-to-physical channel mapping (non-shorted channels)
    rec_to_phys_map = []
    for i in range(num_channels):
        if i < len(shorted_chans) and shorted_chans[i] == 0:
            rec_to_phys_map.append(i + 1)  # 1-indexed like C code
    
    num_recorded_chans = len(rec_to_phys_map)
    
    return ERDHeader(
        file_schema=file_schema,
        creation_time=creation_time,
        sample_freq=sample_freq,
        num_channels=num_channels,
        delta_bits=delta_bits,
        discard_bits=discard_bits,
        headbox_types=headbox_types,
        phys_chan_map=phys_chan_map,
        shorted_chans=shorted_chans,
        num_recorded_chans=num_recorded_chans,
        rec_to_phys_map=rec_to_phys_map
    )


def read_erd_samples(erd_path: str | Path, header: Optional[ERDHeader] = None, 
                     max_samples: Optional[int] = None) -> np.ndarray:
    """
    Read and decompress sample data from ERD file.
    
    The data is delta-compressed: each sample is stored as the difference
    from the previous sample value. Uses variable-length encoding:
    - 1 byte if delta mask bit is 0
    - 2 bytes if delta mask bit is 1
    - 4 bytes if 2-byte value is 0xFFFF (overflow)
    
    Args:
        erd_path: Path to the .erd file
        header: Pre-parsed header (optional, will be read if not provided)
        max_samples: Maximum number of samples to read (optional)
        
    Returns:
        numpy array of shape (num_channels, num_samples) with raw sample values
    """
    erd_path = Path(erd_path)
    
    if header is None:
        header = read_erd_header(erd_path)
    
    with open(erd_path, 'rb') as f:
        # Get file size
        f.seek(0, 2)
        file_size = f.tell()
        
        # Read sample data portion
        f.seek(SAMPLE_PACKET_OFFSET)
        sample_data = f.read()
    
    num_system_chans = header.num_channels
    num_recorded_chans = header.num_recorded_chans
    rec_to_phys_map = header.rec_to_phys_map
    
    # Delta mask length
    delta_mask_len = math.ceil(num_system_chans / 8)
    
    # Estimate max samples per channel
    max_samps_per_chan = math.ceil(
        (file_size - SAMPLE_PACKET_OFFSET) / (1 + delta_mask_len + num_recorded_chans)
    )
    
    if max_samples is not None:
        max_samps_per_chan = min(max_samps_per_chan, max_samples)
    
    # Allocate output arrays
    channel_data = np.zeros((num_recorded_chans, max_samps_per_chan), dtype=np.int32)
    chan_sums = np.zeros(num_recorded_chans, dtype=np.int32)
    overflow_chans = np.zeros(num_recorded_chans, dtype=np.uint8)
    
    # Parse sample packets
    pos = 0
    data_len = len(sample_data)
    num_samps = 0
    
    while pos < data_len and (max_samples is None or num_samps < max_samples):
        # Skip event byte
        pos += 1
        if pos >= data_len:
            break
        
        # Read delta mask
        if pos + delta_mask_len > data_len:
            break
        delta_mask = sample_data[pos:pos + delta_mask_len]
        pos += delta_mask_len
        
        overflows = False
        mask_bit_counter = 1  # 1-indexed
        
        for i in range(num_recorded_chans):
            # Skip mask bits for shorted channels
            while i < len(rec_to_phys_map) and rec_to_phys_map[i] != mask_bit_counter:
                # Move to next mask bit
                byte_idx = (mask_bit_counter - 1) // 8
                mask_bit_counter += 1
            
            # Get mask bit for this channel
            byte_idx = (mask_bit_counter - 1) // 8
            bit_idx = (mask_bit_counter - 1) % 8
            
            if byte_idx < len(delta_mask):
                mask_bit = (delta_mask[byte_idx] >> bit_idx) & 1
            else:
                mask_bit = 0
            
            if mask_bit:
                # 2-byte delta
                if pos + 2 > data_len:
                    break
                short_delta = struct.unpack_from('<h', sample_data, pos)[0]
                pos += 2
                
                if short_delta == -1 or (short_delta & 0xFFFF) == XLTEK_OVERFLOW_FLAG:
                    # Overflow - will read 4 bytes later
                    overflow_chans[i] = 1
                    overflows = True
                else:
                    chan_sums[i] += short_delta
            else:
                # 1-byte delta (signed)
                if pos >= data_len:
                    break
                byte_delta = struct.unpack_from('<b', sample_data, pos)[0]
                pos += 1
                chan_sums[i] += byte_delta
            
            mask_bit_counter += 1
        
        # Handle overflow values
        if overflows:
            for i in range(num_recorded_chans):
                if overflow_chans[i]:
                    if pos + 4 > data_len:
                        break
                    chan_sums[i] = struct.unpack_from('<i', sample_data, pos)[0]
                    pos += 4
                    overflow_chans[i] = 0
        
        # Store sample values
        channel_data[:, num_samps] = chan_sums
        num_samps += 1
    
    # Trim to actual number of samples
    channel_data = channel_data[:, :num_samps]
    
    return channel_data


def get_erd_files_from_data_dir(data_dir: str | Path) -> list[Path]:
    """
    Find all .erd files in a data directory, sorted by name.
    
    Args:
        data_dir: Directory containing XLTEK data files
        
    Returns:
        List of paths to .erd files, sorted
    """
    data_dir = Path(data_dir)
    erd_files = sorted(data_dir.glob("*.erd"))
    return erd_files


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Get data path
    datapath_file = Path(__file__).parent / "datapath.txt"
    if datapath_file.exists():
        with open(datapath_file) as f:
            data_dir = Path(f.readline().strip().strip("'\""))
    else:
        print("datapath.txt not found")
        sys.exit(1)
    
    # Find ERD files
    erd_files = get_erd_files_from_data_dir(data_dir)
    print(f"Found {len(erd_files)} ERD files")
    
    for erd_file in erd_files:
        print(f"\n{'='*60}")
        print(f"File: {erd_file.name}")
        print(f"{'='*60}")
        
        # Read header
        header = read_erd_header(erd_file)
        print(f"  Schema: {header.file_schema}")
        print(f"  Sample rate: {header.sample_freq} Hz")
        print(f"  Num channels (system): {header.num_channels}")
        print(f"  Num channels (recorded): {header.num_recorded_chans}")
        print(f"  Delta bits: {header.delta_bits}")
        print(f"  Discard bits: {header.discard_bits}")
        print(f"  Headbox types: {header.headbox_types}")
        if header.creation_time:
            print(f"  Creation time: {header.creation_time}")
        
        # Read first 1000 samples
        print("\n  Reading first 1000 samples...")
        try:
            samples = read_erd_samples(erd_file, header, max_samples=1000)
            print(f"  Sample data shape: {samples.shape}")
            print(f"  Sample range: [{samples.min()}, {samples.max()}]")
            print(f"  First channel first 10 samples: {samples[0, :10]}")
        except Exception as e:
            print(f"  Error reading samples: {e}")
            import traceback
            traceback.print_exc()
