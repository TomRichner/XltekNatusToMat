"""
montage_extractor.py - Extract channel montage from XLTEK .ent files

This module provides functions to extract actual channel names from .ent files,
mimicking the behavior of xltek2mef's find_montage_in_ent() C function.

The .ent file contains embedded "ChanNames" strings with the actual electrode
names used in the recording, which may differ from the hardcoded names in
XltekDataReader's conversion.json.
"""

import re
from pathlib import Path
from typing import Optional


def find_all_occurrences(buffer: bytes, pattern: bytes) -> list[int]:
    """Find all starting positions of pattern in buffer."""
    positions = []
    start = 0
    while True:
        pos = buffer.find(pattern, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def parse_channel_names(raw_string: str) -> list[str]:
    """
    Parse channel names from a ChanNames data string.
    
    The format is comma-separated quoted channel names like:
    '"Fp1","Fp2","F7","F3",...'
    
    Args:
        raw_string: Raw string containing channel name data
        
    Returns:
        List of channel names with spaces/slashes replaced by dashes
    """
    # Extract all strings between double quotes
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, raw_string)
    
    # Clean up names: replace spaces and slashes with dashes (as in C code)
    cleaned = []
    for name in matches:
        name = name.replace(' ', '-')
        name = name.replace('/', '-')
        if name:  # Skip empty strings
            cleaned.append(name)
    
    return cleaned


def count_montages(ent_file_path: str | Path) -> int:
    """
    Count the number of montage definitions in an .ent file.
    
    Args:
        ent_file_path: Path to the .ent file
        
    Returns:
        Number of "ChanNames" occurrences found
    """
    ent_file_path = Path(ent_file_path)
    
    with open(ent_file_path, 'rb') as f:
        buffer = f.read()
    
    search_term = b"ChanNames"
    locations = find_all_occurrences(buffer, search_term)
    
    return len(locations)


def find_montage_in_ent(ent_file_path: str | Path, instance: Optional[int] = None) -> Optional[list[str]]:
    """
    Extract channel names from .ent file, mimicking xltek2mef's C implementation.
    
    The .ent file contains embedded montage information with a structure like:
    ChanNames", ("Fp1", "Fp2", ...))
    
    This function searches for "ChanNames" and extracts the channel names that follow.
    
    Args:
        ent_file_path: Path to the .ent file
        instance: Which montage instance to get (1-indexed). 
                  Default: last one (matches C code behavior)
    
    Returns:
        List of channel names, or None if not found
    """
    ent_file_path = Path(ent_file_path)
    
    with open(ent_file_path, 'rb') as f:
        buffer = f.read()
    
    # Search for "ChanNames" occurrences
    search_term = b"ChanNames"
    locations = find_all_occurrences(buffer, search_term)
    
    if not locations:
        print(f"Warning: No 'ChanNames' found in {ent_file_path}")
        return None
    
    # Use last instance by default (matches C code behavior)
    if instance is None:
        instance = len(locations)
    
    if instance < 1 or instance > len(locations):
        print(f"Warning: Requested instance {instance} but only {len(locations)} found")
        return None
    
    location_start = locations[instance - 1]
    
    # Find the opening parenthesis after ChanNames - the format is:
    # ChanNames", ("channel1", "channel2", ...))
    # We want to find where the channel list starts (first open paren after ChanNames)
    open_paren = buffer.find(b"(", location_start)
    
    if open_paren == -1:
        print(f"Warning: Could not find opening paren after ChanNames at position {location_start}")
        return None
    
    # Find end marker "))" - the closing of the channel list
    # Search from the opening paren
    location_end = buffer.find(b"))", open_paren)
    
    if location_end == -1:
        print(f"Warning: Could not find end marker for ChanNames at position {location_start}")
        return None
    
    # Extract the substring containing channel names
    substring = buffer[open_paren:location_end + 2]  # +2 to include ))
    
    # Decode to string for parsing
    try:
        montage_string = substring.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Warning: Failed to decode montage data: {e}")
        return None
    
    return parse_channel_names(montage_string)


def get_ent_file_from_data_dir(data_dir: str | Path) -> Optional[Path]:
    """
    Find the .ent file in a data directory.
    
    Args:
        data_dir: Directory containing XLTEK data files
        
    Returns:
        Path to the .ent file, or None if not found
    """
    data_dir = Path(data_dir)
    
    ent_files = list(data_dir.glob("*.ent"))
    
    if not ent_files:
        print(f"Warning: No .ent file found in {data_dir}")
        return None
    
    if len(ent_files) > 1:
        print(f"Warning: Multiple .ent files found, using first: {ent_files[0]}")
    
    return ent_files[0]


def extract_montage_from_data_dir(data_dir: str | Path, instance: Optional[int] = None) -> Optional[list[str]]:
    """
    Convenience function to extract montage from a data directory.
    
    Args:
        data_dir: Directory containing XLTEK data files
        instance: Which montage instance to get (1-indexed, default: last)
        
    Returns:
        List of channel names, or None if extraction fails
    """
    ent_file = get_ent_file_from_data_dir(data_dir)
    
    if ent_file is None:
        return None
    
    return find_montage_in_ent(ent_file, instance)


if __name__ == "__main__":
    # Test the module standalone
    import sys
    
    if len(sys.argv) > 1:
        ent_path = sys.argv[1]
    else:
        # Try to read from datapath.txt
        datapath_file = Path(__file__).parent / "datapath.txt"
        if datapath_file.exists():
            with open(datapath_file) as f:
                data_dir = f.readline().strip().strip("'\"")
            ent_file = get_ent_file_from_data_dir(data_dir)
            if ent_file:
                ent_path = str(ent_file)
            else:
                print("Usage: python montage_extractor.py <ent_file_or_data_dir>")
                sys.exit(1)
        else:
            print("Usage: python montage_extractor.py <ent_file_or_data_dir>")
            sys.exit(1)
    
    print(f"Extracting montage from: {ent_path}")
    
    num_montages = count_montages(ent_path)
    print(f"Found {num_montages} montage(s) in file")
    
    channel_names = find_montage_in_ent(ent_path)
    
    if channel_names:
        print(f"\nExtracted {len(channel_names)} channel names:")
        for i, name in enumerate(channel_names, 1):
            print(f"  {i:3d}: {name}")
    else:
        print("Failed to extract channel names")
