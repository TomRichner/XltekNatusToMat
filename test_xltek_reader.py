"""
test_xltek_reader.py - Test script for reading XLTEK data

This script tests reading XLTEK EEG data using:
1. Custom montage extraction from .ent files (to get actual channel names)
2. XltekDataReader for loading the EEG data

The montage extraction approach bypasses the hardcoded channel mappings in
XltekDataReader's conversion.json that caused issues with non-standard montages.
"""

import sys
from pathlib import Path

# Add XltekDataReader to path
XLTEK_READER_PATH = Path("/Users/richner.thomas/Desktop/local_code/XltekDataReader")
sys.path.insert(0, str(XLTEK_READER_PATH))

from montage_extractor import (
    extract_montage_from_data_dir,
    count_montages,
    get_ent_file_from_data_dir,
)


def get_data_path() -> Path:
    """Read data path from datapath.txt"""
    datapath_file = Path(__file__).parent / "datapath.txt"
    
    if not datapath_file.exists():
        raise FileNotFoundError(f"datapath.txt not found at {datapath_file}")
    
    with open(datapath_file) as f:
        # Read first line and strip quotes
        data_path = f.readline().strip().strip("'\"")
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    
    return data_path


def list_data_files(data_dir: Path) -> dict[str, list[Path]]:
    """List all XLTEK files in the data directory, grouped by extension."""
    extensions = ['eeg', 'erd', 'ent', 'etc', 'snc', 'stc', 'vtc', 'epo']
    
    files = {}
    for ext in extensions:
        ext_files = list(data_dir.glob(f"*.{ext}"))
        if ext_files:
            files[ext] = sorted(ext_files)
    
    return files


def extract_and_display_montage(data_dir: Path) -> list[str] | None:
    """Extract and display the montage from the data directory."""
    ent_file = get_ent_file_from_data_dir(data_dir)
    
    if ent_file is None:
        print("ERROR: No .ent file found!")
        return None
    
    print(f"\n{'='*60}")
    print("MONTAGE EXTRACTION")
    print(f"{'='*60}")
    print(f"ENT file: {ent_file.name}")
    
    num_montages = count_montages(ent_file)
    print(f"Number of montages found: {num_montages}")
    
    # Extract the last montage (most recent)
    channel_names = extract_montage_from_data_dir(data_dir)
    
    if channel_names:
        print(f"\nExtracted {len(channel_names)} channels:")
        
        # Display in columns for readability
        n_cols = 4
        for i in range(0, len(channel_names), n_cols):
            row = channel_names[i:i+n_cols]
            row_str = "  ".join(f"{i+j+1:3d}: {name:<15}" for j, name in enumerate(row))
            print(f"  {row_str}")
    else:
        print("ERROR: Failed to extract channel names!")
    
    return channel_names


def try_xltek_loader(data_dir: Path, extracted_channels: list[str] | None = None):
    """Try to load data using XltekDataReader."""
    print(f"\n{'='*60}")
    print("XLTEK DATA READER TEST")
    print(f"{'='*60}")
    
    # Change to XltekDataReader directory so relative paths work
    import os
    original_dir = os.getcwd()
    os.chdir(XLTEK_READER_PATH)
    
    try:
        from file_loader.xltek_loader import XltekLoader
        
        print(f"Loading from: {data_dir.name}")
        loader = XltekLoader(str(data_dir))
        
        print("\nLoaders initialized:")
        for name in loader.loaders_dict.keys():
            print(f"  - {name}")
        
        print("\nReading files...")
        loader.read()
        
        # Get some basic info
        eeg_loader = loader.loaders_dict.get('eeg_loader')
        erd_loader = loader.loaders_dict.get('erd_loader_0')
        
        if eeg_loader:
            print("\nEEG file metadata:")
            if hasattr(eeg_loader, 'data') and 'study_info' in eeg_loader.data:
                study_info = eeg_loader.data['study_info']
                for key, val in list(study_info.items())[:10]:
                    print(f"  {key}: {val}")
        
        if erd_loader:
            print("\nERD file info:")
            if hasattr(erd_loader, 'data') and 'raw_data_file_header' in erd_loader.data:
                header = erd_loader.data['raw_data_file_header']
                print(f"  Headbox type: {header.get('headbox_type', 'unknown')}")
                print(f"  Num channels: {header.get('num_channels', 'unknown')}")
                print(f"  Sample freq: {header.get('sample_freq', 'unknown')}")
            
            if hasattr(erd_loader, 'channel_names'):
                print(f"\n  XltekDataReader channel names ({len(erd_loader.channel_names)} channels):")
                # Show first 10 channels
                for i, name in enumerate(erd_loader.channel_names[:10]):
                    print(f"    {i+1}: {name}")
                if len(erd_loader.channel_names) > 10:
                    print(f"    ... and {len(erd_loader.channel_names) - 10} more")
        
        # Compare with extracted channels
        if extracted_channels and erd_loader and hasattr(erd_loader, 'channel_names'):
            print(f"\n{'='*60}")
            print("CHANNEL NAME COMPARISON")
            print(f"{'='*60}")
            print(f"Extracted from .ent: {len(extracted_channels)} channels")
            print(f"From XltekDataReader: {len(erd_loader.channel_names)} channels")
            
            if extracted_channels[:5] != erd_loader.channel_names[:5]:
                print("\n⚠️  WARNING: Channel names differ!")
                print("\nFirst 5 channels comparison:")
                print(f"  {'ENT extracted':<20} {'XltekDataReader':<20}")
                print(f"  {'-'*20} {'-'*20}")
                for i in range(min(5, len(extracted_channels), len(erd_loader.channel_names))):
                    ent_name = extracted_channels[i] if i < len(extracted_channels) else "N/A"
                    xltek_name = erd_loader.channel_names[i] if i < len(erd_loader.channel_names) else "N/A"
                    match = "✓" if ent_name == xltek_name else "✗"
                    print(f"  {ent_name:<20} {xltek_name:<20} {match}")
                
                print("\n→ Use extracted channel names for accurate labeling!")
        
        # Try full load
        print("\nAttempting full data load (combine_files)...")
        try:
            loader.validate()
            result = loader.combine_files()
            
            print("\n✓ Data loaded successfully!")
            print(f"  EEG data shape: {result['EEGData'].shape}")
            print(f"  Channels: {len(result['ChannelNames'])} (including SampleStamp)")
            print(f"  Notes: {len(result['Notes'])} entries")
            
        except Exception as e:
            print(f"\n✗ Full load failed: {type(e).__name__}: {e}")
            print("  (This is expected if headbox type is not in conversion.json)")
            
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        os.chdir(original_dir)


def main():
    print("="*60)
    print("XLTEK DATA READER TEST")
    print("="*60)
    
    # Get data path
    try:
        data_dir = get_data_path()
        print(f"Data directory: {data_dir}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    
    # List files
    print(f"\n{'='*60}")
    print("DATA FILES")
    print(f"{'='*60}")
    files = list_data_files(data_dir)
    
    for ext, file_list in files.items():
        print(f"\n.{ext} files ({len(file_list)}):")
        for f in file_list:
            print(f"  - {f.name}")
    
    # Extract montage
    extracted_channels = extract_and_display_montage(data_dir)
    
    # Try XltekDataReader
    try_xltek_loader(data_dir, extracted_channels)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if extracted_channels:
        print(f"✓ Successfully extracted {len(extracted_channels)} channel names from .ent file")
        print("  These are the actual channel names used in the recording.")
    else:
        print("✗ Failed to extract channel names from .ent file")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
