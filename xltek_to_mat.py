"""
xltek_to_mat.py - Convert XLTEK EEG data to MATLAB v7.3 .mat format

This script reads XLTEK .erd files and exports the data to MATLAB v7.3 format,
including:
- EEG data (channels x samples)
- Channel names (from .ent file)
- Sample rate
- Start time
- Metadata

Usage:
    python xltek_to_mat.py /path/to/xltek/data output.mat
    python xltek_to_mat.py /path/to/xltek/data output.mat -q
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import hdf5storage
    HAS_HDF5STORAGE = True
except ImportError:
    HAS_HDF5STORAGE = False
    print("Warning: hdf5storage not installed. Run: pip install hdf5storage")

from erd_reader import read_erd_header, read_erd_samples, get_erd_files_from_data_dir
from snc_reader import read_snc_file, get_snc_file_from_data_dir, get_sample_timestamps
from montage_extractor import extract_montage_from_data_dir


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert XLTEK EEG data to MATLAB v7.3 .mat format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python xltek_to_mat.py /path/to/xltek/data output.mat
  python xltek_to_mat.py /path/to/xltek/data output.mat -q
        '''
    )
    parser.add_argument('input_dir', type=str,
                        help='Directory containing XLTEK files (.erd, .ent, .snc)')
    parser.add_argument('output', type=str,
                        help='Output .mat file path')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress progress messages')
    
    return parser.parse_args()


def convert_xltek_to_mat(data_dir: Path, output_path: Path, 
                         include_timestamps: bool = True,
                         verbose: bool = True) -> dict:
    """
    Convert XLTEK data directory to HDF5 file.
    
    Args:
        data_dir: Directory containing XLTEK files (.erd, .ent, .snc, etc.)
        output_path: Path for output HDF5 file
        include_timestamps: Whether to include full timestamp vector (can be large)
        verbose: Print progress messages
        
    Returns:
        Dictionary with summary information
    """
    if not HAS_HDF5STORAGE:
        raise ImportError("hdf5storage is required for .mat export. Install with: pip install hdf5storage")
    
    info = {}
    
    # Get ERD files
    erd_files = get_erd_files_from_data_dir(data_dir)
    if not erd_files:
        raise FileNotFoundError(f"No .erd files found in {data_dir}")
    
    if verbose:
        print(f"Found {len(erd_files)} ERD files")
    
    # Read header from first ERD file
    header = read_erd_header(erd_files[0])
    sample_rate = header.sample_freq
    num_channels = header.num_recorded_chans
    
    if verbose:
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Channels: {num_channels}")
    
    # Extract channel names from ENT file
    channel_names = extract_montage_from_data_dir(data_dir)
    if channel_names:
        # Trim to number of recorded channels
        channel_names = channel_names[:num_channels]
        if verbose:
            print(f"Extracted {len(channel_names)} channel names from .ent file")
    else:
        # Fallback to generic names
        channel_names = [f"Ch{i+1}" for i in range(num_channels)]
        if verbose:
            print("Using generic channel names")
    
    # Get sync/timestamp info
    snc_file = get_snc_file_from_data_dir(data_dir)
    if snc_file:
        sync_entries = read_snc_file(snc_file)
        if sync_entries:
            start_time = sync_entries[0].datetime
            start_time_usec = sync_entries[0].unix_time_usec
            if verbose:
                print(f"Start time: {start_time}")
        else:
            start_time = header.creation_time
            start_time_usec = int(start_time.timestamp() * 1_000_000) if start_time else 0
    else:
        sync_entries = []
        start_time = header.creation_time
        start_time_usec = int(start_time.timestamp() * 1_000_000) if start_time else 0
    
    # Read all ERD files and concatenate
    if verbose:
        print("\nReading EEG data...")
    
    all_data = []
    total_samples = 0
    
    for i, erd_file in enumerate(erd_files):
        if verbose:
            print(f"  Reading {erd_file.name}...")
        
        file_header = read_erd_header(erd_file)
        samples = read_erd_samples(erd_file, file_header)
        all_data.append(samples)
        total_samples += samples.shape[1]
        
        if verbose:
            print(f"    {samples.shape[1]} samples")
    
    # Concatenate all data
    if verbose:
        print(f"\nConcatenating {len(all_data)} files...")
    
    eeg_data = np.concatenate(all_data, axis=1)
    
    if verbose:
        print(f"Total data shape: {eeg_data.shape}")
        print(f"Duration: {eeg_data.shape[1] / sample_rate:.1f} seconds")
    
    # Apply voltage conversion factor
    # Based on xltek2mef: (8711./(2.**21.-0.5))*2.**discardbits
    discard_bits = header.discard_bits
    voltage_factor = (8711.0 / (2**21 - 0.5)) * (2**discard_bits)
    eeg_data_uv = eeg_data.astype(np.float64) * voltage_factor
    
    if verbose:
        print(f"Applied voltage conversion factor: {voltage_factor:.6f}")
        print(f"Data range: [{eeg_data_uv.min():.1f}, {eeg_data_uv.max():.1f}] µV")
    
    # Create MATLAB v7.3 .mat file using hdf5storage
    if verbose:
        print(f"\nWriting to {output_path}...")
    
    # Build data dictionary for MATLAB
    mat_data = {
        'eeg_data': eeg_data_uv,
        'channel_names': np.array(channel_names, dtype=object),
        'sample_rate': sample_rate,
        'start_time_usec': start_time_usec,
        'num_channels': num_channels,
        'num_samples': eeg_data.shape[1],
        'duration_seconds': eeg_data.shape[1] / sample_rate,
        'voltage_unit': 'microvolts',
        'source_files': np.array([f.name for f in erd_files], dtype=object),
        'created_by': 'xltek_to_mat.py',
        'conversion_date': datetime.now().isoformat(),
    }
    
    # Add timestamps if available
    if include_timestamps and sync_entries:
        timestamps = get_sample_timestamps(sync_entries, sample_rate, total_samples)
        mat_data['timestamps_usec'] = timestamps
    
    # Add start time as ISO string if available
    if start_time:
        mat_data['start_time_iso'] = start_time.isoformat()
    
    # Save as MATLAB v7.3 format (oned_as='row' for MATLAB convention)
    hdf5storage.savemat(str(output_path), mat_data, format='7.3', oned_as='row')
    
    info = {
        'num_channels': num_channels,
        'num_samples': eeg_data.shape[1],
        'sample_rate': sample_rate,
        'duration_sec': eeg_data.shape[1] / sample_rate,
        'start_time': start_time,
        'output_file': str(output_path)
    }
    
    if verbose:
        print("\n✓ Conversion complete!")
        print(f"  Output: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return info


def main():
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet
    
    if verbose:
        print("="*60)
        print("XLTEK to HDF5 Converter")
        print("="*60)
    
    if not HAS_HDF5STORAGE:
        print("\nError: hdf5storage is required. Install with: pip install hdf5storage")
        return 1
    
    # Get paths from arguments
    data_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    if not data_dir.exists():
        print(f"Error: Directory does not exist: {data_dir}")
        return 1
    
    if verbose:
        print(f"\nData directory: {data_dir}")
        print(f"Output file: {output_path}")
    
    # Convert
    try:
        info = convert_xltek_to_mat(
            data_dir, 
            output_path, 
            include_timestamps=True,
            verbose=verbose
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"Channels: {info['num_channels']}")
            print(f"Samples: {info['num_samples']}")
            print(f"Duration: {info['duration_sec']:.1f} seconds ({info['duration_sec']/60:.1f} minutes)")
            print(f"Sample rate: {info['sample_rate']} Hz")
            print(f"Start time: {info['start_time']}")
            print(f"Output file: {info['output_file']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
