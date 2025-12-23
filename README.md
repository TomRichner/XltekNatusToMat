# xltek_read

Python tools to read XLTEK/Natus EEG data and convert to MATLAB v7.3 .mat format.

## Features

- **Extract channel names** from `.ent` files (actual montage, not hardcoded defaults)
- **Read EEG data** from `.erd` files with delta decompression
- **Extract timestamps** from `.snc` sync files
- **Export to MATLAB v7.3** `.mat` format with proper voltage scaling

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy h5py hdf5storage
```

## Usage

### Convert XLTEK to MATLAB

```bash
python xltek_to_mat.py /path/to/xltek/data output.mat
```

Options:
- `-q, --quiet` - Suppress progress messages

### Example

```bash
python xltek_to_mat.py '/path/to/Patient_uuid-guid/' seizure_data.mat
```

### Output Format

The `.mat` file contains:

| Variable | Description |
|----------|-------------|
| `eeg_data` | EEG data in µV (channels × samples) |
| `channel_names` | Cell array of channel names |
| `sample_rate` | Sampling rate in Hz |
| `timestamps_usec` | Unix timestamps for each sample (µs) |
| `start_time_usec` | Recording start time (µs since epoch) |
| `start_time_iso` | Start time as ISO string |
| `num_channels` | Number of channels |
| `num_samples` | Number of samples |
| `duration_seconds` | Recording duration |

### Reading in MATLAB

```matlab
data = load('output.mat');
plot(data.eeg_data(1, 1:1000));  % Plot first 1000 samples of channel 1
xlabel('Sample'); ylabel('µV');
title(data.channel_names{1});
```

## Modules

| File | Description |
|------|-------------|
| `xltek_to_mat.py` | Main conversion script |
| `erd_reader.py` | ERD file parser with delta decompression |
| `snc_reader.py` | SNC sync file parser for timestamps |
| `montage_extractor.py` | Extract channel names from ENT files |

## XLTEK File Types

| Extension | Description |
|-----------|-------------|
| `.erd` | Raw EEG data (delta-compressed) |
| `.ent` | Notes and montage information |
| `.snc` | Sync entries (sample-to-time mapping) |
| `.etc` | Table of contents for ERD files |
| `.eeg` | Patient/study metadata |
| `.stc`, `.vtc` | Video sync information |

## Credits

Based on analysis of [xltek2mef](https://github.com/MayoNeurologyAI/xltek2mef) C source code written by Matt Stead and Dan Crepeau for understanding the XLTEK file format.