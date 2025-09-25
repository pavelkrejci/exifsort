# exifsort

A photo hierarchical sorting Linux CLI Python-based tool that organizes photos based on EXIF metadata using multi-level hierarchical sorting.

## Features

- **Recursive photo discovery**: Searches folders and subfolders for JPEG/TIFF images
- **EXIF data extraction**: Extracts date/time, artist, camera/lens model, and image dimensions
- **Hierarchical sorting**: Organizes photos using strict multi-level hierarchy based on EXIF properties
- **Four-level hierarchy**: Camera model → Artist → Date (day granularity) → Picture size (L/M/S)
- **Size classification**: Automatically classifies photos as Large (12MP+), Medium (3MP+), or Small (<3MP)
- **Unknown value handling**: Gracefully handles missing EXIF data with "unknown" placeholders
- **Organized folder structure**: Creates descriptive folder names based on hierarchical sorting
- **Symlink organization**: Creates symlinks to original photos (preserving original file structure)
- **Relative paths**: Uses relative symlinks for portability

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.7+
- Pillow (for image processing)
- click (for CLI interface)

## Usage

```bash
exifsort SOURCE_FOLDER OUTPUT_FOLDER [OPTIONS]
```

### Arguments

- `SOURCE_FOLDER`: Path to folder containing photos to sort
- `OUTPUT_FOLDER`: Path where hierarchical folder structure will be created

### Options

- `-v, --verbose`: Enable verbose logging
- `--dry-run`: Show what would be done without creating files
- `--help`: Show help message

### Examples

```bash
# Basic usage - hierarchical sorting by camera model > artist > date > size
exifsort /path/to/photos /path/to/output

# Dry run to see what would happen
exifsort /path/to/photos /path/to/output --dry-run --verbose

# Verbose output to see detailed sorting process
exifsort /path/to/photos /path/to/output --verbose
```

## How It Works

1. **Discovery**: Recursively scans the source folder for supported image files (.jpg, .jpeg, .tiff, .tif)
2. **EXIF Extraction**: Extracts relevant EXIF properties:
   - Date and time (DateTime, DateTimeOriginal, DateTimeDigitized)
   - Artist information
   - Camera/lens model (Model, LensModel, Make)
   - Image dimensions (width, height) for size classification
3. **Hierarchical Sorting**: Organizes photos using strict 4-level hierarchy:
   - Level 1: Camera model (including "unknown" for missing data)
   - Level 2: Artist (including "unknown" for missing data)
   - Level 3: Calendar date (day granularity: YYYY_MM_DD)
   - Level 4: Picture size (L: 12MP+, M: 3MP+, S: <3MP)
4. **Organization**: Creates folders with hierarchical names and creates symlinks to original photos

## Output Structure

The tool creates folders with hierarchical names following this pattern:
`camera_{MODEL}_artist_{ARTIST}_date_{YYYY_MM_DD}_size_{L|M|S}`

Examples:
- `camera_Canon_EOS_R5_artist_John_Doe_date_2023_08_15_size_L`
- `camera_Nikon_D850_artist_Jane_Smith_date_2023_08_16_size_M`
- `camera_Sony_A7_III_artist_Bob_Johnson_date_2022_12_25_size_S`
- `camera_unknown_artist_unknown_date_2024_01_01_size_S` (for missing EXIF data)

**Size Classification:**
- **L (Large)**: 12MP+ (12,000,000+ pixels)
- **M (Medium)**: 3MP+ (3,000,000+ pixels)  
- **S (Small)**: Less than 3MP

Each folder contains symlinks to the original photo files, preserving the original file structure while creating an organized hierarchical view.

## Supported Formats

- JPEG (.jpg, .jpeg)
- TIFF (.tiff, .tif)

## License

This project is open source and available under the MIT License.