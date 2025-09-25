# exifsort

A photo clustering Linux CLI Python-based tool that organizes photos based on EXIF metadata using K-means clustering.

## Features

- **Recursive photo discovery**: Searches folders and subfolders for JPEG/TIFF images
- **EXIF data extraction**: Extracts date/time, artist, and camera/lens model information
- **K-means clustering**: Groups photos based on EXIF properties using machine learning
- **Progressive clustering**: Automatically determines optimal number of clusters using silhouette analysis
- **Day-level granularity**: Groups photos with fine-grained date precision (day-level instead of year-level)
- **Organized folder structure**: Creates folders with descriptive labels based on cluster characteristics
- **Symlink organization**: Creates symlinks to original photos (each photo linked to exactly one folder)
- **Relative paths**: Uses relative symlinks for portability

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.7+
- Pillow (for image processing)
- scikit-learn (for K-means clustering)
- numpy (for numerical operations)
- click (for CLI interface)

## Usage

```bash
exifsort SOURCE_FOLDER OUTPUT_FOLDER [OPTIONS]
```

### Arguments

- `SOURCE_FOLDER`: Path to folder containing photos to cluster
- `OUTPUT_FOLDER`: Path where clustered folder structure will be created

### Options

- `-c, --clusters INTEGER`: Number of clusters for K-means (default: auto-determine)
- `--max-clusters INTEGER`: Maximum clusters when auto-determining (default: 10)
- `-v, --verbose`: Enable verbose logging
- `--dry-run`: Show what would be done without creating files
- `--help`: Show help message

### Examples

```bash
# Basic usage - automatically determine optimal number of clusters
exifsort /path/to/photos /path/to/output

# Specify fixed number of clusters
exifsort /path/to/photos /path/to/output --clusters 3

# Set maximum clusters for auto-determination
exifsort /path/to/photos /path/to/output --max-clusters 8

# Dry run to see what would happen
exifsort /path/to/photos /path/to/output --dry-run --verbose

# Verbose output
exifsort /path/to/photos /path/to/output --verbose
```

## How It Works

1. **Discovery**: Recursively scans the source folder for supported image files (.jpg, .jpeg, .tiff, .tif)
2. **EXIF Extraction**: Extracts relevant EXIF properties:
   - Date and time (DateTime, DateTimeOriginal, DateTimeDigitized)
   - Artist information
   - Camera/lens model (Model, LensModel, Make)
3. **Feature Engineering**: Converts EXIF data into numerical features for clustering:
   - Categorical encoding for artists and models
   - Temporal features (days since epoch, hour) from timestamps for day-level granularity
4. **Clustering**: Applies K-means clustering algorithm to group photos with similar characteristics
   - Auto-determines optimal number of clusters using silhouette analysis (default)
   - Or uses fixed number of clusters if specified
5. **Organization**: Creates folders with descriptive names based on cluster characteristics and creates symlinks to original photos

## Output Structure

The tool creates folders with names like:
- `artist_John_Doe_camera_Canon_EOS_R5`
- `artist_Jane_Smith_date_2023_08_15` (day-level granularity)
- `month_2023_07_camera_Nikon_D850` (month-level fallback)
- `cluster_0` (fallback for clusters without clear characteristics)

Each folder contains symlinks to the original photo files, preserving the original file structure while creating an organized view.

## Supported Formats

- JPEG (.jpg, .jpeg)
- TIFF (.tiff, .tif)

## License

This project is open source and available under the MIT License.