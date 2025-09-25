#!/usr/bin/env python3
"""
Command line interface for ExifSort photo clustering tool.
"""

import os
import sys
import logging
from pathlib import Path
import click

from .exif_extractor import ExifExtractor
from .hierarchical_sorter import PhotoHierarchicalSorter
from .organizer import PhotoOrganizer


@click.command()
@click.argument('source_folder', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_folder', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--dry-run', is_flag=True, help='Show what would be done without creating files')
def main(source_folder, output_folder, verbose, dry_run):
    """
    Sort photos hierarchically based on EXIF data and organize them into folders with symlinks.
    
    SOURCE_FOLDER: Path to folder containing photos to sort
    OUTPUT_FOLDER: Path where hierarchical folder structure will be created
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    logger = logging.getLogger(__name__)
    
    try:
        source_path = Path(source_folder).resolve()
        output_path = Path(output_folder).resolve()
        
        logger.info(f"Scanning photos in: {source_path}")
        logger.info(f"Output folder: {output_path}")
        
        if dry_run:
            logger.info("DRY RUN MODE - No files will be created")
        
        # Step 1: Extract EXIF data from all photos
        extractor = ExifExtractor()
        photo_data = extractor.extract_from_folder(source_path)
        
        if not photo_data:
            logger.error("No photos with EXIF data found in the source folder")
            sys.exit(1)
        
        logger.info(f"Found {len(photo_data)} photos with EXIF data")
        
        # Step 2: Sort photos hierarchically based on EXIF properties
        sorter = PhotoHierarchicalSorter()
        sorted_data = sorter.sort_photos(photo_data)
        
        # Step 3: Organize photos into folders with symlinks
        organizer = PhotoOrganizer(output_path, dry_run=dry_run)
        organizer.organize_photos(sorted_data)
        
        logger.info("Photo hierarchical sorting and organization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()