"""
EXIF data extraction from image files.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from PIL import Image
from PIL.ExifTags import TAGS


class ExifExtractor:
    """Extract EXIF data from image files for clustering."""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.tiff', '.tif'}
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_from_folder(self, folder_path: Path) -> List[Dict[str, Any]]:
        """
        Extract EXIF data from all supported images in folder and subfolders.
        
        Args:
            folder_path: Path to folder to scan
            
        Returns:
            List of dictionaries containing image path and extracted EXIF data
        """
        photo_data = []
        
        for image_path in self._find_images(folder_path):
            try:
                exif_data = self._extract_exif_data(image_path)
                if exif_data:
                    photo_data.append({
                        'path': str(image_path),
                        'exif': exif_data
                    })
                    self.logger.debug(f"Extracted EXIF from: {image_path}")
                else:
                    self.logger.debug(f"No relevant EXIF data in: {image_path}")
            except Exception as e:
                self.logger.warning(f"Failed to extract EXIF from {image_path}: {e}")
        
        return photo_data
    
    def _find_images(self, folder_path: Path) -> List[Path]:
        """Find all supported image files in folder and subfolders."""
        images = []
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                images.append(file_path)
        
        self.logger.info(f"Found {len(images)} image files")
        return images
    
    def _extract_exif_data(self, image_path: Path) -> Dict[str, Any]:
        """
        Extract relevant EXIF data from a single image.
        
        Returns:
            Dictionary with extracted EXIF data or None if no relevant data found
        """
        try:
            with Image.open(image_path) as img:
                exif_dict = img._getexif()
                
                if not exif_dict:
                    return None
                
                # Convert numerical tags to readable names
                exif_data = {}
                for tag_id, value in exif_dict.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    exif_data[tag_name] = value
                
                # Extract the properties we're interested in
                extracted = {}
                
                # Date and time
                datetime_taken = self._extract_datetime(exif_data)
                if datetime_taken:
                    extracted['datetime'] = datetime_taken
                
                # Artist
                if 'Artist' in exif_data:
                    extracted['artist'] = str(exif_data['Artist']).strip()
                
                # Camera/lens model
                model = self._extract_model(exif_data)
                if model:
                    extracted['model'] = model
                
                # Image dimensions for size classification
                try:
                    width = img.width
                    height = img.height
                    if width and height:
                        extracted['width'] = width
                        extracted['height'] = height
                        # Calculate total pixels for size classification
                        total_pixels = width * height
                        extracted['total_pixels'] = total_pixels
                except Exception as e:
                    self.logger.debug(f"Could not extract image dimensions: {e}")
                
                # Only return if we have at least one useful property
                return extracted if extracted else None
                
        except Exception as e:
            self.logger.debug(f"Error reading EXIF from {image_path}: {e}")
            return None
    
    def _extract_datetime(self, exif_data: Dict) -> datetime:
        """Extract datetime from EXIF data."""
        # Try different datetime fields
        datetime_fields = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
        
        for field in datetime_fields:
            if field in exif_data:
                try:
                    datetime_str = str(exif_data[field])
                    # EXIF datetime format: "YYYY:MM:DD HH:MM:SS"
                    return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    continue
        
        return None
    
    def _extract_model(self, exif_data: Dict) -> str:
        """Extract camera/lens model from EXIF data."""
        # Try different model fields
        model_fields = ['Model', 'LensModel', 'Make']
        
        for field in model_fields:
            if field in exif_data:
                model = str(exif_data[field]).strip()
                if model:
                    return model
        
        return None