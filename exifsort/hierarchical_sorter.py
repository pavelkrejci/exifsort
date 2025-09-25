"""
Photo hierarchical sorting based on EXIF properties.
Replaces K-means clustering with strict multi-level hierarchical sorting.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional


class PhotoHierarchicalSorter:
    """Sort photos based on EXIF properties using hierarchical sorting."""
    
    # Size thresholds in pixels (total pixels = width * height)
    SIZE_THRESHOLDS = {
        'L': 12_000_000,  # 12MP+ (e.g., 4000x3000 = 12MP)
        'M': 3_000_000,   # 3MP+ (e.g., 2048x1536 = 3.1MP)
        'S': 0            # Everything else
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def sort_photos(self, photo_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort photos hierarchically based on EXIF properties.
        
        Hierarchy:
        1. Camera model (including unknown)
        2. Artist
        3. Calendar datum (day granularity)
        4. Picture size (L, M, S)
        
        Args:
            photo_data: List of photo data with EXIF information
            
        Returns:
            List of photo data with hierarchical sort assignments
        """
        if not photo_data:
            self.logger.warning("No photo data provided for sorting")
            return photo_data
        
        self.logger.info(f"Sorting {len(photo_data)} photos hierarchically")
        
        # Add hierarchical sort keys to each photo
        for photo in photo_data:
            self._add_sort_keys(photo)
        
        # Group photos by their hierarchical path
        groups = self._group_photos_hierarchically(photo_data)
        
        # Log sorting results
        self._log_sorting_results(groups)
        
        return photo_data
    
    def _add_sort_keys(self, photo: Dict[str, Any]):
        """Add hierarchical sort keys to a photo."""
        exif = photo['exif']
        
        # 1. Camera model (including unknown)
        model = exif.get('model', 'unknown').replace(' ', '_').replace('/', '_')
        photo['sort_model'] = model
        
        # 2. Artist  
        artist = exif.get('artist', 'unknown').replace(' ', '_')
        photo['sort_artist'] = artist
        
        # 3. Calendar datum (day granularity)
        if 'datetime' in exif:
            dt = exif['datetime']
            date_str = dt.strftime("%Y_%m_%d")
            photo['sort_date'] = date_str
        else:
            photo['sort_date'] = 'unknown_date'
        
        # 4. Picture size (L, M, S)
        total_pixels = exif.get('total_pixels', 0)
        size = self._classify_size(total_pixels)
        photo['sort_size'] = size
        
        # Create hierarchical folder path
        folder_parts = [
            f"camera_{photo['sort_model']}",
            f"artist_{photo['sort_artist']}",
            f"date_{photo['sort_date']}",
            f"size_{photo['sort_size']}"
        ]
        
        # Create a hierarchical path
        hierarchical_path = "/".join(folder_parts)
        photo['hierarchical_path'] = hierarchical_path
        
        # For compatibility with existing organizer, set cluster info
        photo['cluster'] = 0  # Not used in hierarchical sorting
        photo['cluster_label'] = "_".join(folder_parts)
    
    def _classify_size(self, total_pixels: int) -> str:
        """Classify image size based on total pixels."""
        if total_pixels >= self.SIZE_THRESHOLDS['L']:
            return 'L'
        elif total_pixels >= self.SIZE_THRESHOLDS['M']:
            return 'M'
        else:
            return 'S'
    
    def _group_photos_hierarchically(self, photo_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group photos by their hierarchical path."""
        groups = {}
        
        for photo in photo_data:
            path = photo['hierarchical_path']
            if path not in groups:
                groups[path] = []
            groups[path].append(photo)
        
        return groups
    
    def _log_sorting_results(self, groups: Dict[str, List[Dict[str, Any]]]):
        """Log information about sorting results."""
        self.logger.info(f"Hierarchical sorting results: {len(groups)} groups created")
        
        # Sort groups by name for consistent logging
        for path in sorted(groups.keys()):
            count = len(groups[path])
            # Simplify path for logging
            simple_path = path.replace('camera_', '').replace('artist_', '').replace('date_', '').replace('size_', '')
            simple_path = simple_path.replace('/', ' > ')
            self.logger.info(f"  {simple_path}: {count} photos")