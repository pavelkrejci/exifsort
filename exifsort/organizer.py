"""
Photo organization module - creates folder structure and symlinks.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any


class PhotoOrganizer:
    """Organize photos into folders with symlinks based on cluster assignments."""
    
    def __init__(self, output_path: Path, dry_run: bool = False):
        self.output_path = Path(output_path)
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
    
    def organize_photos(self, clustered_data: List[Dict[str, Any]]):
        """
        Create folder structure and symlinks for clustered photos.
        
        Args:
            clustered_data: List of photo data with cluster assignments
        """
        if not self.dry_run:
            # Create output directory if it doesn't exist
            self.output_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created output directory: {self.output_path}")
        
        # Group photos by cluster
        clusters = {}
        for photo in clustered_data:
            cluster_label = photo['cluster_label']
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append(photo)
        
        # Create folders and symlinks for each cluster
        for cluster_label, photos in clusters.items():
            self._create_cluster_folder(cluster_label, photos)
    
    def _create_cluster_folder(self, cluster_label: str, photos: List[Dict[str, Any]]):
        """Create a folder for a cluster and symlink all photos to it."""
        cluster_path = self.output_path / cluster_label
        
        if not self.dry_run:
            cluster_path.mkdir(exist_ok=True)
        
        self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Created cluster folder: {cluster_path}")
        
        # Create symlinks for each photo
        symlink_count = 0
        for photo in photos:
            source_path = Path(photo['path'])
            
            # Generate unique filename if there are conflicts
            target_filename = self._get_unique_filename(cluster_path, source_path.name)
            target_path = cluster_path / target_filename
            
            try:
                if not self.dry_run:
                    # Create relative symlink to make the structure more portable
                    rel_source = os.path.relpath(source_path, cluster_path)
                    target_path.symlink_to(rel_source)
                
                symlink_count += 1
                self.logger.debug(f"{'[DRY RUN] ' if self.dry_run else ''}Created symlink: {target_path} -> {source_path}")
                
            except OSError as e:
                self.logger.warning(f"Failed to create symlink {target_path}: {e}")
        
        self.logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Created {symlink_count} symlinks in {cluster_label}")
    
    def _get_unique_filename(self, cluster_path: Path, original_filename: str) -> str:
        """
        Generate a unique filename in the cluster directory to avoid conflicts.
        
        Args:
            cluster_path: Path to the cluster directory
            original_filename: Original filename
            
        Returns:
            Unique filename that doesn't conflict with existing files
        """
        if self.dry_run:
            return original_filename
        
        target_path = cluster_path / original_filename
        
        if not target_path.exists():
            return original_filename
        
        # If file exists, add a number suffix
        name_stem = Path(original_filename).stem
        name_suffix = Path(original_filename).suffix
        counter = 1
        
        while True:
            new_filename = f"{name_stem}_{counter}{name_suffix}"
            target_path = cluster_path / new_filename
            if not target_path.exists():
                return new_filename
            counter += 1