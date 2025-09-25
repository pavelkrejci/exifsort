"""
Photo clustering using K-means algorithm based on EXIF properties.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


class PhotoClusterer:
    """Cluster photos based on EXIF properties using K-means algorithm."""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def cluster_photos(self, photo_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cluster photos based on their EXIF properties.
        
        Args:
            photo_data: List of photo data with EXIF information
            
        Returns:
            List of photo data with cluster assignments
        """
        if len(photo_data) < self.n_clusters:
            self.logger.warning(f"Number of photos ({len(photo_data)}) is less than clusters ({self.n_clusters})")
            self.n_clusters = min(len(photo_data), 2)
        
        # Extract features for clustering
        features, feature_names = self._extract_features(photo_data)
        
        if features.size == 0:
            self.logger.error("No features could be extracted for clustering")
            # Assign all photos to cluster 0
            for photo in photo_data:
                photo['cluster'] = 0
                photo['cluster_label'] = 'cluster_0'
            return photo_data
        
        self.logger.info(f"Extracted {features.shape[1]} features from {len(photo_data)} photos")
        self.logger.debug(f"Feature names: {feature_names}")
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Generate cluster descriptions
        cluster_descriptions = self._generate_cluster_descriptions(
            photo_data, cluster_labels, features, feature_names
        )
        
        # Assign cluster information to photos
        for i, photo in enumerate(photo_data):
            cluster_id = cluster_labels[i]
            photo['cluster'] = cluster_id
            photo['cluster_label'] = cluster_descriptions[cluster_id]
        
        # Log clustering results
        self._log_clustering_results(photo_data, cluster_descriptions)
        
        return photo_data
    
    def _extract_features(self, photo_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Extract numerical features from EXIF data for clustering."""
        features_list = []
        feature_names = []
        
        # Collect all unique artists and models for encoding
        all_artists = set()
        all_models = set()
        
        for photo in photo_data:
            exif = photo['exif']
            if 'artist' in exif:
                all_artists.add(exif['artist'])
            if 'model' in exif:
                all_models.add(exif['model'])
        
        # Create label encoders
        if all_artists:
            self.label_encoders['artist'] = LabelEncoder()
            self.label_encoders['artist'].fit(list(all_artists))
            feature_names.append('artist')
        
        if all_models:
            self.label_encoders['model'] = LabelEncoder()
            self.label_encoders['model'].fit(list(all_models))
            feature_names.append('model')
        
        # Add datetime features if available
        has_datetime = any('datetime' in photo['exif'] for photo in photo_data)
        if has_datetime:
            feature_names.extend(['year', 'month', 'day', 'hour'])
        
        # Extract features for each photo
        for photo in photo_data:
            exif = photo['exif']
            photo_features = []
            
            # Artist feature
            if 'artist' in feature_names:
                if 'artist' in exif:
                    artist_encoded = self.label_encoders['artist'].transform([exif['artist']])[0]
                    photo_features.append(artist_encoded)
                else:
                    photo_features.append(-1)  # Missing value
            
            # Model feature
            if 'model' in feature_names:
                if 'model' in exif:
                    model_encoded = self.label_encoders['model'].transform([exif['model']])[0]
                    photo_features.append(model_encoded)
                else:
                    photo_features.append(-1)  # Missing value
            
            # Datetime features
            if has_datetime:
                if 'datetime' in exif:
                    dt = exif['datetime']
                    photo_features.extend([dt.year, dt.month, dt.day, dt.hour])
                else:
                    photo_features.extend([0, 0, 0, 0])  # Missing values
            
            features_list.append(photo_features)
        
        if not features_list or not feature_names:
            return np.array([]), []
        
        return np.array(features_list), feature_names
    
    def _generate_cluster_descriptions(
        self, 
        photo_data: List[Dict[str, Any]], 
        cluster_labels: np.ndarray,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[int, str]:
        """Generate descriptive labels for each cluster."""
        cluster_descriptions = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_photos = [photo_data[i] for i in cluster_indices]
            
            # Analyze cluster characteristics
            characteristics = []
            
            # Most common artist
            artists = [photo['exif'].get('artist') for photo in cluster_photos]
            artists = [a for a in artists if a]
            if artists:
                most_common_artist = max(set(artists), key=artists.count)
                if artists.count(most_common_artist) > len(cluster_photos) * 0.3:
                    characteristics.append(f"artist_{most_common_artist.replace(' ', '_')}")
            
            # Most common model
            models = [photo['exif'].get('model') for photo in cluster_photos]
            models = [m for m in models if m]
            if models:
                most_common_model = max(set(models), key=models.count)
                if models.count(most_common_model) > len(cluster_photos) * 0.3:
                    model_clean = most_common_model.replace(' ', '_').replace('/', '_')
                    characteristics.append(f"camera_{model_clean}")
            
            # Time period analysis
            datetimes = [photo['exif'].get('datetime') for photo in cluster_photos]
            datetimes = [dt for dt in datetimes if dt]
            if datetimes:
                years = [dt.year for dt in datetimes]
                most_common_year = max(set(years), key=years.count)
                if years.count(most_common_year) > len(cluster_photos) * 0.5:
                    characteristics.append(f"year_{most_common_year}")
            
            # Generate cluster label
            if characteristics:
                cluster_label = "_".join(characteristics[:2])  # Max 2 characteristics
            else:
                cluster_label = f"cluster_{cluster_id}"
            
            cluster_descriptions[cluster_id] = cluster_label
        
        return cluster_descriptions
    
    def _log_clustering_results(self, photo_data: List[Dict[str, Any]], cluster_descriptions: Dict[int, str]):
        """Log information about clustering results."""
        cluster_counts = {}
        for photo in photo_data:
            cluster_id = photo['cluster']
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        self.logger.info("Clustering results:")
        for cluster_id, count in sorted(cluster_counts.items()):
            label = cluster_descriptions[cluster_id]
            self.logger.info(f"  {label}: {count} photos")