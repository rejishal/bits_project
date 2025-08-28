# src/models/customer_segmentation.py
"""Customer segmentation module using clustering algorithms"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Any
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import model_logger
from ..utils.config import config

class CustomerSegmentation:
    """Handles customer segmentation using clustering algorithms"""
    
    def __init__(self, algorithm: str = 'kmeans'):
        """
        Initialize the segmentation model
        
        Args:
            algorithm: Clustering algorithm to use ('kmeans', 'dbscan', 'hierarchical')
        """
        self.algorithm = algorithm
        self.model = None
        self.optimal_clusters = None
        self.segment_profiles = None
        self.cluster_centers = None
        model_logger.info(f"CustomerSegmentation initialized with {algorithm} algorithm")
    
    def find_optimal_clusters(self, X: np.ndarray, 
                            k_range: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple metrics
        
        Args:
            X: Preprocessed feature matrix
            k_range: Range of cluster numbers to try (min, max)
            
        Returns:
            Dictionary with evaluation metrics for each k
        """
        if k_range is None:
            k_range = tuple(config.get('segmentation.n_clusters_range', [2, 10]))
        
        model_logger.info(f"Finding optimal clusters in range {k_range}")
        
        results = {
            'n_clusters': [],
            'inertia': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        for k in range(k_range[0], k_range[1] + 1):
            model_logger.info(f"Evaluating k={k}")
            
            # Fit model
            if self.algorithm == 'kmeans':
                model = KMeans(
                    n_clusters=k, 
                    random_state=config.get('segmentation.random_state', 42),
                    n_init=config.get('segmentation.n_init', 10)
                )
                labels = model.fit_predict(X)
                inertia = model.inertia_
            else:
                # For other algorithms, use a default metric
                inertia = None
                labels = None
            
            if labels is not None:
                # Calculate metrics
                results['n_clusters'].append(k)
                results['inertia'].append(inertia)
                results['silhouette'].append(silhouette_score(X, labels))
                results['davies_bouldin'].append(davies_bouldin_score(X, labels))
                results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
        
        # Determine optimal clusters
        self._determine_optimal_clusters(results)
        
        return results
    
    def _determine_optimal_clusters(self, results: Dict[str, List]):
        """Determine optimal number of clusters based on metrics"""
        # Use silhouette score as primary metric
        silhouette_scores = results['silhouette']
        best_idx = np.argmax(silhouette_scores)
        self.optimal_clusters = results['n_clusters'][best_idx]
        
        model_logger.info(f"Optimal clusters determined: {self.optimal_clusters}")
    
    def fit(self, X: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Fit the clustering model
        
        Args:
            X: Preprocessed feature matrix
            n_clusters: Number of clusters (uses optimal if None)
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.optimal_clusters
            if n_clusters is None:
                raise ValueError("Number of clusters not specified. Run find_optimal_clusters first.")
        
        model_logger.info(f"Fitting {self.algorithm} with {n_clusters} clusters")
        
        if self.algorithm == 'kmeans':
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=config.get('segmentation.random_state', 42),
                n_init=config.get('segmentation.n_init', 10),
                max_iter=config.get('segmentation.max_iter', 300)
            )
        elif self.algorithm == 'dbscan':
            eps = getattr(self, 'eps', 0.5)
            min_samples = getattr(self, 'min_samples', 5)
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        elif self.algorithm == 'hierarchical':
            self.model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        labels = self.model.fit_predict(X)
        
        # Store cluster centers for KMeans
        if self.algorithm == 'kmeans':
            self.cluster_centers = self.model.cluster_centers_
        
        # Calculate final metrics only if we have enough clusters
        try:
            # For DBSCAN, exclude noise points (-1) when calculating silhouette score
            if self.algorithm == 'dbscan':
                mask = labels != -1
                unique_clusters = len(set(labels[mask]))
                if unique_clusters > 1:
                    silhouette = silhouette_score(X[mask], labels[mask])
                    model_logger.info(f"DBSCAN clustering complete. Found {unique_clusters} clusters, {(labels == -1).sum()} noise points. Silhouette score: {silhouette:.3f}")
                else:
                    model_logger.info(f"DBSCAN clustering complete. Found {unique_clusters} cluster(s), {(labels == -1).sum()} noise points. Cannot calculate silhouette score.")
            else:
                # For other algorithms
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(X, labels)
                    model_logger.info(f"Clustering complete. Silhouette score: {silhouette:.3f}")
                else:
                    model_logger.info(f"Clustering complete. Only 1 cluster found.")
        except Exception as e:
            model_logger.warning(f"Could not calculate silhouette score: {e}")
        
        return labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            # For algorithms without predict, use fit_predict
            return self.model.fit_predict(X)
    
    def create_segment_profiles(self, data: pd.DataFrame, 
                              cluster_labels: np.ndarray,
                              feature_columns: List[str]) -> pd.DataFrame:
        """
        Create detailed profiles for each segment
        
        Args:
            data: Original dataframe with features
            cluster_labels: Cluster assignments
            feature_columns: List of feature columns to profile
            
        Returns:
            DataFrame with segment profiles
        """
        model_logger.info("Creating segment profiles")
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        
        profiles = []
        
        for cluster in sorted(np.unique(cluster_labels)):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster]
            
            profile = {
                'cluster': cluster,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100
            }
            
            # Add statistics for numerical features
            for col in feature_columns:
                if col in data.columns and data[col].dtype in ['float64', 'int64']:
                    profile[f'{col}_mean'] = cluster_data[col].mean()
                    profile[f'{col}_std'] = cluster_data[col].std()
                    profile[f'{col}_median'] = cluster_data[col].median()
            
            profiles.append(profile)
        
        self.segment_profiles = pd.DataFrame(profiles)
        model_logger.info(f"Created profiles for {len(profiles)} segments")
        
        return self.segment_profiles
    
    def get_segment_names(self, profiles: pd.DataFrame) -> Dict[int, str]:
        """
        Generate business-meaningful names for segments
        
        Args:
            profiles: Segment profiles dataframe
            
        Returns:
            Dictionary mapping cluster IDs to names
        """
        segment_names = {}
        
        for _, row in profiles.iterrows():
            cluster = row['cluster']
            
            # Simple naming logic based on key metrics
            # This should be customized based on business requirements
            if 'income_mean' in row and 'credit_score_mean' in row:
                if row['income_mean'] > 100000 and row['credit_score_mean'] > 750:
                    name = "Premium High-Value"
                elif row['income_mean'] < 40000 and row['credit_score_mean'] < 600:
                    name = "Budget Conscious"
                elif row['age_mean'] < 30:
                    name = "Young Professionals"
                else:
                    name = f"Standard Segment {cluster}"
            else:
                name = f"Segment {cluster}"
            
            segment_names[cluster] = name
        
        return segment_names
    
    def visualize_clusters(self, X: np.ndarray, labels: np.ndarray, 
                          title: str = "Customer Segments") -> plt.Figure:
        """
        Visualize clusters using PCA
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        model_logger.info("Creating cluster visualization")
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=[colors[i]], label=f'Cluster {label}',
                      alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plot centroids if available
        if self.cluster_centers is not None:
            centroids_pca = pca.transform(self.cluster_centers)
            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                      marker='*', s=300, c='red', edgecolor='black',
                      linewidth=2, label='Centroids')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def calculate_stability(self, X: np.ndarray, n_bootstrap: int = 10) -> float:
        """
        Calculate clustering stability using bootstrap sampling
        
        Args:
            X: Feature matrix
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Average stability score
        """
        model_logger.info(f"Calculating stability with {n_bootstrap} bootstrap samples")
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        stability_scores = []
        original_labels = self.predict(X)
        n_clusters = len(np.unique(original_labels))
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            
            # Cluster bootstrap sample
            if self.algorithm == 'kmeans':
                bootstrap_model = KMeans(n_clusters=n_clusters, random_state=i)
                bootstrap_labels = bootstrap_model.fit_predict(X_bootstrap)
                
                # Calculate stability (simplified)
                consistency = np.mean(original_labels[indices] == bootstrap_labels)
                stability_scores.append(consistency)
        
        avg_stability = np.mean(stability_scores)
        model_logger.info(f"Average stability score: {avg_stability:.3f}")
        
        return avg_stability
    
    def save(self, filepath: str):
        """Save the segmentation model"""
        model_data = {
            'model': self.model,
            'algorithm': self.algorithm,
            'optimal_clusters': self.optimal_clusters,
            'segment_profiles': self.segment_profiles,
            'cluster_centers': self.cluster_centers
        }
        joblib.dump(model_data, filepath)
        model_logger.info(f"Segmentation model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a saved segmentation model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.algorithm = model_data['algorithm']
        self.optimal_clusters = model_data['optimal_clusters']
        self.segment_profiles = model_data['segment_profiles']
        self.cluster_centers = model_data['cluster_centers']
        model_logger.info(f"Segmentation model loaded from {filepath}")