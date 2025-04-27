#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Topological Data Analysis module for ATFNet.
Implements TDA for identifying market structures and patterns.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import networkx as nx
from datetime import datetime

logger = logging.getLogger('atfnet.models.topological')


class TopologicalAnalyzer:
    """
    Topological data analysis for ATFNet.
    Implements TDA for identifying market structures and patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'topological_analysis'):
        """
        Initialize the topological analyzer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.n_neighbors = self.config.get('n_neighbors', 15)
        self.resolution = self.config.get('resolution', 0.5)
        self.min_cluster_size = self.config.get('min_cluster_size', 5)
        self.metric = self.config.get('metric', 'euclidean')
        
        # Initialize state variables
        self.mapper_graph = None
        self.persistence_diagram = None
        self.clusters = None
        self.embeddings = None
        
        logger.info("Topological Analyzer initialized")
    
    def create_point_cloud(self, data: np.ndarray, 
                          normalize: bool = True,
                          reduce_dimensions: bool = True,
                          n_components: int = 2) -> np.ndarray:
        """
        Create point cloud from data.
        
        Args:
            data: Input data
            normalize: Whether to normalize data
            reduce_dimensions: Whether to reduce dimensions
            n_components: Number of components for dimension reduction
            
        Returns:
            Point cloud
        """
        # Ensure data is a numpy array
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Normalize data
        if normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
        
        # Reduce dimensions if needed
        if reduce_dimensions and data.shape[1] > n_components:
            # Use PCA for dimension reduction
            pca = PCA(n_components=n_components)
            data_reduced = pca.fit_transform(data)
            
            # Store explained variance
            self.explained_variance = pca.explained_variance_ratio_
            
            logger.info(f"Reduced dimensions from {data.shape[1]} to {n_components} "
                       f"(explained variance: {sum(self.explained_variance):.2f})")
            
            return data_reduced
        else:
            return data
    
    def compute_distance_matrix(self, point_cloud: np.ndarray, 
                               metric: str = None) -> np.ndarray:
        """
        Compute distance matrix for point cloud.
        
        Args:
            point_cloud: Point cloud data
            metric: Distance metric
            
        Returns:
            Distance matrix
        """
        if metric is None:
            metric = self.metric
        
        # Compute pairwise distances
        distances = pdist(point_cloud, metric=metric)
        
        # Convert to square matrix
        distance_matrix = squareform(distances)
        
        return distance_matrix
    
    def build_mapper_graph(self, data: np.ndarray, 
                          filter_function: Callable = None,
                          n_intervals: int = 10,
                          overlap: float = 0.5,
                          n_clusters: int = None) -> nx.Graph:
        """
        Build mapper graph from data.
        
        Args:
            data: Input data
            filter_function: Filter function (default: first PCA component)
            n_intervals: Number of intervals for filter function
            overlap: Overlap between intervals
            n_clusters: Number of clusters per interval
            
        Returns:
            Mapper graph
        """
        # Create point cloud
        point_cloud = self.create_point_cloud(data)
        
        # Apply filter function
        if filter_function is None:
            # Default: use first principal component
            pca = PCA(n_components=1)
            filter_values = pca.fit_transform(data).flatten()
        else:
            filter_values = filter_function(data)
        
        # Create intervals
        min_val, max_val = np.min(filter_values), np.max(filter_values)
        interval_size = (max_val - min_val) / n_intervals
        overlap_size = interval_size * overlap
        
        intervals = []
        for i in range(n_intervals):
            start = min_val + i * interval_size - (i > 0) * overlap_size
            end = min_val + (i + 1) * interval_size + (i < n_intervals - 1) * overlap_size
            intervals.append((start, end))
        
        # Create graph
        G = nx.Graph()
        
        # Process each interval
        node_id = 0
        interval_clusters = []
        
        for i, (start, end) in enumerate(intervals):
            # Get points in interval
            mask = (filter_values >= start) & (filter_values <= end)
            if np.sum(mask) < self.min_cluster_size:
                continue
            
            # Get subcloud
            subcloud = point_cloud[mask]
            subcloud_indices = np.where(mask)[0]
            
            # Cluster subcloud
            clusters = self._cluster_points(subcloud, n_clusters)
            
            # Add nodes and track points
            for cluster_id, cluster in enumerate(clusters):
                if len(cluster) < self.min_cluster_size:
                    continue
                
                # Get original indices
                cluster_indices = subcloud_indices[cluster]
                
                # Add node
                G.add_node(node_id, 
                          interval=i, 
                          cluster=cluster_id, 
                          size=len(cluster),
                          points=cluster_indices.tolist())
                
                # Store for edge creation
                interval_clusters.append((node_id, cluster_indices))
                node_id += 1
        
        # Create edges between nodes that share points
        for i, (node_i, points_i) in enumerate(interval_clusters):
            for j, (node_j, points_j) in enumerate(interval_clusters[i+1:], i+1):
                # Find shared points
                shared_points = np.intersect1d(points_i, points_j)
                
                if len(shared_points) > 0:
                    # Add edge with weight based on number of shared points
                    G.add_edge(node_i, node_j, 
                              weight=len(shared_points),
                              shared_points=shared_points.tolist())
        
        # Store graph
        self.mapper_graph = G
        
        logger.info(f"Built mapper graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def _cluster_points(self, points: np.ndarray, n_clusters: int = None) -> List[np.ndarray]:
        """
        Cluster points using hierarchical clustering.
        
        Args:
            points: Points to cluster
            n_clusters: Number of clusters
            
        Returns:
            List of cluster indices
        """
        if len(points) < 2:
            return [np.array([0])]
        
        # Compute distance matrix
        distances = pdist(points, metric=self.metric)
        
        # Perform hierarchical clustering
        Z = linkage(distances, method='ward')
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            # Estimate based on data
            n_clusters = max(1, min(int(np.sqrt(len(points)) / 2), 10))
        
        # Get cluster labels
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Group points by cluster
        clusters = []
        for i in range(1, np.max(labels) + 1):
            cluster_indices = np.where(labels == i)[0]
            clusters.append(cluster_indices)
        
        return clusters
    
    def compute_persistent_homology(self, point_cloud: np.ndarray, 
                                   max_dimension: int = 1) -> Dict[str, Any]:
        """
        Compute persistent homology.
        
        Args:
            point_cloud: Point cloud data
            max_dimension: Maximum homology dimension
            
        Returns:
            Persistent homology results
        """
        try:
            # Try to import ripser
            from ripser import ripser
            
            # Compute persistent homology
            result = ripser(point_cloud, maxdim=max_dimension)
            
            # Extract persistence diagrams
            diagrams = result['dgms']
            
            # Store result
            self.persistence_diagram = diagrams
            
            logger.info(f"Computed persistent homology up to dimension {max_dimension}")
            
            return {
                'diagrams': diagrams,
                'cocycles': result.get('cocycles', None)
            }
            
        except ImportError:
            logger.warning("Ripser not installed. Using simplified persistence calculation.")
            
            # Compute distance matrix
            distance_matrix = self.compute_distance_matrix(point_cloud)
            
            # Simplified persistence calculation
            diagrams = self._simplified_persistence(distance_matrix, max_dimension)
            
            # Store result
            self.persistence_diagram = diagrams
            
            return {
                'diagrams': diagrams,
                'cocycles': None
            }
    
    def _simplified_persistence(self, distance_matrix: np.ndarray, 
                               max_dimension: int) -> List[np.ndarray]:
        """
        Simplified persistence calculation.
        
        Args:
            distance_matrix: Distance matrix
            max_dimension: Maximum homology dimension
            
        Returns:
            List of persistence diagrams
        """
        # This is a very simplified approach
        # For dimension 0 (connected components)
        n = distance_matrix.shape[0]
        
        # Sort edges by distance
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j, distance_matrix[i, j]))
        
        edges.sort(key=lambda x: x[2])
        
        # Union-find data structure
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        # Track births and deaths
        births = [0] * n
        deaths = []
        
        # Process edges
        for i, j, distance in edges:
            if find(i) != find(j):
                # Components merge
                deaths.append((births[find(i)], distance))
                union(i, j)
        
        # Add infinity for remaining component
        deaths.append((0, np.inf))
        
        # Create diagram for dimension 0
        diagram_0 = np.array(deaths)
        
        # For higher dimensions, return empty diagrams
        diagrams = [diagram_0]
        for _ in range(1, max_dimension + 1):
            diagrams.append(np.array([]))
        
        return diagrams
    
    def plot_mapper_graph(self, node_color: str = 'interval',
                         save_plot: bool = True,
                         filename: str = 'mapper_graph.png') -> None:
        """
        Plot mapper graph.
        
        Args:
            node_color: Attribute to use for node color
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if self.mapper_graph is None:
            logger.warning("No mapper graph to plot. Run build_mapper_graph first.")
            return
        
        try:
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Get graph
            G = self.mapper_graph
            
            # Set node positions using spring layout
            pos = nx.spring_layout(G, seed=42)
            
            # Set node colors
            if node_color == 'interval':
                node_colors = [G.nodes[n].get('interval', 0) for n in G.nodes()]
                cmap = plt.cm.viridis
            elif node_color == 'size':
                node_colors = [G.nodes[n].get('size', 1) for n in G.nodes()]
                cmap = plt.cm.plasma
            else:
                node_colors = 'skyblue'
                cmap = None
            
            # Set node sizes
            node_sizes = [50 + 10 * G.nodes[n].get('size', 1) for n in G.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, 
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 cmap=cmap,
                                 alpha=0.8)
            
            # Draw edges with width based on weight
            edge_widths = [0.5 + 0.5 * G.edges[e].get('weight', 1) / 10 for e in G.edges()]
            nx.draw_networkx_edges(G, pos, 
                                 width=edge_widths,
                                 alpha=0.6)
            
            # Draw labels for larger nodes
            large_nodes = {n: n for n, attr in G.nodes(data=True) if attr.get('size', 0) > 20}
            nx.draw_networkx_labels(G, pos, 
                                  labels=large_nodes,
                                  font_size=8)
            
            plt.title('Mapper Graph')
            plt.axis('off')
            
            # Add colorbar if using color map
            if cmap is not None:
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                plt.colorbar(sm, label=node_color.capitalize())
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Mapper graph plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting mapper graph: {e}")
            raise
    
    def plot_persistence_diagram(self, save_plot: bool = True,
                               filename: str = 'persistence_diagram.png') -> None:
        """
        Plot persistence diagram.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if self.persistence_diagram is None:
            logger.warning("No persistence diagram to plot. Run compute_persistent_homology first.")
            return
        
        try:
            # Create figure
            fig, axes = plt.subplots(1, len(self.persistence_diagram), figsize=(15, 5))
            
            # Ensure axes is a list
            if len(self.persistence_diagram) == 1:
                axes = [axes]
            
            # Plot each dimension
            for i, diagram in enumerate(self.persistence_diagram):
                ax = axes[i]
                
                if len(diagram) > 0:
                    # Plot points
                    ax.scatter(diagram[:, 0], diagram[:, 1], alpha=0.8)
                    
                    # Plot diagonal
                    min_val = np.min(diagram[:, 0]) if len(diagram) > 0 else 0
                    max_val = np.max(diagram[:, 1]) if len(diagram) > 0 else 1
                    if np.isinf(max_val):
                        max_val = np.max(diagram[~np.isinf(diagram[:, 1]), 1]) if np.sum(~np.isinf(diagram[:, 1])) > 0 else 1
                    
                    lim = max(max_val, 1) * 1.1
                    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
                    
                    # Set limits
                    ax.set_xlim([0, lim])
                    ax.set_ylim([0, lim])
                
                ax.set_title(f'Dimension {i}')
                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Persistence diagram plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting persistence diagram: {e}")
            raise
    
    def compute_persistence_landscape(self, diagram: np.ndarray, 
                                     num_landscapes: int = 5,
                                     resolution: int = 100) -> np.ndarray:
        """
        Compute persistence landscape from persistence diagram.
        
        Args:
            diagram: Persistence diagram
            num_landscapes: Number of landscape functions
            resolution: Resolution of landscape
            
        Returns:
            Persistence landscape
        """
        if len(diagram) == 0:
            return np.zeros((num_landscapes, resolution))
        
        # Remove points with infinite death
        finite_points = diagram[~np.isinf(diagram[:, 1])]
        
        if len(finite_points) == 0:
            return np.zeros((num_landscapes, resolution))
        
        # Determine range
        min_birth = np.min(finite_points[:, 0])
        max_death = np.max(finite_points[:, 1])
        
        # Create grid
        grid = np.linspace(min_birth, max_death, resolution)
        
        # Initialize landscape
        landscape = np.zeros((num_landscapes, resolution))
        
        # Compute landscape functions
        for i, x in enumerate(grid):
            # Compute all tent functions at x
            tent_values = []
            
            for birth, death in finite_points:
                if birth <= x <= death:
                    # In the middle of the tent
                    tent_values.append(min(x - birth, death - x))
                else:
                    # Outside the tent
                    tent_values.append(0)
            
            # Sort tent values in descending order
            tent_values.sort(reverse=True)
            
            # Fill landscape functions
            for j in range(min(num_landscapes, len(tent_values))):
                landscape[j, i] = tent_values[j]
        
        return landscape
    
    def plot_persistence_landscape(self, diagram: np.ndarray = None,
                                  save_plot: bool = True,
                                  filename: str = 'persistence_landscape.png') -> None:
        """
        Plot persistence landscape.
        
        Args:
            diagram: Persistence diagram (default: first diagram)
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if self.persistence_diagram is None:
            logger.warning("No persistence diagram available. Run compute_persistent_homology first.")
            return
        
        if diagram is None:
            diagram = self.persistence_diagram[0]  # Use dimension 0 by default
        
        try:
            # Compute landscape
            landscape = self.compute_persistence_landscape(diagram)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot landscape functions
            x = np.linspace(0, 1, landscape.shape[1])
            
            for i in range(landscape.shape[0]):
                plt.plot(x, landscape[i], label=f'λ_{i+1}')
            
            plt.xlabel('x')
            plt.ylabel('λ(x)')
            plt.title('Persistence Landscape')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Persistence landscape plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting persistence landscape: {e}")
            raise
    
    def compute_betti_curves(self, diagrams: List[np.ndarray] = None,
                            resolution: int = 100) -> np.ndarray:
        """
        Compute Betti curves from persistence diagrams.
        
        Args:
            diagrams: List of persistence diagrams
            resolution: Resolution of curves
            
        Returns:
            Betti curves
        """
        if diagrams is None:
            if self.persistence_diagram is None:
                logger.warning("No persistence diagram available. Run compute_persistent_homology first.")
                return None
            diagrams = self.persistence_diagram
        
        # Determine range
        min_val = float('inf')
        max_val = 0
        
        for diagram in diagrams:
            if len(diagram) > 0:
                min_val = min(min_val, np.min(diagram[:, 0]))
                
                # Handle infinite values
                finite_deaths = diagram[~np.isinf(diagram[:, 1]), 1]
                if len(finite_deaths) > 0:
                    max_val = max(max_val, np.max(finite_deaths))
        
        if min_val == float('inf'):
            min_val = 0
        
        # Create grid
        grid = np.linspace(min_val, max_val, resolution)
        
        # Initialize Betti curves
        betti_curves = np.zeros((len(diagrams), resolution))
        
        # Compute Betti curves
        for dim, diagram in enumerate(diagrams):
            for i, threshold in enumerate(grid):
                # Count points born before threshold and dying after threshold
                if len(diagram) > 0:
                    betti_curves[dim, i] = np.sum((diagram[:, 0] <= threshold) & 
                                                 ((diagram[:, 1] > threshold) | np.isinf(diagram[:, 1])))
        
        return betti_curves
    
    def plot_betti_curves(self, save_plot: bool = True,
                         filename: str = 'betti_curves.png') -> None:
        """
        Plot Betti curves.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if self.persistence_diagram is None:
            logger.warning("No persistence diagram available. Run compute_persistent_homology first.")
            return
        
        try:
            # Compute Betti curves
            betti_curves = self.compute_betti_curves()
            
            if betti_curves is None:
                return
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot Betti curves
            x = np.linspace(0, 1, betti_curves.shape[1])
            
            for dim in range(betti_curves.shape[0]):
                plt.plot(x, betti_curves[dim], label=f'Dimension {dim}')
            
            plt.xlabel('Threshold')
            plt.ylabel('Betti Number')
            plt.title('Betti Curves')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Betti curves plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting Betti curves: {e}")
            raise
    
    def detect_market_regimes(self, data: np.ndarray, 
                             n_regimes: int = 4) -> Dict[str, Any]:
        """
        Detect market regimes using topological features.
        
        Args:
            data: Input data
            n_regimes: Number of regimes to detect
            
        Returns:
            Dictionary with regime detection results
        """
        try:
            # Create point cloud
            point_cloud = self.create_point_cloud(data)
            
            # Compute persistent homology
            self.compute_persistent_homology(point_cloud)
            
            # Extract topological features
            features = self._extract_topological_features()
            
            # Cluster features to identify regimes
            from sklearn.cluster import KMeans
            
            # Ensure we have enough samples
            if len(features) < n_regimes:
                logger.warning(f"Not enough samples for {n_regimes} regimes. Using {len(features)} regimes instead.")
                n_regimes = max(2, len(features))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            regime_labels = kmeans.fit_predict(features)
            
            # Characterize regimes
            regime_characteristics = {}
            
            for i in range(n_regimes):
                mask = regime_labels == i
                regime_data = data[mask]
                
                if len(regime_data) > 0:
                    characteristics = {
                        'count': np.sum(mask),
                        'mean': np.mean(regime_data, axis=0),
                        'std': np.std(regime_data, axis=0),
                        'min': np.min(regime_data, axis=0),
                        'max': np.max(regime_data, axis=0)
                    }
                    
                    regime_characteristics[f'regime_{i}'] = characteristics
            
            # Store results
            self.clusters = {
                'labels': regime_labels,
                'centers': kmeans.cluster_centers_,
                'characteristics': regime_characteristics
            }
            
            logger.info(f"Detected {n_regimes} market regimes")
            
            return {
                'regime_labels': regime_labels,
                'regime_characteristics': regime_characteristics,
                'topological_features': features
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regimes: {e}")
            raise
    
    def _extract_topological_features(self) -> np.ndarray:
        """
        Extract topological features from persistence diagrams.
        
        Returns:
            Array of topological features
        """
        if self.persistence_diagram is None:
            raise ValueError("No persistence diagram available. Run compute_persistent_homology first.")
        
        features = []
        
        # Extract features from each dimension
        for dim, diagram in enumerate(self.persistence_diagram):
            if len(diagram) > 0:
                # Remove infinite values for statistics
                finite_diagram = diagram[~np.isinf(diagram[:, 1])]
                
                if len(finite_diagram) > 0:
                    # Persistence (death - birth)
                    persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
                    
                    # Extract statistics
                    features.extend([
                        len(diagram),  # Number of features
                        np.mean(persistence) if len(persistence) > 0 else 0,  # Mean persistence
                        np.std(persistence) if len(persistence) > 0 else 0,  # Std of persistence
                        np.max(persistence) if len(persistence) > 0 else 0,  # Max persistence
                        np.sum(persistence) if len(persistence) > 0 else 0,  # Total persistence
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def visualize_market_regimes(self, data: np.ndarray, 
                               regime_labels: np.ndarray = None,
                               save_plot: bool = True,
                               filename: str = 'market_regimes.png') -> None:
        """
        Visualize market regimes.
        
        Args:
            data: Input data
            regime_labels: Regime labels (default: from detect_market_regimes)
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if regime_labels is None:
            if self.clusters is None:
                logger.warning("No regime labels available. Run detect_market_regimes first.")
                return
            regime_labels = self.clusters['labels']
        
        try:
            # Create 2D embedding for visualization
            if data.shape[1] > 2:
                # Use t-SNE for dimension reduction
                tsne = TSNE(n_components=2, random_state=42)
                embedding = tsne.fit_transform(data)
            else:
                embedding = data
            
            # Store embedding
            self.embeddings = embedding
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot regimes
            unique_regimes = np.unique(regime_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
            
            for i, regime in enumerate(unique_regimes):
                mask = regime_labels == regime
                plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                          color=colors[i], label=f'Regime {regime}',
                          alpha=0.7)
            
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('Market Regimes')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Market regimes plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing market regimes: {e}")
            raise
    
    def get_regime_transitions(self, regime_labels: np.ndarray = None) -> np.ndarray:
        """
        Get regime transition matrix.
        
        Args:
            regime_labels: Regime labels (default: from detect_market_regimes)
            
        Returns:
            Transition matrix
        """
        if regime_labels is None:
            if self.clusters is None:
                logger.warning("No regime labels available. Run detect_market_regimes first.")
                return None
            regime_labels = self.clusters['labels']
        
        # Get unique regimes
        unique_regimes = np.unique(regime_labels)
        n_regimes = len(unique_regimes)
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regime_labels) - 1):
            from_regime = regime_labels[i]
            to_regime = regime_labels[i + 1]
            
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            
            transition_matrix[from_idx, to_idx] += 1
        
        # Normalize by row
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), where=row_sums!=0)
        
        return transition_matrix
    
    def plot_regime_transitions(self, transition_matrix: np.ndarray = None,
                              save_plot: bool = True,
                              filename: str = 'regime_transitions.png') -> None:
        """
        Plot regime transition matrix.
        
        Args:
            transition_matrix: Transition matrix (default: from get_regime_transitions)
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if transition_matrix is None:
            transition_matrix = self.get_regime_transitions()
            
        if transition_matrix is None:
            return
        
        try:
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot transition matrix as heatmap
            plt.imshow(transition_matrix, cmap='viridis', interpolation='nearest')
            
            # Add colorbar
            plt.colorbar(label='Transition Probability')
            
            # Add labels
            n_regimes = transition_matrix.shape[0]
            plt.xticks(range(n_regimes), [f'Regime {i}' for i in range(n_regimes)])
            plt.yticks(range(n_regimes), [f'Regime {i}' for i in range(n_regimes)])
            
            plt.xlabel('To Regime')
            plt.ylabel('From Regime')
            plt.title('Regime Transition Matrix')
            
            # Add text annotations
            for i in range(n_regimes):
                for j in range(n_regimes):
                    plt.text(j, i, f'{transition_matrix[i, j]:.2f}',
                           ha='center', va='center',
                           color='white' if transition_matrix[i, j] > 0.5 else 'black')
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Regime transitions plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting regime transitions: {e}")
            raise
