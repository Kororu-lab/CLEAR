"""
News clustering module for grouping related financial news articles.
Based on NAVER's hierarchical agglomerative clustering approach.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsClustering:
    """
    Clustering engine for grouping related financial news articles.
    Based on NAVER's hierarchical agglomerative clustering approach.
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.7,
                 min_cluster_size: int = 3,
                 max_cluster_size: int = 20,
                 linkage: str = 'average',
                 n_clusters: int = None,
                 models_dir: str = None):
        """
        Initialize the news clustering engine.
        
        Args:
            distance_threshold: Threshold for forming clusters
            min_cluster_size: Minimum number of articles to form a cluster
            max_cluster_size: Maximum number of articles in a cluster
            linkage: Linkage criterion for hierarchical clustering
            n_clusters: Number of clusters (if None, determined by distance_threshold)
            models_dir: Directory to save/load models
        """
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.linkage = linkage
        self.n_clusters = n_clusters
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "clustering")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize clustering model
        self.clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=None if n_clusters else distance_threshold,
            metric='cosine',
            linkage=linkage,
            compute_distances=True
        )
        
        logger.info(f"Initialized NewsClustering with distance_threshold={distance_threshold}, min_cluster_size={min_cluster_size}")
    
    def cluster_articles(self, articles_df: pd.DataFrame, 
                        vector_col: str = 'vector',
                        title_col: str = 'Title',
                        content_col: str = 'Body') -> pd.DataFrame:
        """
        Cluster news articles based on their vector representations.
        
        Args:
            articles_df: DataFrame containing news articles with vector representations
            vector_col: Column name for vector representations
            title_col: Column name for article titles
            content_col: Column name for article content
            
        Returns:
            DataFrame with cluster assignments
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for clustering")
            return articles_df
            
        if vector_col not in articles_df.columns:
            logger.error(f"Vector column '{vector_col}' not found in DataFrame")
            raise ValueError(f"Vector column '{vector_col}' not found in DataFrame")
            
        logger.info(f"Clustering {len(articles_df)} articles")
        
        # Create a copy to avoid modifying the original
        df = articles_df.copy()
        
        try:
            # Extract vectors from DataFrame
            vectors = np.array(df[vector_col].tolist())
            
            # Apply clustering
            if len(vectors) > 1:  # Need at least 2 articles to cluster
                cluster_labels = self.clustering.fit_predict(vectors)
                df['cluster_id'] = cluster_labels
                
                # Count articles per cluster
                cluster_counts = df['cluster_id'].value_counts().to_dict()
                
                # Filter out small clusters
                df['cluster_size'] = df['cluster_id'].map(cluster_counts)
                df['valid_cluster'] = df['cluster_size'] >= self.min_cluster_size
                
                # For articles in invalid clusters, set cluster_id to -1 (no cluster)
                df.loc[~df['valid_cluster'], 'cluster_id'] = -1
                
                # Split large clusters if needed
                if self.max_cluster_size:
                    df = self._split_large_clusters(df, vectors)
                
                # Generate cluster names/topics
                clusters = df[df['cluster_id'] >= 0]['cluster_id'].unique()
                cluster_topics = {}
                
                for cluster_id in clusters:
                    cluster_articles = df[df['cluster_id'] == cluster_id]
                    topic = self._generate_cluster_topic(cluster_articles, title_col, content_col)
                    cluster_topics[cluster_id] = topic
                
                # Add cluster topics to DataFrame
                df['cluster_topic'] = df['cluster_id'].map(
                    lambda x: cluster_topics.get(x, "") if x >= 0 else ""
                )
                
                # Calculate cluster quality metrics
                if len(clusters) > 1 and len(df[df['cluster_id'] >= 0]) > 1:
                    try:
                        valid_vectors = vectors[df['cluster_id'] >= 0]
                        valid_labels = df[df['cluster_id'] >= 0]['cluster_id'].values
                        silhouette = silhouette_score(valid_vectors, valid_labels, metric='cosine')
                        df['silhouette_score'] = silhouette
                        logger.info(f"Clustering silhouette score: {silhouette:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not calculate silhouette score: {str(e)}")
                
                logger.info(f"Created {len(clusters)} valid clusters")
                
                # Save clustering results
                self._save_clustering_results(df)
                
                # Visualize clusters if there are enough articles
                if len(vectors) >= 10:
                    self._visualize_clusters(vectors, df['cluster_id'].values)
            else:
                # Not enough articles to cluster
                df['cluster_id'] = -1
                df['cluster_size'] = 1
                df['valid_cluster'] = False
                df['cluster_topic'] = ""
                logger.info("Not enough articles to perform clustering")
            
            return df
        except Exception as e:
            logger.error(f"Error clustering articles: {str(e)}")
            raise
    
    def get_similar_articles(self, article_vector: np.ndarray, 
                           articles_df: pd.DataFrame,
                           vector_col: str = 'vector',
                           top_n: int = 5) -> pd.DataFrame:
        """
        Find articles similar to a given article vector.
        
        Args:
            article_vector: Vector representation of the query article
            articles_df: DataFrame containing candidate articles
            vector_col: Column name for vector representations
            top_n: Number of similar articles to return
            
        Returns:
            DataFrame with similar articles and similarity scores
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for similarity search")
            return pd.DataFrame()
            
        if vector_col not in articles_df.columns:
            logger.error(f"Vector column '{vector_col}' not found in DataFrame")
            raise ValueError(f"Vector column '{vector_col}' not found in DataFrame")
            
        logger.info(f"Finding {top_n} articles similar to query vector")
        
        try:
            # Extract vectors from DataFrame
            vectors = np.array(articles_df[vector_col].tolist())
            
            # Reshape query vector if needed
            if article_vector.ndim == 1:
                article_vector = article_vector.reshape(1, -1)
                
            # Calculate cosine similarities
            similarities = cosine_similarity(article_vector, vectors)[0]
            
            # Add similarity scores to DataFrame
            result_df = articles_df.copy()
            result_df['similarity'] = similarities
            
            # Sort by similarity (descending) and take top_n
            result_df = result_df.sort_values('similarity', ascending=False).head(top_n)
            
            logger.info(f"Found {len(result_df)} similar articles")
            return result_df
        except Exception as e:
            logger.error(f"Error finding similar articles: {str(e)}")
            raise
    
    def update_clusters(self, existing_df: pd.DataFrame, 
                       new_articles_df: pd.DataFrame,
                       vector_col: str = 'vector') -> pd.DataFrame:
        """
        Update existing clusters with new articles.
        
        Args:
            existing_df: DataFrame with existing clustered articles
            new_articles_df: DataFrame with new articles to add
            vector_col: Column name for vector representations
            
        Returns:
            Updated DataFrame with cluster assignments
        """
        if len(new_articles_df) == 0:
            logger.info("No new articles to add to clusters")
            return existing_df
            
        if vector_col not in existing_df.columns or vector_col not in new_articles_df.columns:
            logger.error(f"Vector column '{vector_col}' not found in one of the DataFrames")
            raise ValueError(f"Vector column '{vector_col}' not found in one of the DataFrames")
            
        logger.info(f"Updating clusters with {len(new_articles_df)} new articles")
        
        try:
            # Combine existing and new articles
            combined_df = pd.concat([existing_df, new_articles_df], ignore_index=True)
            
            # Re-cluster all articles
            result_df = self.cluster_articles(combined_df, vector_col)
            
            logger.info(f"Updated clusters, now have {len(result_df['cluster_id'].unique())} clusters")
            return result_df
        except Exception as e:
            logger.error(f"Error updating clusters: {str(e)}")
            raise
    
    def get_trending_clusters(self, articles_df: pd.DataFrame, 
                             timeframe_hours: int = 24,
                             min_articles: int = 5) -> List[int]:
        """
        Identify trending clusters based on article volume and recency.
        
        Args:
            articles_df: DataFrame containing clustered articles
            timeframe_hours: Timeframe in hours to consider for trending
            min_articles: Minimum number of articles to consider a cluster trending
            
        Returns:
            List of trending cluster IDs
        """
        if len(articles_df) == 0 or 'cluster_id' not in articles_df.columns:
            logger.warning("Invalid DataFrame provided for trending clusters")
            return []
            
        if 'Date' not in articles_df.columns:
            logger.warning("Date column not found in DataFrame")
            return []
            
        logger.info(f"Identifying trending clusters in the last {timeframe_hours} hours")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Convert Date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                # Handle the specific date format (20250101 18:56)
                df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(str(x).split()[0], format='%Y%m%d'))
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
            
            # Filter articles within timeframe
            recent_df = df[df['Date'] >= cutoff_time]
            
            if len(recent_df) == 0:
                logger.warning(f"No articles found within the last {timeframe_hours} hours")
                return []
            
            # Count articles per cluster
            cluster_counts = recent_df['cluster_id'].value_counts()
            
            # Filter clusters with minimum number of articles
            trending_clusters = cluster_counts[cluster_counts >= min_articles].index.tolist()
            
            # Filter out invalid clusters (-1)
            trending_clusters = [c for c in trending_clusters if c >= 0]
            
            logger.info(f"Identified {len(trending_clusters)} trending clusters")
            return trending_clusters
        except Exception as e:
            logger.error(f"Error identifying trending clusters: {str(e)}")
            return []
    
    def _split_large_clusters(self, df: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
        """
        Split large clusters into smaller ones.
        
        Args:
            df: DataFrame with initial cluster assignments
            vectors: Vector representations of articles
            
        Returns:
            DataFrame with updated cluster assignments
        """
        # Find clusters that exceed the maximum size
        large_clusters = df[df['cluster_size'] > self.max_cluster_size]['cluster_id'].unique()
        
        if not large_clusters.size:
            return df
            
        logger.info(f"Splitting {len(large_clusters)} large clusters")
        
        # Get the maximum cluster ID to start assigning new IDs
        max_cluster_id = df['cluster_id'].max()
        next_cluster_id = max_cluster_id + 1
        
        # Process each large cluster
        for cluster_id in large_clusters:
            # Get articles in this cluster
            cluster_mask = df['cluster_id'] == cluster_id
            cluster_articles = df[cluster_mask]
            cluster_vectors = vectors[cluster_mask.values]
            
            # Create a sub-clustering model with more clusters
            n_subclusters = max(2, len(cluster_articles) // self.max_cluster_size + 1)
            subclustering = AgglomerativeClustering(
                n_clusters=n_subclusters,
                metric='cosine',
                linkage=self.linkage
            )
            
            # Apply sub-clustering
            subcluster_labels = subclustering.fit_predict(cluster_vectors)
            
            # Map sub-cluster labels to new cluster IDs
            subcluster_mapping = {}
            for subcluster_id in range(n_subclusters):
                subcluster_mapping[subcluster_id] = next_cluster_id
                next_cluster_id += 1
            
            # Update cluster IDs in the DataFrame
            for i, (idx, _) in enumerate(cluster_articles.iterrows()):
                subcluster_id = subcluster_labels[i]
                new_cluster_id = subcluster_mapping[subcluster_id]
                df.at[idx, 'cluster_id'] = new_cluster_id
        
        # Recalculate cluster sizes
        cluster_counts = df['cluster_id'].value_counts().to_dict()
        df['cluster_size'] = df['cluster_id'].map(cluster_counts)
        
        # Update valid_cluster flag
        df['valid_cluster'] = df['cluster_size'] >= self.min_cluster_size
        
        # Set cluster_id to -1 for invalid clusters
        df.loc[~df['valid_cluster'], 'cluster_id'] = -1
        
        logger.info(f"Split large clusters, now have {len(df['cluster_id'].unique())} clusters")
        return df
    
    def _generate_cluster_topic(self, cluster_articles: pd.DataFrame,
                              title_col: str = 'Title',
                              content_col: str = 'Body') -> str:
        """
        Generate a topic name for a cluster based on its articles.
        
        Args:
            cluster_articles: DataFrame containing articles in a cluster
            title_col: Column name for article titles
            content_col: Column name for article content
            
        Returns:
            Topic name for the cluster
        """
        if len(cluster_articles) == 0:
            return "Empty Cluster"
            
        try:
            # Use titles for topic generation if available
            if title_col in cluster_articles.columns:
                # Concatenate all titles
                all_titles = ' '.join(cluster_articles[title_col].fillna(''))
                
                # Simple word frequency analysis
                words = all_titles.lower().split()
                word_freq = {}
                
                # Skip common words
                stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'and', 'or',
                            '이', '그', '저', '것', '수', '등', '들', '및', '에', '에서', '의', '을', '를', '이', '가',
                            '은', '는', '이다', '있다', '하다'}
                
                for word in words:
                    if word not in stopwords and len(word) > 1:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top words
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                
                if top_words:
                    # Create topic from top words
                    topic = ' '.join([word for word, _ in top_words])
                    return topic
            
            # Fallback to first article title
            if title_col in cluster_articles.columns:
                return cluster_articles.iloc[0][title_col]
            
            # Last resort
            return f"Cluster {cluster_articles.iloc[0]['cluster_id']}"
        except Exception as e:
            logger.error(f"Error generating cluster topic: {str(e)}")
            return f"Cluster {cluster_articles.iloc[0]['cluster_id']}"
    
    def _save_clustering_results(self, df: pd.DataFrame) -> None:
        """
        Save clustering results for later analysis.
        
        Args:
            df: DataFrame with cluster assignments
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save cluster statistics
            cluster_stats = df[df['cluster_id'] >= 0].groupby('cluster_id').agg({
                'cluster_size': 'first',
                'cluster_topic': 'first'
            }).reset_index()
            
            stats_path = os.path.join(self.models_dir, f"cluster_stats_{timestamp}.csv")
            cluster_stats.to_csv(stats_path, index=False)
            
            logger.info(f"Saved cluster statistics to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving clustering results: {str(e)}")
    
    def _visualize_clusters(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """
        Visualize clusters using dimensionality reduction.
        
        Args:
            vectors: Vector representations of articles
            labels: Cluster labels
        """
        try:
            # Only visualize if there are valid clusters
            if np.all(labels == -1):
                logger.warning("No valid clusters to visualize")
                return
                
            # Use SVD to reduce to 2 dimensions for visualization
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=2, random_state=42)
            reduced_vectors = svd.fit_transform(vectors)
            
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot points with cluster colors (-1 is noise, shown in gray)
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                if label == -1:
                    color = 'gray'
                else:
                    color = colors[i]
                    
                mask = labels == label
                plt.scatter(
                    reduced_vectors[mask, 0],
                    reduced_vectors[mask, 1],
                    c=[color],
                    label=f'Cluster {label}' if label != -1 else 'Noise',
                    alpha=0.7
                )
            
            plt.title('News Article Clusters')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(loc='best')
            
            # Save the plot
            plot_path = os.path.join(self.models_dir, f"cluster_visualization_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved cluster visualization to {plot_path}")
        except Exception as e:
            logger.error(f"Error visualizing clusters: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Sample vectors (simplified for demonstration)
    sample_vectors = np.random.rand(20, 50)
    
    # Create a DataFrame with sample data
    df = pd.DataFrame({
        'Title': [f"Article {i}" for i in range(20)],
        'Body': [f"Content {i}" for i in range(20)],
        'Date': pd.date_range(start='2025-01-01', periods=20),
        'vector': list(sample_vectors)
    })
    
    # Create clustering instance
    clustering = NewsClustering(distance_threshold=0.5, min_cluster_size=2)
    
    # Cluster articles
    clustered_df = clustering.cluster_articles(df)
    
    print(f"Number of clusters: {len(clustered_df['cluster_id'].unique())}")
    print(f"Cluster sizes: {clustered_df['cluster_size'].value_counts().to_dict()}")
    print(f"Cluster topics: {clustered_df['cluster_topic'].unique()}")
