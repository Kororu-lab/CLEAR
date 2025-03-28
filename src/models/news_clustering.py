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
                 models_dir: str = None,
                 use_stock_impact: bool = True,
                 stock_impact_weight: float = 0.3,
                 time_decay_factor: float = 0.1,
                 max_days_window: int = 7):
        """
        Initialize the news clustering engine.
        
        Args:
            distance_threshold: Threshold for forming clusters
            min_cluster_size: Minimum number of articles to form a cluster
            max_cluster_size: Maximum number of articles in a cluster
            linkage: Linkage criterion for hierarchical clustering
            n_clusters: Number of clusters (if None, determined by distance_threshold)
            models_dir: Directory to save/load models
            use_stock_impact: Whether to incorporate stock impact in clustering
            stock_impact_weight: Weight of stock impact in similarity calculation
            time_decay_factor: Factor for time-based similarity decay
            max_days_window: Maximum days window for time-based clustering
        """
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.linkage = linkage
        self.n_clusters = n_clusters
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "clustering")
        self.use_stock_impact = use_stock_impact
        self.stock_impact_weight = stock_impact_weight
        self.time_decay_factor = time_decay_factor
        self.max_days_window = max_days_window
        
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
                        content_col: str = 'Body',
                        date_col: str = 'Date',
                        impact_col: str = 'impact_overall') -> pd.DataFrame:
        """
        Cluster news articles based on their vector representations.
        
        Args:
            articles_df: DataFrame containing news articles with vector representations
            vector_col: Column name for vector representations
            title_col: Column name for article titles
            content_col: Column name for article content
            date_col: Column name for article dates
            impact_col: Column name for stock impact scores
            
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
                # If using stock impact and time-based clustering
                if self.use_stock_impact and impact_col in df.columns and date_col in df.columns:
                    # Calculate custom similarity matrix
                    similarity_matrix = self._calculate_custom_similarity(df, vectors, date_col, impact_col)
                    
                    # Convert similarity to distance
                    distance_matrix = 1 - similarity_matrix
                    
                    # Use precomputed distances
                    clustering_model = AgglomerativeClustering(
                        n_clusters=self.n_clusters,
                        distance_threshold=None if self.n_clusters else self.distance_threshold,
                        metric='precomputed',
                        linkage=self.linkage
                    )
                    
                    cluster_labels = clustering_model.fit_predict(distance_matrix)
                else:
                    # Use standard clustering with cosine distance
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
                
                return df
            else:
                # Not enough articles to cluster
                df['cluster_id'] = -1
                df['cluster_size'] = 1
                df['valid_cluster'] = False
                df['cluster_topic'] = ""
                logger.warning("Not enough articles to perform clustering")
                return df
                
        except Exception as e:
            logger.error(f"Error clustering articles: {str(e)}")
            raise
    
    def _calculate_custom_similarity(self, df: pd.DataFrame, 
                                   vectors: np.ndarray,
                                   date_col: str,
                                   impact_col: str) -> np.ndarray:
        """
        Calculate custom similarity matrix incorporating content, time, and impact.
        
        Args:
            df: DataFrame containing articles
            vectors: Article vector representations
            date_col: Column name for article dates
            impact_col: Column name for stock impact scores
            
        Returns:
            Custom similarity matrix
        """
        # Calculate content-based similarity (cosine)
        content_similarity = cosine_similarity(vectors)
        
        # Initialize custom similarity with content similarity
        custom_similarity = content_similarity.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
        
        # Calculate time-based similarity
        n = len(df)
        time_similarity = np.ones((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Calculate time difference in days
                time_diff = abs((df.iloc[i][date_col] - df.iloc[j][date_col]).total_seconds() / (24 * 3600))
                
                # Apply time decay
                if time_diff > self.max_days_window:
                    time_similarity[i, j] = 0
                else:
                    time_similarity[i, j] = np.exp(-self.time_decay_factor * time_diff)
                
                # Make symmetric
                time_similarity[j, i] = time_similarity[i, j]
        
        # Calculate impact-based similarity
        impact_similarity = np.ones((n, n))
        
        if impact_col in df.columns:
            for i in range(n):
                for j in range(i+1, n):
                    # Calculate impact difference
                    impact_i = df.iloc[i].get(impact_col, 0)
                    impact_j = df.iloc[j].get(impact_col, 0)
                    
                    if pd.isna(impact_i):
                        impact_i = 0
                    if pd.isna(impact_j):
                        impact_j = 0
                    
                    # Similar impact scores increase similarity
                    impact_diff = abs(impact_i - impact_j) / 10  # Normalize to 0-1 range (impact is -5 to 5)
                    impact_similarity[i, j] = 1 - impact_diff
                    
                    # Make symmetric
                    impact_similarity[j, i] = impact_similarity[i, j]
        
        # Combine similarities with weights
        content_weight = 1 - self.stock_impact_weight
        impact_weight = self.stock_impact_weight
        
        # Final similarity is weighted combination of content, time, and impact
        custom_similarity = (
            content_weight * content_similarity * time_similarity + 
            impact_weight * impact_similarity
        )
        
        return custom_similarity
    
    def _split_large_clusters(self, df: pd.DataFrame, vectors: np.ndarray) -> pd.DataFrame:
        """
        Split large clusters into smaller ones.
        
        Args:
            df: DataFrame with initial cluster assignments
            vectors: Article vector representations
            
        Returns:
            DataFrame with updated cluster assignments
        """
        # Find large clusters
        large_clusters = df[df['cluster_size'] > self.max_cluster_size]['cluster_id'].unique()
        
        if len(large_clusters) == 0:
            return df
            
        logger.info(f"Splitting {len(large_clusters)} large clusters")
        
        # Get the maximum cluster ID
        max_cluster_id = df['cluster_id'].max()
        next_cluster_id = max_cluster_id + 1
        
        # Process each large cluster
        for cluster_id in large_clusters:
            # Get articles in this cluster
            cluster_mask = df['cluster_id'] == cluster_id
            cluster_size = df[cluster_mask].shape[0]
            
            # Skip if not actually large (shouldn't happen, but just in case)
            if cluster_size <= self.max_cluster_size:
                continue
                
            # Get vectors for this cluster
            cluster_vectors = vectors[cluster_mask]
            
            # Calculate number of subclusters
            n_subclusters = int(np.ceil(cluster_size / self.max_cluster_size))
            
            # Create subclustering model
            subclustering = AgglomerativeClustering(
                n_clusters=n_subclusters,
                linkage=self.linkage
            )
            
            # Apply subclustering
            subcluster_labels = subclustering.fit_predict(cluster_vectors)
            
            # Assign new cluster IDs
            for i, subcluster_id in enumerate(np.unique(subcluster_labels)):
                # Get indices of articles in this subcluster
                subcluster_indices = np.where(cluster_mask)[0][subcluster_labels == subcluster_id]
                
                # Skip if too small
                if len(subcluster_indices) < self.min_cluster_size:
                    df.loc[df.index[subcluster_indices], 'cluster_id'] = -1
                    df.loc[df.index[subcluster_indices], 'valid_cluster'] = False
                else:
                    # Assign new cluster ID
                    df.loc[df.index[subcluster_indices], 'cluster_id'] = next_cluster_id
                    df.loc[df.index[subcluster_indices], 'cluster_size'] = len(subcluster_indices)
                    df.loc[df.index[subcluster_indices], 'valid_cluster'] = True
                    next_cluster_id += 1
        
        # Update cluster sizes
        cluster_counts = df['cluster_id'].value_counts().to_dict()
        df['cluster_size'] = df['cluster_id'].map(lambda x: cluster_counts.get(x, 0) if x >= 0 else 0)
        
        logger.info(f"Split large clusters, now have {len(df[df['cluster_id'] >= 0]['cluster_id'].unique())} clusters")
        
        return df
    
    def _generate_cluster_topic(self, cluster_df: pd.DataFrame, 
                              title_col: str, 
                              content_col: str) -> str:
        """
        Generate a topic name for a cluster based on common terms.
        
        Args:
            cluster_df: DataFrame containing articles in a cluster
            title_col: Column name for article titles
            content_col: Column name for article content
            
        Returns:
            Topic name for the cluster
        """
        try:
            # Use titles for topic generation
            if title_col in cluster_df.columns:
                titles = cluster_df[title_col].tolist()
                
                # Find common words in titles
                common_words = self._find_common_words(titles, min_count=2, max_words=5)
                
                if common_words:
                    return " ".join(common_words)
                    
            # Fallback to first title
            if title_col in cluster_df.columns and len(cluster_df) > 0:
                first_title = cluster_df.iloc[0][title_col]
                
                # Truncate if too long
                if len(first_title) > 50:
                    return first_title[:47] + "..."
                else:
                    return first_title
                    
            # Fallback to cluster ID
            return f"Cluster {cluster_df['cluster_id'].iloc[0]}"
            
        except Exception as e:
            logger.error(f"Error generating cluster topic: {str(e)}")
            return f"Cluster {cluster_df['cluster_id'].iloc[0]}"
    
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

    def _find_common_words(self, texts: List[str], 
                         min_count: int = 2, 
                         max_words: int = 5) -> List[str]:
        """
        Find common words across a list of texts.
        
        Args:
            texts: List of text strings
            min_count: Minimum number of occurrences to consider a word common
            max_words: Maximum number of words to include in the result
            
        Returns:
            List of common words
        """
        if not texts:
            return []
            
        try:
            # Tokenize texts
            tokenized_texts = []
            for text in texts:
                # Simple tokenization by splitting on spaces and removing punctuation
                tokens = text.lower()
                for char in ".,!?;:()[]{}\"'":
                    tokens = tokens.replace(char, " ")
                tokens = tokens.split()
                tokenized_texts.append(tokens)
                
            # Count word occurrences
            word_counts = {}
            for tokens in tokenized_texts:
                # Count unique words in each text
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    word_counts[token] = word_counts.get(token, 0) + 1
                    
            # Filter by minimum count
            common_words = [word for word, count in word_counts.items() if count >= min_count]
            
            # Sort by count (descending)
            common_words.sort(key=lambda x: word_counts[x], reverse=True)
            
            # Take top N words
            return common_words[:max_words]
            
        except Exception as e:
            logger.error(f"Error finding common words: {str(e)}")
            return []
    
    def _save_clustering_results(self, df: pd.DataFrame) -> None:
        """
        Save clustering results to disk.
        
        Args:
            df: DataFrame with cluster assignments
        """
        try:
            # Save cluster assignments
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.models_dir, f"clustering_results_{timestamp}.joblib")
            
            # Extract relevant columns
            results_df = df[['cluster_id', 'cluster_size', 'valid_cluster', 'cluster_topic']].copy()
            
            joblib.dump(results_df, results_path)
            logger.info(f"Saved clustering results to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving clustering results: {str(e)}")
    
    def visualize_clusters(self, df: pd.DataFrame, 
                          vector_col: str = 'vector',
                          save_path: Optional[str] = None) -> None:
        """
        Visualize clusters using dimensionality reduction.
        
        Args:
            df: DataFrame with cluster assignments
            vector_col: Column name for vector representations
            save_path: Path to save the visualization
        """
        if len(df) == 0 or 'cluster_id' not in df.columns:
            logger.warning("Cannot visualize clusters: no cluster assignments found")
            return
            
        if vector_col not in df.columns:
            logger.error(f"Vector column '{vector_col}' not found in DataFrame")
            return
            
        try:
            # Extract vectors and valid cluster assignments
            vectors = np.array(df[vector_col].tolist())
            cluster_ids = df['cluster_id'].values
            
            # Filter out unclustered articles
            valid_mask = cluster_ids >= 0
            valid_vectors = vectors[valid_mask]
            valid_clusters = cluster_ids[valid_mask]
            
            if len(valid_vectors) < 2:
                logger.warning("Not enough clustered articles to visualize")
                return
                
            # Reduce to 2D for visualization
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            vectors_2d = tsne.fit_transform(valid_vectors)
            
            # Create plot
            plt.figure(figsize=(12, 10))
            
            # Get unique clusters
            unique_clusters = np.unique(valid_clusters)
            
            # Create colormap
            cmap = plt.cm.get_cmap('tab20', len(unique_clusters))
            
            # Plot each cluster
            for i, cluster_id in enumerate(unique_clusters):
                mask = valid_clusters == cluster_id
                plt.scatter(
                    vectors_2d[mask, 0],
                    vectors_2d[mask, 1],
                    c=[cmap(i)],
                    label=f"Cluster {cluster_id}",
                    alpha=0.7
                )
                
            plt.title('News Article Clusters')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved cluster visualization to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing clusters: {str(e)}")
    
    def get_cluster_summary(self, df: pd.DataFrame, 
                           cluster_id: int,
                           title_col: str = 'Title',
                           content_col: str = 'Body',
                           date_col: str = 'Date',
                           impact_col: str = 'impact_overall',
                           max_articles: int = 5) -> Dict[str, Any]:
        """
        Get a summary of a specific cluster.
        
        Args:
            df: DataFrame with cluster assignments
            cluster_id: ID of the cluster to summarize
            title_col: Column name for article titles
            content_col: Column name for article content
            date_col: Column name for article dates
            impact_col: Column name for stock impact scores
            max_articles: Maximum number of articles to include in the summary
            
        Returns:
            Dictionary with cluster summary
        """
        if len(df) == 0 or 'cluster_id' not in df.columns:
            logger.warning("Cannot get cluster summary: no cluster assignments found")
            return {}
            
        # Filter for the specified cluster
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        if len(cluster_df) == 0:
            logger.warning(f"Cluster {cluster_id} not found")
            return {}
            
        try:
            # Get cluster topic
            topic = cluster_df['cluster_topic'].iloc[0] if 'cluster_topic' in cluster_df.columns else f"Cluster {cluster_id}"
            
            # Get cluster size
            size = len(cluster_df)
            
            # Get date range
            if date_col in cluster_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(cluster_df[date_col]):
                    cluster_df[date_col] = pd.to_datetime(cluster_df[date_col])
                    
                start_date = cluster_df[date_col].min()
                end_date = cluster_df[date_col].max()
                date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            else:
                date_range = "Unknown"
                
            # Get average impact
            if impact_col in cluster_df.columns:
                avg_impact = cluster_df[impact_col].mean()
                max_impact = cluster_df[impact_col].max()
                min_impact = cluster_df[impact_col].min()
                impact_info = {
                    'average': avg_impact,
                    'max': max_impact,
                    'min': min_impact
                }
            else:
                impact_info = {}
                
            # Get top articles (by impact if available, otherwise most recent)
            if impact_col in cluster_df.columns and not cluster_df[impact_col].isna().all():
                top_articles = cluster_df.sort_values(impact_col, ascending=False).head(max_articles)
            elif date_col in cluster_df.columns:
                top_articles = cluster_df.sort_values(date_col, ascending=False).head(max_articles)
            else:
                top_articles = cluster_df.head(max_articles)
                
            # Format article info
            articles_info = []
            for _, article in top_articles.iterrows():
                article_info = {}
                
                if title_col in article:
                    article_info['title'] = article[title_col]
                    
                if date_col in article:
                    article_info['date'] = article[date_col]
                    
                if impact_col in article and not pd.isna(article[impact_col]):
                    article_info['impact'] = article[impact_col]
                    
                articles_info.append(article_info)
                
            # Create summary
            summary = {
                'cluster_id': cluster_id,
                'topic': topic,
                'size': size,
                'date_range': date_range,
                'impact': impact_info,
                'top_articles': articles_info
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting cluster summary: {str(e)}")
            return {'cluster_id': cluster_id, 'error': str(e)}
    
    def get_all_clusters_summary(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get summaries for all valid clusters.
        
        Args:
            df: DataFrame with cluster assignments
            
        Returns:
            List of cluster summaries
        """
        if len(df) == 0 or 'cluster_id' not in df.columns:
            logger.warning("Cannot get cluster summaries: no cluster assignments found")
            return []
            
        # Get all valid clusters
        valid_clusters = df[df['cluster_id'] >= 0]['cluster_id'].unique()
        
        # Get summary for each cluster
        summaries = []
        for cluster_id in valid_clusters:
            summary = self.get_cluster_summary(df, cluster_id)
            summaries.append(summary)
            
        # Sort by size (descending)
        summaries.sort(key=lambda x: x.get('size', 0), reverse=True)
        
        return summaries
    
    def find_similar_articles(self, article_vector: np.ndarray, 
                            df: pd.DataFrame,
                            vector_col: str = 'vector',
                            top_n: int = 5) -> pd.DataFrame:
        """
        Find articles similar to a given article vector.
        
        Args:
            article_vector: Vector representation of the query article
            df: DataFrame containing articles with vector representations
            vector_col: Column name for vector representations
            top_n: Number of similar articles to return
            
        Returns:
            DataFrame with similar articles and similarity scores
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for similarity search")
            return pd.DataFrame()
            
        if vector_col not in df.columns:
            logger.error(f"Vector column '{vector_col}' not found in DataFrame")
            return pd.DataFrame()
            
        try:
            # Extract vectors
            vectors = np.array(df[vector_col].tolist())
            
            # Calculate similarities
            similarities = cosine_similarity([article_vector], vectors)[0]
            
            # Add similarity scores to DataFrame
            result_df = df.copy()
            result_df['similarity'] = similarities
            
            # Sort by similarity (descending)
            result_df = result_df.sort_values('similarity', ascending=False)
            
            # Return top N
            return result_df.head(top_n)
            
        except Exception as e:
            logger.error(f"Error finding similar articles: {str(e)}")
            return pd.DataFrame()
    
    def evaluate_clustering(self, df: pd.DataFrame, 
                          vector_col: str = 'vector') -> Dict[str, float]:
        """
        Evaluate clustering quality using various metrics.
        
        Args:
            df: DataFrame with cluster assignments
            vector_col: Column name for vector representations
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(df) == 0 or 'cluster_id' not in df.columns:
            logger.warning("Cannot evaluate clustering: no cluster assignments found")
            return {}
            
        if vector_col not in df.columns:
            logger.error(f"Vector column '{vector_col}' not found in DataFrame")
            return {}
            
        try:
            # Extract vectors and valid cluster assignments
            vectors = np.array(df[vector_col].tolist())
            cluster_ids = df['cluster_id'].values
            
            # Filter out unclustered articles
            valid_mask = cluster_ids >= 0
            valid_vectors = vectors[valid_mask]
            valid_clusters = cluster_ids[valid_mask]
            
            if len(valid_vectors) < 2:
                logger.warning("Not enough clustered articles to evaluate")
                return {}
                
            # Calculate silhouette score
            silhouette = silhouette_score(valid_vectors, valid_clusters, metric='cosine')
            
            # Calculate intra-cluster distances
            intra_cluster_distances = {}
            for cluster_id in np.unique(valid_clusters):
                cluster_vectors = valid_vectors[valid_clusters == cluster_id]
                if len(cluster_vectors) > 1:
                    # Calculate pairwise distances within cluster
                    distances = 1 - cosine_similarity(cluster_vectors)
                    # Take average of upper triangle (excluding diagonal)
                    intra_cluster_distances[cluster_id] = np.mean(distances[np.triu_indices(len(distances), k=1)])
            
            # Calculate average intra-cluster distance
            avg_intra_cluster_distance = np.mean(list(intra_cluster_distances.values()))
            
            # Calculate number of clusters and unclustered articles
            n_clusters = len(np.unique(valid_clusters))
            n_unclustered = np.sum(~valid_mask)
            
            # Calculate percentage of articles in clusters
            clustering_coverage = np.sum(valid_mask) / len(df) * 100
            
            # Return metrics
            metrics = {
                'silhouette_score': silhouette,
                'avg_intra_cluster_distance': avg_intra_cluster_distance,
                'n_clusters': n_clusters,
                'n_unclustered': n_unclustered,
                'clustering_coverage': clustering_coverage
            }
            
            logger.info(f"Clustering evaluation: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating clustering: {str(e)}")
            return {}
