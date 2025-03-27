"""
News recommendation module for recommending news articles based on stock impact.
Adapts NAVER's AiRS recommendation system to focus on stock impact rather than personalization.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsRecommender:
    """
    Recommender for financial news articles based on stock impact.
    Adapts NAVER's AiRS recommendation system to focus on stock impact rather than personalization.
    """
    
    def __init__(self, 
                 weights: Dict[str, float] = None,
                 models_dir: str = None):
        """
        Initialize the news recommender.
        
        Args:
            weights: Dictionary of weights for different recommendation factors
            models_dir: Directory to save/load models
        """
        # Default weights for different recommendation factors
        self.weights = weights or {
            'impact': 0.4,      # Stock Impact (SI) - replaces Social Interest in AiRS
            'quality': 0.2,     # Quality Estimation (QE)
            'content': 0.2,     # Content-Based Filtering (CBF)
            'collaborative': 0.1, # Collaborative Filtering (CF)
            'recency': 0.1      # Latest news prioritization
        }
        
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "recommender")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Initialized NewsRecommender with weights: {self.weights}")
    
    def recommend_articles(self, articles_df: pd.DataFrame, 
                          top_n: int = 10,
                          filter_criteria: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Recommend news articles based on stock impact and other factors.
        
        Args:
            articles_df: DataFrame containing news articles with features
            top_n: Number of articles to recommend
            filter_criteria: Criteria to filter articles
            
        Returns:
            DataFrame with recommended articles
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for recommendations")
            return pd.DataFrame()
            
        logger.info(f"Generating recommendations from {len(articles_df)} articles")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Apply filters if provided
            if filter_criteria:
                df = self._apply_filters(df, filter_criteria)
                logger.info(f"Applied filters, {len(df)} articles remaining")
            
            # Calculate recommendation scores
            df['recommendation_score'] = self._calculate_recommendation_scores(df)
            
            # Sort by recommendation score (descending) and take top_n
            recommended_df = df.sort_values('recommendation_score', ascending=False).head(top_n)
            
            logger.info(f"Generated {len(recommended_df)} recommendations")
            
            # Save recommendations
            self._save_recommendations(recommended_df)
            
            return recommended_df
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def recommend_clusters(self, articles_df: pd.DataFrame,
                          top_n: int = 5,
                          articles_per_cluster: int = 3) -> Dict[int, pd.DataFrame]:
        """
        Recommend news clusters based on stock impact and other factors.
        
        Args:
            articles_df: DataFrame containing news articles with cluster assignments
            top_n: Number of clusters to recommend
            articles_per_cluster: Number of articles to include per cluster
            
        Returns:
            Dictionary mapping cluster IDs to DataFrames with articles
        """
        if len(articles_df) == 0 or 'cluster_id' not in articles_df.columns:
            logger.warning("Invalid DataFrame provided for cluster recommendations")
            return {}
            
        logger.info(f"Generating cluster recommendations from {len(articles_df)} articles")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Filter out articles not in valid clusters
            df = df[df['cluster_id'] >= 0]
            
            if len(df) == 0:
                logger.warning("No articles in valid clusters")
                return {}
            
            # Calculate cluster scores
            cluster_scores = self._calculate_cluster_scores(df)
            
            # Sort clusters by score (descending) and take top_n
            top_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Get top articles for each cluster
            recommendations = {}
            
            for cluster_id, score in top_clusters:
                cluster_df = df[df['cluster_id'] == cluster_id].copy()
                
                # Calculate article scores within cluster
                cluster_df['article_score'] = self._calculate_recommendation_scores(cluster_df)
                
                # Sort by article score (descending) and take top articles
                top_articles = cluster_df.sort_values('article_score', ascending=False).head(articles_per_cluster)
                
                recommendations[cluster_id] = top_articles
            
            logger.info(f"Generated recommendations for {len(recommendations)} clusters")
            
            # Save cluster recommendations
            self._save_cluster_recommendations(recommendations)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating cluster recommendations: {str(e)}")
            raise
    
    def recommend_trending_topics(self, articles_df: pd.DataFrame,
                                top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend trending topics based on news clusters and stock impact.
        
        Args:
            articles_df: DataFrame containing news articles with cluster assignments
            top_n: Number of trending topics to recommend
            
        Returns:
            List of dictionaries with trending topic information
        """
        if len(articles_df) == 0 or 'cluster_id' not in articles_df.columns:
            logger.warning("Invalid DataFrame provided for trending topics")
            return []
            
        logger.info(f"Identifying trending topics from {len(articles_df)} articles")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Filter out articles not in valid clusters
            df = df[df['cluster_id'] >= 0]
            
            if len(df) == 0:
                logger.warning("No articles in valid clusters")
                return []
            
            # Calculate cluster scores with emphasis on recency
            cluster_scores = self._calculate_cluster_scores(df, recency_weight=0.6)
            
            # Sort clusters by score (descending) and take top_n
            top_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Create trending topics
            trending_topics = []
            
            for cluster_id, score in top_clusters:
                cluster_df = df[df['cluster_id'] == cluster_id]
                
                # Get cluster topic
                if 'cluster_topic' in cluster_df.columns and not cluster_df['cluster_topic'].empty:
                    topic = cluster_df['cluster_topic'].iloc[0]
                else:
                    topic = f"Trending Topic {cluster_id}"
                
                # Get representative article
                if 'impact_score' in cluster_df.columns:
                    # Use article with highest impact
                    rep_article = cluster_df.sort_values('impact_score', ascending=False).iloc[0]
                else:
                    # Use most recent article
                    if 'Date' in cluster_df.columns:
                        rep_article = cluster_df.sort_values('Date', ascending=False).iloc[0]
                    else:
                        rep_article = cluster_df.iloc[0]
                
                # Calculate average impact if available
                avg_impact = None
                if 'impact_score' in cluster_df.columns:
                    avg_impact = cluster_df['impact_score'].mean()
                
                # Create topic info
                topic_info = {
                    'cluster_id': cluster_id,
                    'topic': topic,
                    'score': score,
                    'article_count': len(cluster_df),
                    'avg_impact': avg_impact,
                    'representative_article': {
                        'title': rep_article.get('Title', ''),
                        'link': rep_article.get('Link', ''),
                        'date': rep_article.get('Date', None)
                    }
                }
                
                trending_topics.append(topic_info)
            
            logger.info(f"Identified {len(trending_topics)} trending topics")
            
            # Save trending topics
            self._save_trending_topics(trending_topics)
            
            return trending_topics
        except Exception as e:
            logger.error(f"Error identifying trending topics: {str(e)}")
            return []
    
    def _apply_filters(self, df: pd.DataFrame, filter_criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the articles DataFrame.
        
        Args:
            df: DataFrame containing news articles
            filter_criteria: Dictionary of filter criteria
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        try:
            # Filter by date range
            if 'start_date' in filter_criteria and 'Date' in filtered_df.columns:
                start_date = filter_criteria['start_date']
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                filtered_df = filtered_df[filtered_df['Date'] >= start_date]
            
            if 'end_date' in filter_criteria and 'Date' in filtered_df.columns:
                end_date = filter_criteria['end_date']
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                filtered_df = filtered_df[filtered_df['Date'] <= end_date]
            
            # Filter by press/source
            if 'press' in filter_criteria and 'Press' in filtered_df.columns:
                press_list = filter_criteria['press']
                if isinstance(press_list, str):
                    press_list = [press_list]
                filtered_df = filtered_df[filtered_df['Press'].isin(press_list)]
            
            # Filter by minimum impact score
            if 'min_impact' in filter_criteria and 'impact_score' in filtered_df.columns:
                min_impact = filter_criteria['min_impact']
                filtered_df = filtered_df[filtered_df['impact_score'] >= min_impact]
            
            # Filter by cluster
            if 'cluster_id' in filter_criteria and 'cluster_id' in filtered_df.columns:
                cluster_id = filter_criteria['cluster_id']
                filtered_df = filtered_df[filtered_df['cluster_id'] == cluster_id]
            
            # Filter by keywords in title or content
            if 'keywords' in filter_criteria:
                keywords = filter_criteria['keywords']
                if isinstance(keywords, str):
                    keywords = [keywords]
                
                keyword_mask = pd.Series(False, index=filtered_df.index)
                
                for keyword in keywords:
                    if 'Title' in filtered_df.columns:
                        title_mask = filtered_df['Title'].str.contains(keyword, case=False, na=False)
                        keyword_mask = keyword_mask | title_mask
                    
                    if 'Body' in filtered_df.columns:
                        body_mask = filtered_df['Body'].str.contains(keyword, case=False, na=False)
                        keyword_mask = keyword_mask | body_mask
                
                filtered_df = filtered_df[keyword_mask]
            
            return filtered_df
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return df
    
    def _calculate_recommendation_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate recommendation scores for articles.
        
        Args:
            df: DataFrame containing news articles
            
        Returns:
            Series with recommendation scores
        """
        # Initialize scores with zeros
        scores = pd.Series(0.0, index=df.index)
        
        try:
            # Factor 1: Stock Impact (SI)
            # Similar to NAVER's Social Interest (SI) but focused on stock impact
            if 'impact_score' in df.columns:
                # Normalize impact scores to 0-1 range
                impact_scores = df['impact_score'].copy()
                
                # Handle NaN values
                impact_scores = impact_scores.fillna(0)
                
                # Map from -5 to +5 scale to 0-1 scale
                normalized_impact = (impact_scores + 5) / 10
                
                # Add to scores with weight
                scores += normalized_impact * self.weights.get('impact', 0.4)
            
            # Factor 2: Quality Estimation (QE)
            # Use cluster size as a proxy for quality
            if 'cluster_size' in df.columns:
                # Normalize cluster sizes
                max_size = df['cluster_size'].max()
                if max_size > 0:
                    normalized_quality = df['cluster_size'] / max_size
                    scores += normalized_quality * self.weights.get('quality', 0.2)
            
            # Factor 3: Content-Based Filtering (CBF)
            # Use similarity to popular articles
            if 'similarity' in df.columns:
                scores += df['similarity'] * self.weights.get('content', 0.2)
            
            # Factor 4: Collaborative Filtering (CF)
            # Use NPMI scores if available
            if 'npmi_score' in df.columns:
                scores += df['npmi_score'] * self.weights.get('collaborative', 0.1)
            
            # Factor 5: Recency (Latest)
            if 'Date' in df.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                    dates = pd.to_datetime(df['Date'], errors='coerce')
                else:
                    dates = df['Date']
                
                # Calculate recency score (newer is better)
                max_date = dates.max()
                if pd.notna(max_date):
                    date_range = (max_date - dates.min()).total_seconds()
                    if date_range > 0:
                        recency = (dates - dates.min()).dt.total_seconds() / date_range
                        scores += recency * self.weights.get('recency', 0.1)
            
            return scores
        except Exception as e:
            logger.error(f"Error calculating recommendation scores: {str(e)}")
            return pd.Series(0.0, index=df.index)
    
    def _calculate_cluster_scores(self, df: pd.DataFrame, recency_weight: float = None) -> Dict[int, float]:
        """
        Calculate scores for clusters.
        
        Args:
            df: DataFrame containing news articles with cluster assignments
            recency_weight: Optional override for recency weight
            
        Returns:
            Dictionary mapping cluster IDs to scores
        """
        cluster_scores = {}
        
        try:
            # Get unique valid clusters
            clusters = df[df['cluster_id'] >= 0]['cluster_id'].unique()
            
            for cluster_id in clusters:
                cluster_df = df[df['cluster_id'] == cluster_id]
                
                # Base score is the cluster size
                size_score = len(cluster_df) / len(df)
                
                # Impact score (if available)
                impact_score = 0.0
                if 'impact_score' in cluster_df.columns:
                    avg_impact = cluster_df['impact_score'].mean()
                    # Normalize from -5 to +5 scale to 0-1 scale
                    impact_score = (avg_impact + 5) / 10
                
                # Recency score
                recency_score = 0.0
                if 'Date' in cluster_df.columns:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(cluster_df['Date']):
                        dates = pd.to_datetime(cluster_df['Date'], errors='coerce')
                    else:
                        dates = cluster_df['Date']
                    
                    # Calculate average date
                    avg_date = dates.mean()
                    max_date = df['Date'].max()
                    min_date = df['Date'].min()
                    
                    if pd.notna(avg_date) and pd.notna(max_date) and pd.notna(min_date):
                        date_range = (max_date - min_date).total_seconds()
                        if date_range > 0:
                            recency_score = (avg_date - min_date).total_seconds() / date_range
                
                # Calculate final score
                # Use provided recency weight or default from self.weights
                rec_weight = recency_weight if recency_weight is not None else self.weights.get('recency', 0.1)
                
                # Adjust other weights proportionally
                impact_weight = self.weights.get('impact', 0.4) * (1 - rec_weight) / (1 - self.weights.get('recency', 0.1))
                size_weight = 1 - impact_weight - rec_weight
                
                final_score = (
                    size_weight * size_score +
                    impact_weight * impact_score +
                    rec_weight * recency_score
                )
                
                cluster_scores[cluster_id] = final_score
            
            return cluster_scores
        except Exception as e:
            logger.error(f"Error calculating cluster scores: {str(e)}")
            return {}
    
    def _save_recommendations(self, df: pd.DataFrame) -> None:
        """
        Save recommendations for later analysis.
        
        Args:
            df: DataFrame with recommended articles
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save recommendations
            recommendations_path = os.path.join(self.models_dir, f"recommendations_{timestamp}.csv")
            df.to_csv(recommendations_path, index=False)
            
            logger.info(f"Saved recommendations to {recommendations_path}")
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
    
    def _save_cluster_recommendations(self, recommendations: Dict[int, pd.DataFrame]) -> None:
        """
        Save cluster recommendations for later analysis.
        
        Args:
            recommendations: Dictionary mapping cluster IDs to DataFrames with articles
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save cluster recommendations
            for cluster_id, df in recommendations.items():
                cluster_path = os.path.join(self.models_dir, f"cluster_{cluster_id}_recommendations_{timestamp}.csv")
                df.to_csv(cluster_path, index=False)
            
            logger.info(f"Saved recommendations for {len(recommendations)} clusters")
        except Exception as e:
            logger.error(f"Error saving cluster recommendations: {str(e)}")
    
    def _save_trending_topics(self, trending_topics: List[Dict[str, Any]]) -> None:
        """
        Save trending topics for later analysis.
        
        Args:
            trending_topics: List of dictionaries with trending topic information
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Convert to DataFrame
            topics_df = pd.DataFrame(trending_topics)
            
            # Save trending topics
            topics_path = os.path.join(self.models_dir, f"trending_topics_{timestamp}.csv")
            topics_df.to_csv(topics_path, index=False)
            
            logger.info(f"Saved {len(trending_topics)} trending topics to {topics_path}")
        except Exception as e:
            logger.error(f"Error saving trending topics: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = {
        'Title': [f"Article {i}" for i in range(20)],
        'Date': pd.date_range(start='2025-01-01', periods=20),
        'cluster_id': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, -1, -1, -1],
        'cluster_size': [3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 1, 1, 1],
        'impact_score': np.random.uniform(-5, 5, 20),
        'cluster_topic': ['Topic A', 'Topic A', 'Topic A', 
                         'Topic B', 'Topic B', 'Topic B', 'Topic B',
                         'Topic C', 'Topic C', 'Topic C',
                         'Topic D', 'Topic D', 'Topic D', 'Topic D',
                         'Topic E', 'Topic E', 'Topic E',
                         '', '', '']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create recommender
    recommender = NewsRecommender()
    
    # Get recommendations
    recommendations = recommender.recommend_articles(df, top_n=5)
    print(f"Top 5 recommended articles:\n{recommendations[['Title', 'impact_score', 'recommendation_score']]}")
    
    # Get cluster recommendations
    cluster_recommendations = recommender.recommend_clusters(df, top_n=3)
    print(f"\nRecommended clusters: {list(cluster_recommendations.keys())}")
    
    # Get trending topics
    trending_topics = recommender.recommend_trending_topics(df, top_n=3)
    print(f"\nTrending topics:")
    for topic in trending_topics:
        print(f"  {topic['topic']} (Score: {topic['score']:.4f}, Articles: {topic['article_count']})")
