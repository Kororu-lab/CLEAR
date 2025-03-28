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
                 models_dir: str = None,
                 use_advanced_features: bool = False,
                 recency_decay_days: float = 1.0,
                 min_quality_threshold: float = 0.3,
                 stock_tickers: List[str] = None):
        """
        Initialize the news recommender.
        
        Args:
            weights: Dictionary of weights for different recommendation factors
            models_dir: Directory to save/load models
            use_advanced_features: Whether to use advanced recommendation features
            recency_decay_days: Days parameter for recency score decay
            min_quality_threshold: Minimum quality threshold for recommendations
            stock_tickers: List of stock tickers to focus on
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
        self.use_advanced_features = use_advanced_features
        self.recency_decay_days = recency_decay_days
        self.min_quality_threshold = min_quality_threshold
        self.stock_tickers = stock_tickers or []
        
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
                topic = cluster_df['cluster_topic'].iloc[0] if 'cluster_topic' in cluster_df.columns else f"Cluster {cluster_id}"
                
                # Get representative article
                if 'impact_overall' in cluster_df.columns:
                    # Use article with highest impact
                    representative = cluster_df.loc[cluster_df['impact_overall'].idxmax()]
                else:
                    # Use most recent article
                    if 'Date' in cluster_df.columns:
                        representative = cluster_df.sort_values('Date', ascending=False).iloc[0]
                    else:
                        representative = cluster_df.iloc[0]
                
                # Create topic info
                topic_info = {
                    'cluster_id': cluster_id,
                    'topic': topic,
                    'score': score,
                    'size': len(cluster_df),
                    'representative_article': {
                        'title': representative.get('Title', ''),
                        'date': representative.get('Date', ''),
                        'impact': representative.get('impact_overall', 0)
                    }
                }
                
                trending_topics.append(topic_info)
            
            logger.info(f"Identified {len(trending_topics)} trending topics")
            
            return trending_topics
        except Exception as e:
            logger.error(f"Error identifying trending topics: {str(e)}")
            return []
    
    def recommend_stock_specific(self, articles_df: pd.DataFrame,
                               ticker: str,
                               top_n: int = 5) -> pd.DataFrame:
        """
        Recommend articles specific to a stock ticker.
        
        Args:
            articles_df: DataFrame containing news articles
            ticker: Stock ticker to focus on
            top_n: Number of articles to recommend
            
        Returns:
            DataFrame with recommended articles
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for stock-specific recommendations")
            return pd.DataFrame()
            
        logger.info(f"Generating stock-specific recommendations for {ticker}")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Filter for articles mentioning the ticker
            ticker_df = self._filter_by_ticker(df, ticker)
            
            if len(ticker_df) == 0:
                logger.warning(f"No articles found for ticker {ticker}")
                return pd.DataFrame()
            
            # Calculate recommendation scores
            ticker_df['recommendation_score'] = self._calculate_recommendation_scores(ticker_df)
            
            # Sort by recommendation score (descending) and take top_n
            recommended_df = ticker_df.sort_values('recommendation_score', ascending=False).head(top_n)
            
            logger.info(f"Generated {len(recommended_df)} recommendations for {ticker}")
            
            return recommended_df
        except Exception as e:
            logger.error(f"Error generating stock-specific recommendations: {str(e)}")
            return pd.DataFrame()
    
    def recommend_by_impact(self, articles_df: pd.DataFrame,
                          impact_type: str = 'positive',
                          top_n: int = 5) -> pd.DataFrame:
        """
        Recommend articles by impact type (positive/negative).
        
        Args:
            articles_df: DataFrame containing news articles with impact scores
            impact_type: Type of impact to focus on ('positive', 'negative', 'high_impact')
            top_n: Number of articles to recommend
            
        Returns:
            DataFrame with recommended articles
        """
        if len(articles_df) == 0 or 'impact_overall' not in articles_df.columns:
            logger.warning("Invalid DataFrame provided for impact-based recommendations")
            return pd.DataFrame()
            
        logger.info(f"Generating {impact_type} impact recommendations")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Filter by impact type
            if impact_type == 'positive':
                filtered_df = df[df['impact_overall'] > 0]
                sort_ascending = False  # Higher is better
            elif impact_type == 'negative':
                filtered_df = df[df['impact_overall'] < 0]
                sort_ascending = True  # Lower is worse (more negative)
            elif impact_type == 'high_impact':
                filtered_df = df.copy()
                filtered_df['impact_overall'] = filtered_df['impact_overall'].abs()
                sort_ascending = False  # Higher absolute impact
            else:
                logger.warning(f"Unknown impact type: {impact_type}")
                return pd.DataFrame()
            
            if len(filtered_df) == 0:
                logger.warning(f"No articles found for impact type {impact_type}")
                return pd.DataFrame()
            
            # Sort by impact and take top_n
            recommended_df = filtered_df.sort_values('impact_overall', ascending=sort_ascending).head(top_n)
            
            logger.info(f"Generated {len(recommended_df)} {impact_type} impact recommendations")
            
            return recommended_df
        except Exception as e:
            logger.error(f"Error generating impact-based recommendations: {str(e)}")
            return pd.DataFrame()
    
    def _apply_filters(self, df: pd.DataFrame, 
                     filter_criteria: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the articles DataFrame.
        
        Args:
            df: DataFrame to filter
            filter_criteria: Dictionary of filter criteria
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        try:
            # Filter by date range
            if 'start_date' in filter_criteria and 'Date' in filtered_df.columns:
                start_date = filter_criteria['start_date']
                if not isinstance(start_date, datetime):
                    start_date = pd.to_datetime(start_date)
                filtered_df = filtered_df[filtered_df['Date'] >= start_date]
            
            if 'end_date' in filter_criteria and 'Date' in filtered_df.columns:
                end_date = filter_criteria['end_date']
                if not isinstance(end_date, datetime):
                    end_date = pd.to_datetime(end_date)
                filtered_df = filtered_df[filtered_df['Date'] <= end_date]
            
            # Filter by press/source
            if 'press' in filter_criteria and 'Press' in filtered_df.columns:
                press_list = filter_criteria['press']
                if isinstance(press_list, str):
                    press_list = [press_list]
                filtered_df = filtered_df[filtered_df['Press'].isin(press_list)]
            
            # Filter by minimum impact
            if 'min_impact' in filter_criteria and 'impact_overall' in filtered_df.columns:
                min_impact = filter_criteria['min_impact']
                filtered_df = filtered_df[filtered_df['impact_overall'] >= min_impact]
            
            # Filter by cluster
            if 'cluster_id' in filter_criteria and 'cluster_id' in filtered_df.columns:
                cluster_id = filter_criteria['cluster_id']
                filtered_df = filtered_df[filtered_df['cluster_id'] == cluster_id]
            
            # Filter by ticker
            if 'ticker' in filter_criteria:
                ticker = filter_criteria['ticker']
                filtered_df = self._filter_by_ticker(filtered_df, ticker)
            
            # Filter by minimum quality
            if 'min_quality' in filter_criteria and 'quality_score' in filtered_df.columns:
                min_quality = filter_criteria['min_quality']
                filtered_df = filtered_df[filtered_df['quality_score'] >= min_quality]
            
            return filtered_df
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return df
    
    def _filter_by_ticker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Filter articles by stock ticker.
        
        Args:
            df: DataFrame to filter
            ticker: Stock ticker to filter by
            
        Returns:
            Filtered DataFrame
        """
        try:
            # Check if tickers column exists
            if 'tickers' in df.columns:
                # Filter articles that mention the ticker
                ticker_df = df[df['tickers'].apply(
                    lambda x: ticker in x if isinstance(x, list) else 
                    (ticker in eval(x) if isinstance(x, str) and x.startswith('[') else 
                     ticker == x)
                )]
                
                if len(ticker_df) > 0:
                    return ticker_df
            
            # Check if Title contains ticker or company name
            if 'Title' in df.columns:
                # Map ticker to company name (simplified)
                company_map = {
                    '005930': '삼성전자',
                    '000660': 'SK하이닉스',
                    '035420': 'NAVER',
                    '035720': '카카오',
                    '005380': '현대자동차',
                    '000270': '기아',
                    '051910': 'LG화학',
                    '068270': '셀트리온'
                }
                
                company_name = company_map.get(ticker, ticker)
                
                # Filter by title
                title_df = df[df['Title'].str.contains(ticker, na=False) | 
                             df['Title'].str.contains(company_name, na=False)]
                
                if len(title_df) > 0:
                    return title_df
                
                # Check Body as fallback
                if 'Body' in df.columns:
                    body_df = df[df['Body'].str.contains(ticker, na=False) | 
                                df['Body'].str.contains(company_name, na=False)]
                    
                    if len(body_df) > 0:
                        return body_df
            
            # No matches found
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error filtering by ticker: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_recommendation_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate recommendation scores for articles.
        
        Args:
            df: DataFrame containing articles
            
        Returns:
            Series with recommendation scores
        """
        try:
            # Initialize scores with zeros
            scores = pd.Series(0.0, index=df.index)
            
            # Stock Impact (SI) score
            if 'impact_overall' in df.columns:
                # Normalize impact scores to 0-1 range
                impact_scores = df['impact_overall'].copy()
                if not impact_scores.isna().all():
                    # Handle NaN values
                    impact_scores = impact_scores.fillna(0)
                    
                    # Convert from -5 to +5 scale to 0-1 scale
                    impact_scores = (impact_scores + 5) / 10
                    
                    # Add to total score
                    scores += self.weights['impact'] * impact_scores
            
            # Quality Estimation (QE) score
            if 'quality_score' in df.columns:
                quality_scores = df['quality_score'].copy()
                if not quality_scores.isna().all():
                    # Handle NaN values
                    quality_scores = quality_scores.fillna(self.min_quality_threshold)
                    
                    # Add to total score
                    scores += self.weights['quality'] * quality_scores
            elif 'Num_comment' in df.columns:
                # Use comment count as a proxy for quality/engagement
                comment_counts = df['Num_comment'].copy()
                if not comment_counts.isna().all():
                    # Handle NaN values
                    comment_counts = comment_counts.fillna(0)
                    
                    # Normalize to 0-1 range
                    max_comments = comment_counts.max()
                    if max_comments > 0:
                        normalized_comments = comment_counts / max_comments
                        scores += self.weights['quality'] * normalized_comments
            
            # Content-Based Filtering (CBF) score
            # This would typically use similarity to user preferences
            # Since we're not using personalization, we'll use cluster quality instead
            if 'cluster_id' in df.columns and 'silhouette_score' in df.columns:
                silhouette_scores = df['silhouette_score'].copy()
                if not silhouette_scores.isna().all():
                    # Handle NaN values
                    silhouette_scores = silhouette_scores.fillna(0)
                    
                    # Normalize to 0-1 range (silhouette is already -1 to 1)
                    normalized_silhouette = (silhouette_scores + 1) / 2
                    
                    # Add to total score
                    scores += self.weights['content'] * normalized_silhouette
            
            # Collaborative Filtering (CF) score
            # This would typically use user behavior patterns
            # Since we're not using personalization, we'll use cluster size as a proxy
            if 'cluster_size' in df.columns:
                cluster_sizes = df['cluster_size'].copy()
                if not cluster_sizes.isna().all():
                    # Handle NaN values
                    cluster_sizes = cluster_sizes.fillna(1)
                    
                    # Normalize to 0-1 range
                    max_size = cluster_sizes.max()
                    if max_size > 1:
                        normalized_sizes = cluster_sizes / max_size
                        scores += self.weights['collaborative'] * normalized_sizes
            
            # Recency score
            if 'Date' in df.columns:
                dates = df['Date'].copy()
                if not dates.isna().all():
                    # Ensure dates are datetime
                    if not pd.api.types.is_datetime64_any_dtype(dates):
                        dates = pd.to_datetime(dates)
                    
                    # Calculate recency score based on days from most recent
                    most_recent = dates.max()
                    days_old = (most_recent - dates).dt.total_seconds() / (24 * 3600)
                    
                    # Apply exponential decay
                    recency_scores = np.exp(-days_old / self.recency_decay_days)
                    
                    # Add to total score
                    scores += self.weights['recency'] * recency_scores
            
            # Advanced features if enabled
            if self.use_advanced_features:
                # Sentiment analysis score if available
                if 'sentiment_score' in df.columns:
                    sentiment_scores = df['sentiment_score'].copy()
                    if not sentiment_scores.isna().all():
                        # Handle NaN values
                        sentiment_scores = sentiment_scores.fillna(0)
                        
                        # Normalize to 0-1 range (assuming -1 to 1 scale)
                        normalized_sentiment = (sentiment_scores + 1) / 2
                        
                        # Add to total score with a small weight
                        scores += 0.1 * normalized_sentiment
                
                # Use AI Summary presence as a quality indicator
                if 'AI Summary' in df.columns:
                    has_summary = df['AI Summary'].notna() & (df['AI Summary'] != '')
                    scores += 0.05 * has_summary.astype(float)
            
            return scores
        except Exception as e:
            logger.error(f"Error calculating recommendation scores: {str(e)}")
            return pd.Series(0.0, index=df.index)
    
    def _calculate_cluster_scores(self, df: pd.DataFrame, 
                                recency_weight: float = None) -> Dict[int, float]:
        """
        Calculate scores for clusters.
        
        Args:
            df: DataFrame containing articles with cluster assignments
            recency_weight: Optional override for recency weight
            
        Returns:
            Dictionary mapping cluster IDs to scores
        """
        try:
            # Get unique clusters
            clusters = df['cluster_id'].unique()
            
            # Initialize scores
            cluster_scores = {}
            
            # Use provided recency weight or default from weights
            recency_weight = recency_weight or self.weights['recency']
            
            # Adjust other weights proportionally
            total_other_weight = sum(w for k, w in self.weights.items() if k != 'recency')
            if total_other_weight > 0:
                weight_factor = (1 - recency_weight) / total_other_weight
            else:
                weight_factor = 0
                
            adjusted_weights = {
                k: (w * weight_factor if k != 'recency' else recency_weight) 
                for k, w in self.weights.items()
            }
            
            # Calculate score for each cluster
            for cluster_id in clusters:
                cluster_df = df[df['cluster_id'] == cluster_id]
                
                # Initialize cluster score
                score = 0.0
                
                # Size component
                size_score = len(cluster_df) / len(df)
                score += 0.1 * size_score  # Small weight for size
                
                # Impact component
                if 'impact_overall' in cluster_df.columns:
                    # Use average absolute impact
                    avg_impact = cluster_df['impact_overall'].abs().mean()
                    if not pd.isna(avg_impact):
                        # Normalize to 0-1 scale
                        impact_score = avg_impact / 5  # Assuming -5 to +5 scale
                        score += adjusted_weights['impact'] * impact_score
                
                # Recency component
                if 'Date' in cluster_df.columns:
                    dates = cluster_df['Date']
                    if not dates.isna().all():
                        # Ensure dates are datetime
                        if not pd.api.types.is_datetime64_any_dtype(dates):
                            dates = pd.to_datetime(dates)
                        
                        # Use most recent article in cluster
                        most_recent = dates.max()
                        most_recent_overall = df['Date'].max()
                        
                        # Calculate days from most recent overall
                        days_old = (most_recent_overall - most_recent).total_seconds() / (24 * 3600)
                        
                        # Apply exponential decay
                        recency_score = np.exp(-days_old / self.recency_decay_days)
                        
                        score += adjusted_weights['recency'] * recency_score
                
                # Quality component
                if 'quality_score' in cluster_df.columns:
                    avg_quality = cluster_df['quality_score'].mean()
                    if not pd.isna(avg_quality):
                        score += adjusted_weights['quality'] * avg_quality
                
                # Store cluster score
                cluster_scores[cluster_id] = score
            
            return cluster_scores
        except Exception as e:
            logger.error(f"Error calculating cluster scores: {str(e)}")
            return {}
    
    def _save_recommendations(self, df: pd.DataFrame) -> None:
        """
        Save recommendations to disk.
        
        Args:
            df: DataFrame with recommended articles
        """
        try:
            # Save recommendations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recommendations_path = os.path.join(self.models_dir, f"recommendations_{timestamp}.joblib")
            
            joblib.dump(df, recommendations_path)
            logger.info(f"Saved recommendations to {recommendations_path}")
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
    
    def _save_cluster_recommendations(self, recommendations: Dict[int, pd.DataFrame]) -> None:
        """
        Save cluster recommendations to disk.
        
        Args:
            recommendations: Dictionary mapping cluster IDs to DataFrames with articles
        """
        try:
            # Save cluster recommendations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recommendations_path = os.path.join(self.models_dir, f"cluster_recommendations_{timestamp}.joblib")
            
            joblib.dump(recommendations, recommendations_path)
            logger.info(f"Saved cluster recommendations to {recommendations_path}")
            
        except Exception as e:
            logger.error(f"Error saving cluster recommendations: {str(e)}")
    
    def calculate_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quality scores for articles.
        
        Args:
            df: DataFrame containing articles
            
        Returns:
            DataFrame with added quality scores
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for quality scoring")
            return df
            
        logger.info(f"Calculating quality scores for {len(df)} articles")
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Initialize quality scores
            result_df['quality_score'] = 0.0
            
            # Component 1: Comment count (if available)
            if 'Num_comment' in result_df.columns:
                comment_counts = result_df['Num_comment'].fillna(0)
                max_comments = max(1, comment_counts.max())
                normalized_comments = comment_counts / max_comments
                result_df['quality_score'] += 0.3 * normalized_comments
            
            # Component 2: Content length (if available)
            if 'Body' in result_df.columns:
                content_lengths = result_df['Body'].fillna('').apply(len)
                max_length = max(1, content_lengths.max())
                normalized_lengths = content_lengths / max_length
                result_df['quality_score'] += 0.2 * normalized_lengths
            
            # Component 3: Has AI Summary
            if 'AI Summary' in result_df.columns:
                has_summary = result_df['AI Summary'].notna() & (result_df['AI Summary'] != '')
                result_df['quality_score'] += 0.2 * has_summary.astype(float)
            
            # Component 4: Source/Press quality (if available)
            if 'Press' in result_df.columns:
                # Define quality scores for known sources
                press_quality = {
                    '연합뉴스': 0.9,
                    '한국경제': 0.85,
                    '매일경제': 0.85,
                    '조선비즈': 0.8,
                    '중앙일보': 0.8,
                    '한겨레': 0.8,
                    '동아일보': 0.8,
                    '서울경제': 0.75
                }
                
                # Map press to quality scores
                result_df['press_quality'] = result_df['Press'].map(
                    lambda x: press_quality.get(x, 0.5)
                )
                
                result_df['quality_score'] += 0.3 * result_df['press_quality']
                
                # Remove temporary column
                result_df = result_df.drop('press_quality', axis=1)
            
            # Ensure scores are in 0-1 range
            result_df['quality_score'] = result_df['quality_score'].clip(0, 1)
            
            logger.info(f"Calculated quality scores for {len(result_df)} articles")
            
            return result_df
        except Exception as e:
            logger.error(f"Error calculating quality scores: {str(e)}")
            return df
    
    def generate_daily_digest(self, articles_df: pd.DataFrame,
                            date: Optional[datetime] = None,
                            top_n: int = 10) -> Dict[str, Any]:
        """
        Generate a daily digest of news articles.
        
        Args:
            articles_df: DataFrame containing news articles
            date: Date for the digest (defaults to today)
            top_n: Number of articles to include
            
        Returns:
            Dictionary with digest information
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for daily digest")
            return {'date': date or datetime.now(), 'articles': []}
            
        logger.info(f"Generating daily digest for {date or 'today'}")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Filter by date if provided
            if date is not None:
                if 'Date' in df.columns:
                    # Ensure dates are datetime
                    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                        df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Filter for the specified date
                    date_start = pd.Timestamp(date.year, date.month, date.day)
                    date_end = date_start + timedelta(days=1)
                    df = df[(df['Date'] >= date_start) & (df['Date'] < date_end)]
            
            if len(df) == 0:
                logger.warning(f"No articles found for date {date}")
                return {'date': date or datetime.now(), 'articles': []}
            
            # Calculate recommendation scores
            df['recommendation_score'] = self._calculate_recommendation_scores(df)
            
            # Get top articles
            top_articles = df.sort_values('recommendation_score', ascending=False).head(top_n)
            
            # Format articles for digest
            articles_info = []
            for _, article in top_articles.iterrows():
                article_info = {
                    'title': article.get('Title', ''),
                    'press': article.get('Press', ''),
                    'date': article.get('Date', ''),
                    'link': article.get('Link', ''),
                    'impact': article.get('impact_overall', 0),
                    'cluster_id': article.get('cluster_id', -1),
                    'cluster_topic': article.get('cluster_topic', '')
                }
                
                articles_info.append(article_info)
            
            # Create digest
            digest = {
                'date': date or datetime.now(),
                'articles_count': len(df),
                'articles': articles_info
            }
            
            # Get trending topics if clustering is available
            if 'cluster_id' in df.columns:
                trending_topics = self.recommend_trending_topics(df, top_n=3)
                digest['trending_topics'] = trending_topics
            
            logger.info(f"Generated daily digest with {len(articles_info)} articles")
            
            return digest
        except Exception as e:
            logger.error(f"Error generating daily digest: {str(e)}")
            return {'date': date or datetime.now(), 'articles': []}
