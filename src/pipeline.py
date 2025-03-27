"""
Main pipeline module for integrating all components of the CLEAR system.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import argparse
import yaml
import json
import schedule
import time
import torch

# Import CLEAR modules
from src.data.news_crawler import NewsCrawler
from src.data.stock_data_collector import StockDataCollector
from src.data.text_preprocessor import TextPreprocessor
from src.models.news_vectorizer import NewsVectorizer
from src.models.news_clustering import NewsClustering
from src.models.news_recommender import NewsRecommender
from src.models.stock_impact_analyzer import StockImpactAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clear_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CLEARPipeline:
    """
    Main pipeline for the CLEAR system.
    Integrates all components for news clustering and stock impact-based recommendations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the CLEAR pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize data directories
        self.data_dir = self.config.get('data_dir', 'data')
        self.news_dir = os.path.join(self.data_dir, 'news')
        self.stock_dir = os.path.join(self.data_dir, 'stock')
        self.models_dir = os.path.join(self.config.get('models_dir', 'models'))
        self.results_dir = os.path.join(self.config.get('results_dir', 'results'))
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.news_dir, self.stock_dir, 
                         self.models_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        logger.info("Initialized CLEAR pipeline")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary with configuration parameters
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.info("Using default configuration")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Dictionary with default configuration parameters
        """
        return {
            'data_dir': 'data',
            'models_dir': 'models',
            'results_dir': 'results',
            'use_gpu': True,
            'stock_tickers': ['005930'],  # Samsung Electronics
            'news_crawler': {
                'source': 'yna',
                'use_ai_summary': True
            },
            'text_preprocessor': {
                'use_mecab': True,
                'remove_stopwords': True
            },
            'news_vectorizer': {
                'method': 'tfidf',
                'max_features': 10000,
                'title_weight': 2.0
            },
            'news_clustering': {
                'distance_threshold': 0.7,
                'min_cluster_size': 3
            },
            'stock_impact': {
                'time_windows': [
                    {"name": "immediate", "hours": 1},
                    {"name": "short_term", "hours": 24},
                    {"name": "medium_term", "days": 3}
                ]
            },
            'news_recommender': {
                'weights': {
                    'impact': 0.4,
                    'quality': 0.2,
                    'content': 0.2,
                    'collaborative': 0.1,
                    'recency': 0.1
                }
            },
            'schedule': {
                'market_open': '09:00',
                'market_close': '15:30'
            }
        }
    
    def _init_components(self) -> None:
        """
        Initialize all components of the CLEAR system.
        """
        try:
            # Initialize news crawler
            news_config = self.config.get('news_crawler', {})
            self.news_crawler = NewsCrawler(
                source=news_config.get('source', 'yna'),
                use_ai_summary=news_config.get('use_ai_summary', True)
            )
            
            # Initialize stock data collector
            stock_config = self.config.get('stock_data_collector', {})
            self.stock_collector = StockDataCollector(
                data_dir=self.stock_dir
            )
            
            # Initialize text preprocessor
            preprocessor_config = self.config.get('text_preprocessor', {})
            self.text_preprocessor = TextPreprocessor(
                language=preprocessor_config.get('language', 'ko'),
                use_mecab=preprocessor_config.get('use_mecab', True),
                remove_stopwords=preprocessor_config.get('remove_stopwords', True),
                custom_stopwords=preprocessor_config.get('custom_stopwords', None),
                min_token_length=preprocessor_config.get('min_token_length', 2)
            )
            
            # Initialize news vectorizer
            vectorizer_config = self.config.get('news_vectorizer', {})
            self.news_vectorizer = NewsVectorizer(
                method=vectorizer_config.get('method', 'tfidf'),
                max_features=vectorizer_config.get('max_features', 10000),
                embedding_dim=vectorizer_config.get('embedding_dim', 300),
                use_gpu=self.config.get('use_gpu', True),
                title_weight=vectorizer_config.get('title_weight', 2.0),
                models_dir=os.path.join(self.models_dir, 'vectorizers')
            )
            
            # Initialize news clustering
            clustering_config = self.config.get('news_clustering', {})
            self.news_clustering = NewsClustering(
                distance_threshold=clustering_config.get('distance_threshold', 0.7),
                min_cluster_size=clustering_config.get('min_cluster_size', 3),
                max_cluster_size=clustering_config.get('max_cluster_size', 20),
                linkage=clustering_config.get('linkage', 'average'),
                models_dir=os.path.join(self.models_dir, 'clustering')
            )
            
            # Initialize stock impact analyzer
            impact_config = self.config.get('stock_impact', {})
            self.impact_analyzer = StockImpactAnalyzer(
                time_windows=impact_config.get('time_windows', None),
                impact_thresholds=impact_config.get('impact_thresholds', None),
                use_gpu=self.config.get('use_gpu', True),
                models_dir=os.path.join(self.models_dir, 'stock_impact')
            )
            
            # Initialize news recommender
            recommender_config = self.config.get('news_recommender', {})
            self.news_recommender = NewsRecommender(
                weights=recommender_config.get('weights', None),
                models_dir=os.path.join(self.models_dir, 'recommender')
            )
            
            logger.info("Initialized all CLEAR components")
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def load_data(self, use_existing: bool = True, 
                 start_date: str = None, end_date: str = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load news and stock data.
        
        Args:
            use_existing: Whether to use existing data files
            start_date: Start date for data collection (format: YYYYMMDD)
            end_date: End date for data collection (format: YYYYMMDD)
            
        Returns:
            Tuple of (news_df, stock_data_dict)
        """
        logger.info("Loading data...")
        
        # Load stock data
        stock_data = {}
        stock_tickers = self.config.get('stock_tickers', ['005930'])
        
        for ticker in stock_tickers:
            if use_existing:
                # Load from existing files
                stock_file = os.path.join(self.stock_dir, f"stockprice_{ticker}.csv")
                if os.path.exists(stock_file):
                    stock_df = pd.read_csv(stock_file)
                    logger.info(f"Loaded stock data for {ticker} from {stock_file}")
                else:
                    logger.warning(f"Stock file not found: {stock_file}")
                    stock_df = self.stock_collector.collect_stock_data(ticker, start_date, end_date)
            else:
                # Collect new data
                stock_df = self.stock_collector.collect_stock_data(ticker, start_date, end_date)
            
            stock_data[ticker] = stock_df
        
        # Load news data
        news_df = pd.DataFrame()
        
        if use_existing:
            # Load from existing files
            news_files = []
            for ticker in stock_tickers:
                news_file = os.path.join(self.news_dir, f"yna_{ticker}_2025.csv")
                if os.path.exists(news_file):
                    news_files.append(news_file)
                else:
                    logger.warning(f"News file not found: {news_file}")
            
            if news_files:
                # Concatenate all news files
                dfs = []
                for file in news_files:
                    df = pd.read_csv(file)
                    dfs.append(df)
                
                if dfs:
                    news_df = pd.concat(dfs, ignore_index=True)
                    logger.info(f"Loaded {len(news_df)} news articles from {len(news_files)} files")
            else:
                logger.warning("No news files found")
        else:
            # Collect new data
            for ticker in stock_tickers:
                ticker_news = self.news_crawler.crawl_by_stock_code(ticker, start_date, end_date)
                if not news_df.empty:
                    news_df = pd.concat([news_df, ticker_news], ignore_index=True)
                else:
                    news_df = ticker_news
        
        logger.info(f"Loaded {len(news_df)} news articles and stock data for {len(stock_data)} tickers")
        return news_df, stock_data
    
    def preprocess_data(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess news data.
        
        Args:
            news_df: DataFrame containing news articles
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing {len(news_df)} news articles")
        
        try:
            # Apply text preprocessing
            use_summary = self.config.get('news_crawler', {}).get('use_ai_summary', True)
            preprocessed_df = self.text_preprocessor.preprocess_dataframe(
                news_df,
                title_col='Title',
                body_col='Body',
                summary_col='AI Summary',
                use_summary=use_summary
            )
            
            # Extract keywords
            preprocessed_df = self.text_preprocessor.extract_keywords_from_df(
                preprocessed_df,
                text_col='processed_content',
                top_n=10
            )
            
            logger.info(f"Completed preprocessing for {len(preprocessed_df)} articles")
            return preprocessed_df
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def vectorize_articles(self, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorize preprocessed news articles.
        
        Args:
            preprocessed_df: DataFrame containing preprocessed news articles
            
        Returns:
            DataFrame with vector representations
        """
        logger.info(f"Vectorizing {len(preprocessed_df)} articles")
        
        try:
            # Apply vectorization
            vectorizer_config = self.config.get('news_vectorizer', {})
            vectorized_df = self.news_vectorizer.vectorize_articles(
                preprocessed_df,
                content_col='processed_content',
                title_col='processed_title',
                combine_title_content=vectorizer_config.get('combine_title_content', True),
                reduce_dims=vectorizer_config.get('reduce_dims', True),
                n_components=vectorizer_config.get('n_components', 100)
            )
            
            logger.info(f"Completed vectorization for {len(vectorized_df)} articles")
            return vectorized_df
        except Exception as e:
            logger.error(f"Error vectorizing articles: {str(e)}")
            raise
    
    def cluster_articles(self, vectorized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster vectorized news articles.
        
        Args:
            vectorized_df: DataFrame containing vectorized news articles
            
        Returns:
            DataFrame with cluster assignments
        """
        logger.info(f"Clustering {len(vectorized_df)} articles")
        
        try:
            # Apply clustering
            clustered_df = self.news_clustering.cluster_articles(
                vectorized_df,
                vector_col='vector',
                title_col='Title',
                content_col='Body'
            )
            
            # Get trending clusters
            trending_clusters = self.news_clustering.get_trending_clusters(
                clustered_df,
                timeframe_hours=24,
                min_articles=3
            )
            
            logger.info(f"Created {len(clustered_df['cluster_id'].unique())} clusters, {len(trending_clusters)} trending")
            return clustered_df
        except Exception as e:
            logger.error(f"Error clustering articles: {str(e)}")
            raise
    
    def analyze_impact(self, clustered_df: pd.DataFrame, 
                      stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze the impact of news on stock prices.
        
        Args:
            clustered_df: DataFrame containing clustered news articles
            stock_data: Dictionary mapping tickers to stock price DataFrames
            
        Returns:
            DataFrame with impact scores
        """
        logger.info(f"Analyzing impact for {len(clustered_df)} articles")
        
        try:
            # Apply impact analysis
            impact_df = self.impact_analyzer.analyze_news_impact(
                clustered_df,
                stock_data
            )
            
            # Train impact model if enough data
            if len(impact_df.dropna(subset=['impact_score'])) >= 20:
                impact_config = self.config.get('stock_impact', {})
                self.impact_analyzer.train_impact_model(
                    impact_df,
                    stock_data,
                    model_type=impact_config.get('model_type', 'random_forest')
                )
                logger.info("Trained impact prediction model")
            
            logger.info(f"Completed impact analysis for {len(impact_df)} articles")
            return impact_df
        except Exception as e:
            logger.error(f"Error analyzing impact: {str(e)}")
            raise
    
    def generate_recommendations(self, impact_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate news recommendations based on stock impact.
        
        Args:
            impact_df: DataFrame containing news articles with impact scores
            
        Returns:
            Dictionary with different types of recommendations
        """
        logger.info("Generating recommendations")
        
        try:
            # Generate article recommendations
            top_articles = self.news_recommender.recommend_articles(
                impact_df,
                top_n=10
            )
            
            # Generate cluster recommendations
            top_clusters = self.news_recommender.recommend_clusters(
                impact_df,
                top_n=5,
                articles_per_cluster=3
            )
            
            # Generate trending topics
            trending_topics = self.news_recommender.recommend_trending_topics(
                impact_df,
                top_n=5
            )
            
            # Create recommendations dictionary
            recommendations = {
                'top_articles': top_articles,
                'top_clusters': top_clusters,
                'trending_topics': trending_topics,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save recommendations
            self._save_recommendations(recommendations)
            
            logger.info(f"Generated recommendations: {len(top_articles)} articles, {len(top_clusters)} clusters, {len(trending_topics)} topics")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
    
    def _save_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """
        Save recommendations to file.
        
        Args:
            recommendations: Dictionary with recommendations
        """
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save top articles
            if 'top_articles' in recommendations and not recommendations['top_articles'].empty:
                articles_path = os.path.join(self.results_dir, f"top_articles_{timestamp}.csv")
                recommendations['top_articles'].to_csv(articles_path, index=False)
            
            # Save trending topics
            if 'trending_topics' in recommendations and recommendations['trending_topics']:
                topics_path = os.path.join(self.results_dir, f"trending_topics_{timestamp}.json")
                with open(topics_path, 'w', encoding='utf-8') as f:
                    json.dump(recommendations['trending_topics'], f, ensure_ascii=False, indent=2)
            
            # Save full recommendations
            full_path = os.path.join(self.results_dir, f"recommendations_{timestamp}.json")
            
            # Convert DataFrames to dictionaries for JSON serialization
            rec_copy = recommendations.copy()
            if 'top_articles' in rec_copy and not rec_copy['top_articles'].empty:
                rec_copy['top_articles'] = rec_copy['top_articles'].to_dict(orient='records')
            
            if 'top_clusters' in rec_copy:
                clusters_dict = {}
                for cluster_id, df in rec_copy['top_clusters'].items():
                    clusters_dict[str(cluster_id)] = df.to_dict(orient='records')
                rec_copy['top_clusters'] = clusters_dict
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(rec_copy, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved recommendations to {self.results_dir}")
        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
    
    def visualize_results(self, impact_df: pd.DataFrame, 
                         stock_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Create visualizations of results.
        
        Args:
            impact_df: DataFrame containing news articles with impact scores
            stock_data: Dictionary mapping tickers to stock price DataFrames
            
        Returns:
            List of paths to visualization files
        """
        logger.info("Creating visualizations")
        
        try:
            visualization_paths = []
            
            # Visualize impact for each ticker
            for ticker in stock_data.keys():
                viz_path = self.impact_analyzer.visualize_impact(
                    impact_df,
                    stock_data,
                    ticker
                )
                
                if viz_path:
                    visualization_paths.append(viz_path)
            
            logger.info(f"Created {len(visualization_paths)} visualizations")
            return visualization_paths
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            return []
    
    def run_pipeline(self, use_existing_data: bool = True) -> Dict[str, Any]:
        """
        Run the complete CLEAR pipeline.
        
        Args:
            use_existing_data: Whether to use existing data files
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting CLEAR pipeline")
        
        try:
            # Step 1: Load data
            news_df, stock_data = self.load_data(use_existing=use_existing_data)
            
            if news_df.empty:
                logger.error("No news data available")
                return {'error': 'No news data available'}
            
            # Step 2: Preprocess data
            preprocessed_df = self.preprocess_data(news_df)
            
            # Step 3: Vectorize articles
            vectorized_df = self.vectorize_articles(preprocessed_df)
            
            # Step 4: Cluster articles
            clustered_df = self.cluster_articles(vectorized_df)
            
            # Step 5: Analyze impact
            impact_df = self.analyze_impact(clustered_df, stock_data)
            
            # Step 6: Generate recommendations
            recommendations = self.generate_recommendations(impact_df)
            
            # Step 7: Create visualizations
            visualization_paths = self.visualize_results(impact_df, stock_data)
            
            # Create results dictionary
            results = {
                'news_count': len(news_df),
                'cluster_count': len(clustered_df['cluster_id'].unique()),
                'recommendations': recommendations,
                'visualizations': visualization_paths,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info("Completed CLEAR pipeline")
            return results
        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            return {'error': str(e)}
    
    def schedule_pipeline(self) -> None:
        """
        Schedule pipeline to run at market open and close.
        """
        schedule_config = self.config.get('schedule', {})
        market_open = schedule_config.get('market_open', '09:00')
        market_close = schedule_config.get('market_close', '15:30')
        
        logger.info(f"Scheduling pipeline to run at market open ({market_open}) and close ({market_close})")
        
        # Schedule runs
        schedule.every().day.at(market_open).do(self.run_pipeline)
        schedule.every().day.at(market_close).do(self.run_pipeline)
        
        # Run continuously
        while True:
            schedule.run_pending()
            time.sleep(60)


def main():
    """
    Main function to run the CLEAR pipeline.
    """
    parser = argparse.ArgumentParser(description='CLEAR Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--use-existing', action='store_true',
                       help='Use existing data files')
    parser.add_argument('--schedule', action='store_true',
                       help='Schedule pipeline to run at market open and close')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = CLEARPipeline(config_path=args.config)
    
    if args.schedule:
        # Run scheduled pipeline
        pipeline.schedule_pipeline()
    else:
        # Run pipeline once
        results = pipeline.run_pipeline(use_existing_data=args.use_existing)
        
        # Print summary
        print("\nCLEAR Pipeline Results:")
        print(f"Processed {results.get('news_count', 0)} news articles")
        print(f"Created {results.get('cluster_count', 0)} clusters")
        print(f"Generated recommendations for {len(results.get('recommendations', {}).get('top_articles', []))} articles")
        print(f"Created {len(results.get('visualizations', []))} visualizations")


if __name__ == "__main__":
    main()
