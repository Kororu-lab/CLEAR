"""
Stock impact analyzer module for measuring the impact of news on stock prices.
Replaces personalization components in NAVER's AiRS with stock impact analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockImpactAnalyzer:
    """
    Analyzer for measuring the impact of news on stock prices.
    Replaces personalization components in NAVER's AiRS with stock impact analysis.
    """
    
    def __init__(self, 
                 time_windows: List[Dict[str, Any]] = None,
                 impact_thresholds: Dict[str, float] = None,
                 use_gpu: bool = True,
                 models_dir: str = None,
                 company_ticker_map: Dict[str, str] = None):
        """
        Initialize the stock impact analyzer.
        
        Args:
            time_windows: List of time windows for impact analysis
            impact_thresholds: Thresholds for impact categorization
            use_gpu: Whether to use GPU for neural network models
            models_dir: Directory to save/load models
            company_ticker_map: Mapping of company names to ticker symbols
        """
        # Default time windows
        self.time_windows = time_windows or [
            {"name": "immediate", "days": 1},  # Daily data
            {"name": "short_term", "days": 3},
            {"name": "medium_term", "days": 7}
        ]
        
        # Default impact thresholds
        self.impact_thresholds = impact_thresholds or {
            "high": 0.02,    # 2% price change
            "medium": 0.01,  # 1% price change
            "low": 0.005     # 0.5% price change
        }
        
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "stock_impact")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check GPU availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available, falling back to CPU")
        
        # Initialize impact prediction model
        self.impact_model = None
        self.impact_model_type = None
        
        # Default company to ticker mapping
        self.company_ticker_map = company_ticker_map or {
            "삼성전자": "005930",
            "SK하이닉스": "000660",
            "LG전자": "066570",
            "현대자동차": "005380",
            "NAVER": "035420",
            "카카오": "035720",
            "셀트리온": "068270",
            "삼성바이오로직스": "207940",
            "기아": "000270",
            "POSCO홀딩스": "005490"
        }
        
        logger.info(f"Initialized StockImpactAnalyzer with {len(self.time_windows)} time windows")
    
    def analyze_news_impact(self, news_df: pd.DataFrame, 
                           stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze the impact of news articles on stock prices.
        
        Args:
            news_df: DataFrame containing news articles with tickers
            stock_data: Dictionary mapping tickers to stock price DataFrames
            
        Returns:
            DataFrame with impact scores
        """
        if len(news_df) == 0:
            logger.warning("Empty DataFrame provided for impact analysis")
            return news_df
            
        logger.info(f"Analyzing impact for {len(news_df)} news articles")
        
        # Create a copy to avoid modifying the original
        df = news_df.copy()
        
        try:
            # Ensure Date column is datetime
            if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                # Handle the specific date format (20250101 18:56)
                df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(str(x).split()[0], format='%Y%m%d'))
            
            # Initialize impact columns
            for window in self.time_windows:
                window_name = window['name']
                df[f'impact_{window_name}'] = np.nan
                df[f'price_change_{window_name}'] = np.nan
                df[f'volume_change_{window_name}'] = np.nan
                df[f'volatility_{window_name}'] = np.nan  # Added volatility metric
            
            # Calculate impact for each article
            for idx, row in df.iterrows():
                # Skip if no date
                if 'Date' not in row or pd.isna(row['Date']):
                    continue
                
                # Get tickers mentioned in the article
                tickers = self._extract_tickers(row)
                if not tickers:
                    continue
                
                # Calculate impact for each ticker
                ticker_impacts = {}
                
                for ticker in tickers:
                    if ticker not in stock_data:
                        continue
                        
                    ticker_df = stock_data[ticker]
                    
                    # Skip if stock data is empty
                    if ticker_df.empty:
                        continue
                    
                    # Calculate impact for each time window
                    for window in self.time_windows:
                        window_name = window['name']
                        window_days = window.get('days', 0)
                        
                        # Calculate time delta
                        time_delta = timedelta(days=window_days)
                        
                        # Calculate impact
                        impact_data = self._calculate_impact(
                            ticker_df, row['Date'], time_delta
                        )
                        
                        if impact_data:
                            # Store impact data
                            if window_name not in ticker_impacts:
                                ticker_impacts[window_name] = []
                            ticker_impacts[window_name].append(impact_data)
                
                # Aggregate impacts across tickers for each time window
                for window in self.time_windows:
                    window_name = window['name']
                    
                    if window_name in ticker_impacts and ticker_impacts[window_name]:
                        # Calculate average impact
                        price_changes = [data['price_change'] for data in ticker_impacts[window_name]]
                        volume_changes = [data['volume_change'] for data in ticker_impacts[window_name]]
                        volatilities = [data['volatility'] for data in ticker_impacts[window_name]]
                        
                        avg_price_change = np.mean(price_changes)
                        avg_volume_change = np.mean(volume_changes)
                        avg_volatility = np.mean(volatilities)
                        
                        # Calculate impact score
                        impact_score = self._calculate_impact_score(avg_price_change, avg_volume_change, avg_volatility)
                        
                        # Store in DataFrame
                        df.at[idx, f'impact_{window_name}'] = impact_score
                        df.at[idx, f'price_change_{window_name}'] = avg_price_change
                        df.at[idx, f'volume_change_{window_name}'] = avg_volume_change
                        df.at[idx, f'volatility_{window_name}'] = avg_volatility
            
            # Calculate overall impact
            df['impact_overall'] = df.apply(self._calculate_overall_impact, axis=1)
            
            return df
        except Exception as e:
            logger.error(f"Error in analyze_news_impact: {str(e)}")
            return news_df
    
    def predict_price_impact(self, news_df: pd.DataFrame, 
                            stock_data: Dict[str, pd.DataFrame],
                            model_type: str = 'linear') -> pd.DataFrame:
        """
        Predict the impact of news articles on future stock prices.
        
        Args:
            news_df: DataFrame containing news articles with tickers
            stock_data: Dictionary mapping tickers to stock price DataFrames
            model_type: Type of prediction model to use ('linear', 'rf', 'nn')
            
        Returns:
            DataFrame with predicted impact
        """
        if len(news_df) == 0:
            logger.warning("Empty DataFrame provided for impact prediction")
            return news_df
            
        logger.info(f"Predicting impact for {len(news_df)} news articles using {model_type} model")
        
        # Create a copy to avoid modifying the original
        df = news_df.copy()
        
        try:
            # Analyze news impact first to get features
            df = self.analyze_news_impact(df, stock_data)
            
            # Prepare features
            feature_cols = []
            for window in self.time_windows:
                window_name = window['name']
                feature_cols.extend([
                    f'price_change_{window_name}',
                    f'volume_change_{window_name}',
                    f'volatility_{window_name}'
                ])
            
            # Drop rows with missing features
            df_clean = df.dropna(subset=feature_cols)
            
            if len(df_clean) < 10:
                logger.warning("Not enough data for prediction after dropping missing values")
                return df
            
            # Train or load model
            if self.impact_model is None or self.impact_model_type != model_type:
                self._train_impact_model(df_clean, feature_cols, model_type)
            
            # Make predictions
            X = df_clean[feature_cols].values
            
            if model_type == 'nn':
                # Convert to tensor
                X_tensor = torch.tensor(X, dtype=torch.float32)
                
                # Move to GPU if available
                if self.use_gpu:
                    X_tensor = X_tensor.cuda()
                    self.impact_model = self.impact_model.cuda()
                
                # Set model to evaluation mode
                self.impact_model.eval()
                
                # Make predictions
                with torch.no_grad():
                    y_pred = self.impact_model(X_tensor).cpu().numpy()
            else:
                # Use scikit-learn model
                y_pred = self.impact_model.predict(X)
            
            # Add predictions to DataFrame
            df_clean['predicted_impact'] = y_pred
            
            # Merge predictions back to original DataFrame
            df = df.merge(df_clean[['predicted_impact']], left_index=True, right_index=True, how='left')
            
            return df
        except Exception as e:
            logger.error(f"Error in predict_price_impact: {str(e)}")
            return news_df
    
    def visualize_impact(self, news_df: pd.DataFrame, 
                        time_window: str = 'immediate',
                        top_n: int = 10,
                        plot_type: str = 'bar') -> None:
        """
        Visualize the impact of news articles on stock prices.
        
        Args:
            news_df: DataFrame containing news articles with impact scores
            time_window: Time window to visualize
            top_n: Number of top articles to visualize
            plot_type: Type of plot ('bar', 'scatter', 'heatmap')
        """
        if len(news_df) == 0:
            logger.warning("Empty DataFrame provided for visualization")
            return
            
        impact_col = f'impact_{time_window}'
        price_change_col = f'price_change_{time_window}'
        
        if impact_col not in news_df.columns:
            logger.warning(f"Impact column {impact_col} not found in DataFrame")
            return
            
        try:
            # Create a copy and sort by impact
            df = news_df.copy()
            df = df.sort_values(by=impact_col, ascending=False)
            
            # Take top and bottom N articles
            top_articles = df.head(top_n)
            bottom_articles = df.tail(top_n)
            
            if plot_type == 'bar':
                # Create bar plot
                plt.figure(figsize=(12, 8))
                
                # Plot top articles
                plt.subplot(2, 1, 1)
                plt.bar(range(len(top_articles)), top_articles[impact_col], color='green')
                plt.title(f'Top {top_n} Articles by Impact ({time_window})')
                plt.ylabel('Impact Score')
                plt.xticks(range(len(top_articles)), top_articles.index, rotation=45)
                
                # Plot bottom articles
                plt.subplot(2, 1, 2)
                plt.bar(range(len(bottom_articles)), bottom_articles[impact_col], color='red')
                plt.title(f'Bottom {top_n} Articles by Impact ({time_window})')
                plt.ylabel('Impact Score')
                plt.xticks(range(len(bottom_articles)), bottom_articles.index, rotation=45)
                
                plt.tight_layout()
                plt.show()
                
            elif plot_type == 'scatter':
                # Create scatter plot
                if price_change_col in news_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(news_df[price_change_col], news_df[impact_col], alpha=0.6)
                    plt.title(f'Price Change vs. Impact ({time_window})')
                    plt.xlabel('Price Change (%)')
                    plt.ylabel('Impact Score')
                    plt.grid(True, alpha=0.3)
                    plt.show()
                else:
                    logger.warning(f"Price change column {price_change_col} not found in DataFrame")
                    
            elif plot_type == 'heatmap':
                # Create heatmap of correlations
                corr_cols = [col for col in news_df.columns if col.startswith('impact_') or col.startswith('price_change_')]
                if len(corr_cols) > 1:
                    corr = news_df[corr_cols].corr()
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.title('Correlation between Impact and Price Change')
                    plt.tight_layout()
                    plt.show()
                else:
                    logger.warning("Not enough correlation columns found in DataFrame")
        except Exception as e:
            logger.error(f"Error in visualize_impact: {str(e)}")
    
    def _extract_tickers(self, row: pd.Series) -> List[str]:
        """
        Extract ticker symbols from a news article.
        
        Args:
            row: Series representing a news article
            
        Returns:
            List of ticker symbols
        """
        tickers = []
        
        # Check if ticker is already in the row
        if 'ticker' in row and pd.notna(row['ticker']):
            tickers.append(row['ticker'])
            return tickers
            
        # Extract from title and body
        text = ""
        if 'Title' in row and pd.notna(row['Title']):
            text += str(row['Title'])<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>