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
            df['impact_score'] = df.apply(self._calculate_overall_impact, axis=1)
            # df['impact_overall'] = df.apply(self._calculate_overall_impact, axis=1)
            
            return df
        except Exception as e:
            logger.error(f"Error in analyze_news_impact: {str(e)}")
            return news_df

    def train_impact_model(self, news_df: pd.DataFrame, 
                          stock_data: Dict[str, pd.DataFrame],
                          model_type: str = 'random_forest',
                          features: List[str] = None) -> None:
        """
        Train a model to predict the impact of news on stock prices.
        
        Args:
            news_df: DataFrame containing news articles with impact scores
            stock_data: Dictionary mapping tickers to stock price DataFrames
            model_type: Type of model to train ('random_forest', 'linear', 'neural')
            features: List of features to use for prediction
        """
        if len(news_df) == 0:
            logger.warning("Empty DataFrame provided for model training")
            return
            
        logger.info(f"Training {model_type} impact prediction model on {len(news_df)} articles")
        
        try:
            # Default features if not provided
            if features is None:
                features = ['cluster_id', 'cluster_size']
                
                # Add vector features if available
                if 'vector' in news_df.columns:
                    # Extract vector components as features
                    vector_df = pd.DataFrame(
                        news_df['vector'].tolist(), 
                        index=news_df.index
                    )
                    vector_df.columns = [f'vector_{i}' for i in range(vector_df.shape[1])]
                    
                    # Combine with original DataFrame
                    train_df = pd.concat([news_df, vector_df], axis=1)
                    
                    # Add vector component names to features
                    features.extend([f'vector_{i}' for i in range(vector_df.shape[1])])
                else:
                    train_df = news_df.copy()
                    
                # Add other available features
                for col in ['cluster_topic', 'Press', 'Emotion']:
                    if col in train_df.columns:
                        # Convert categorical to one-hot
                        one_hot = pd.get_dummies(train_df[col], prefix=col)
                        train_df = pd.concat([train_df, one_hot], axis=1)
                        features.extend(one_hot.columns.tolist())
            else:
                train_df = news_df.copy()
            
            # Filter out rows without impact scores
            train_df = train_df.dropna(subset=['impact_score'])
            
            if len(train_df) < 10:
                logger.warning("Not enough data for model training")
                return
            
            # Prepare features and target
            X = train_df[features].fillna(0)
            y = train_df['impact_score']
            
            # Train model based on type
            if model_type == 'random_forest':
                self.impact_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.impact_model.fit(X, y)
                
            elif model_type == 'linear':
                self.impact_model = LinearRegression()
                self.impact_model.fit(X, y)
                
            elif model_type == 'neural':
                # Convert to PyTorch tensors
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
                
                # Create dataset and dataloader
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Define neural network
                input_size = X.shape[1]
                self.impact_model = NeuralImpactModel(input_size)
                
                # Move to GPU if available
                if self.use_gpu:
                    self.impact_model = self.impact_model.cuda()
                
                # Train the model
                self._train_neural_model(self.impact_model, dataloader)
            
            else:
                logger.warning(f"Unknown model type: {model_type}, using random forest")
                self.impact_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                self.impact_model.fit(X, y)
                model_type = 'random_forest'
            
            self.impact_model_type = model_type
            self.model_features = features
            
            # Save the model
            self._save_impact_model()
            
            logger.info(f"Trained {model_type} impact model with {len(features)} features")
        except Exception as e:
            logger.error(f"Error training impact model: {str(e)}")

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

    def predict_impact(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the impact of news articles on stock prices.
        
        Args:
            articles_df: DataFrame containing news articles
            
        Returns:
            DataFrame with predicted impact scores
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for impact prediction")
            return articles_df
            
        if self.impact_model is None:
            logger.warning("Impact model not trained, cannot predict")
            return articles_df
            
        logger.info(f"Predicting impact for {len(articles_df)} articles")
        
        try:
            # Create a copy to avoid modifying the original
            df = articles_df.copy()
            
            # Prepare features
            if 'vector' in df.columns and any(f.startswith('vector_') for f in self.model_features):
                # Extract vector components as features
                vector_df = pd.DataFrame(
                    df['vector'].tolist(), 
                    index=df.index
                )
                vector_df.columns = [f'vector_{i}' for i in range(vector_df.shape[1])]
                
                # Combine with original DataFrame
                df = pd.concat([df, vector_df], axis=1)
            
            # Prepare one-hot features if needed
            for col in ['cluster_topic', 'Press', 'Emotion']:
                if col in df.columns and any(f.startswith(f'{col}_') for f in self.model_features):
                    one_hot = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, one_hot], axis=1)
            
            # Select features and fill missing values
            X = df[self.model_features].fillna(0)
            
            # Predict impact
            if self.impact_model_type in ['random_forest', 'linear']:
                df['predicted_impact'] = self.impact_model.predict(X)
                
            elif self.impact_model_type == 'neural':
                # Convert to PyTorch tensor
                X_tensor = torch.tensor(X.values, dtype=torch.float32)
                
                # Move to GPU if available
                if self.use_gpu:
                    X_tensor = X_tensor.cuda()
                
                # Set model to evaluation mode
                self.impact_model.eval()
                
                # Predict
                with torch.no_grad():
                    predictions = self.impact_model(X_tensor)
                    
                    # Move back to CPU if needed
                    if self.use_gpu:
                        predictions = predictions.cpu()
                    
                    df['predicted_impact'] = predictions.numpy().flatten()
            
            logger.info(f"Predicted impact for {len(df)} articles")
            return df
        except Exception as e:
            logger.error(f"Error predicting impact: {str(e)}")
            return articles_df
   

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
            text += str(row['Title']) + " "
        if 'Body' in row and pd.notna(row['Body']):
            text += str(row['Body'])
            
        # Check for company names in text
        for company, ticker in self.company_ticker_map.items():
            if company in text:
                tickers.append(ticker)
                
        return list(set(tickers))  # Remove duplicates
    
    def _calculate_impact(self, stock_df: pd.DataFrame, 
                        article_date: datetime,
                        time_delta: timedelta) -> Dict[str, float]:
        """
        Calculate the impact of a news article on stock price.
        
        Args:
            stock_df: DataFrame containing stock price data
            article_date: Publication date of the article
            time_delta: Time window to measure impact
            
        Returns:
            Dictionary with impact metrics
        """
        try:
            # Ensure Date column is datetime
            if 'Date' in stock_df.columns and not pd.api.types.is_datetime64_any_dtype(stock_df['Date']):
                stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y%m%d')
            
            # Ignore Time column as it's always 0 (daily data)
            # Treat Start as 09:00 opening price and End as 15:00 closing price
            
            # Find the closest trading day on or after the article date
            start_date = article_date
            start_row = stock_df[stock_df['Date'] >= start_date].sort_values('Date').iloc[0] if not stock_df[stock_df['Date'] >= start_date].empty else None
            
            if start_row is None:
                return None
            
            # Find the trading day after the time window
            end_date = article_date + time_delta
            end_row = stock_df[stock_df['Date'] >= end_date].sort_values('Date').iloc[0] if not stock_df[stock_df['Date'] >= end_date].empty else None
            
            if end_row is None:
                # Use the latest available data if end date is beyond available data
                end_row = stock_df.sort_values('Date').iloc[-1]
            
            # Calculate price change (using End price which is 15:00 closing price)
            price_change_pct = (end_row['End'] - start_row['End']) / start_row['End'] * 100
            
            # Calculate volume change
            volume_change_pct = (end_row['Volume'] - start_row['Volume']) / start_row['Volume'] * 100 if start_row['Volume'] > 0 else 0
            
            # Calculate volatility (high-low range)
            volatility = ((end_row['High'] - end_row['Low']) / end_row['End']) * 100
            
            # Calculate additional metrics using open/close data
            # Start is 09:00 opening price, End is 15:00 closing price
            open_close_delta = ((end_row['End'] - end_row['Start']) / end_row['Start']) * 100
            
            return {
                'price_change': price_change_pct,
                'volume_change': volume_change_pct,
                'volatility': volatility,
                'open_close_delta': open_close_delta,
                'start_date': start_row['Date'],
                'end_date': end_row['Date'],
                'start_price': start_row['End'],  # 15:00 closing price on start day
                'end_price': end_row['End'],      # 15:00 closing price on end day
                'open_price': end_row['Start'],   # 09:00 opening price on end day
                'high_price': end_row['High'],
                'low_price': end_row['Low']
            }
        except Exception as e:
            logger.error(f"Error calculating impact: {str(e)}")
            return None
    
    def _calculate_impact_score(self, price_change: float, volume_change: float, volatility: float = 0) -> float:
        """
        Calculate an impact score on a scale from -5 to +5.
        
        Args:
            price_change: Percentage change in price
            volume_change: Percentage change in volume
            volatility: Percentage volatility (high-low range)
            
        Returns:
            Impact score from -5 to +5
        """
        # Weight price change more heavily than volume change, and include volatility
        weighted_change = 0.6 * price_change + 0.25 * volume_change + 0.15 * volatility
        
        # Map to -5 to +5 scale
        # Thresholds from config
        high_threshold = self.impact_thresholds['high'] * 100  # Convert to percentage
        medium_threshold = self.impact_thresholds['medium'] * 100
        low_threshold = self.impact_thresholds['low'] * 100
        
        if weighted_change > high_threshold:
            impact_score = 5
        elif weighted_change > medium_threshold:
            impact_score = 3
        elif weighted_change > low_threshold:
            impact_score = 1
        elif weighted_change > -low_threshold:
            impact_score = 0
        elif weighted_change > -medium_threshold:
            impact_score = -1
        elif weighted_change > -high_threshold:
            impact_score = -3
        else:
            impact_score = -5
        
        return impact_score
    
    def _calculate_overall_impact(self, row: pd.Series) -> float:
        """
        Calculate overall impact score from individual time window scores.
        
        Args:
            row: Series representing a news article with impact scores
            
        Returns:
            Overall impact score
        """
        # Weights for different time windows
        weights = {
            'immediate': 0.5,
            'short_term': 0.3,
            'medium_term': 0.2
        }
        
        # Calculate weighted average
        impact_sum = 0
        weight_sum = 0
        
        for window in self.time_windows:
            window_name = window['name']
            impact_col = f'impact_{window_name}'
            
            if impact_col in row and pd.notna(row[impact_col]):
                weight = weights.get(window_name, 0.1)
                impact_sum += row[impact_col] * weight
                weight_sum += weight
        
        if weight_sum == 0:
            return 0
            
        return impact_sum / weight_sum
    
    def _train_impact_model(self, df: pd.DataFrame, feature_cols: List[str], model_type: str = 'linear') -> None:
        """
        Train a model to predict impact from features.
        
        Args:
            df: DataFrame with features and impact scores
            feature_cols: List of feature column names
            model_type: Type of model to train ('linear', 'rf', 'nn')
        """
        logger.info(f"Training {model_type} impact model with {len(df)} samples")
        
        try:
            # Prepare data
            X = df[feature_cols].values
            y = df['impact_overall'].values
            
            if model_type == 'linear':
                # Train linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Save model
                self.impact_model = model
                self.impact_model_type = 'linear'
                
                # Save to disk
                model_path = os.path.join(self.models_dir, 'linear_impact_model.joblib')
                joblib.dump(model, model_path)
                
                logger.info(f"Linear impact model trained and saved to {model_path}")
                
            elif model_type == 'rf':
                # Train random forest model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Save model
                self.impact_model = model
                self.impact_model_type = 'rf'
                
                # Save to disk
                model_path = os.path.join(self.models_dir, 'rf_impact_model.joblib')
                joblib.dump(model, model_path)
                
                logger.info(f"Random forest impact model trained and saved to {model_path}")
                
            elif model_type == 'nn':
                # Define neural network model
                input_dim = X.shape[1]
                
                class ImpactNN(nn.Module):
                    def __init__(self, input_dim):
                        super(ImpactNN, self).__init__()
                        self.fc1 = nn.Linear(input_dim, 64)
                        self.fc2 = nn.Linear(64, 32)
                        self.fc3 = nn.Linear(32, 1)
                        self.relu = nn.ReLU()
                        
                    def forward(self, x):
                        x = self.relu(self.fc1(x))
                        x = self.relu(self.fc2(x))
                        x = self.fc3(x)
                        return x.squeeze()
                
                model = ImpactNN(input_dim)
                
                # Convert data to tensors
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32)
                
                # Move to GPU if available
                if self.use_gpu:
                    X_tensor = X_tensor.cuda()
                    y_tensor = y_tensor.cuda()
                    model = model.cuda()
                
                # Define loss function and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Train model
                model.train()
                batch_size = 32
                n_epochs = 100
                
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                for epoch in range(n_epochs):
                    running_loss = 0.0
                    for inputs, targets in dataloader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(dataloader):.4f}")
                
                # Save model
                self.impact_model = model
                self.impact_model_type = 'nn'
                
                # Save to disk
                model_path = os.path.join(self.models_dir, 'nn_impact_model.pt')
                torch.save(model.state_dict(), model_path)
                
                logger.info(f"Neural network impact model trained and saved to {model_path}")
                
            else:
                logger.error(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error training impact model: {str(e)}")
    
    def _load_impact_model(self, model_type: str = 'linear') -> bool:
        """
        Load a trained impact model from disk.
        
        Args:
            model_type: Type of model to load ('linear', 'rf', 'nn')
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            if model_type == 'linear':
                model_path = os.path.join(self.models_dir, 'linear_impact_model.joblib')
                if os.path.exists(model_path):
                    self.impact_model = joblib.load(model_path)
                    self.impact_model_type = 'linear'
                    logger.info(f"Linear impact model loaded from {model_path}")
                    return True
                    
            elif model_type == 'rf':
                model_path = os.path.join(self.models_dir, 'rf_impact_model.joblib')
                if os.path.exists(model_path):
                    self.impact_model = joblib.load(model_path)
                    self.impact_model_type = 'rf'
                    logger.info(f"Random forest impact model loaded from {model_path}")
                    return True
                    
            elif model_type == 'nn':
                model_path = os.path.join(self.models_dir, 'nn_impact_model.pt')
                if os.path.exists(model_path):
                    input_dim = len(self.time_windows) * 3  # Assuming 3 features per time window
                    
                    class ImpactNN(nn.Module):
                        def __init__(self, input_dim):
                            super(ImpactNN, self).__init__()
                            self.fc1 = nn.Linear(input_dim, 64)
                            self.fc2 = nn.Linear(64, 32)
                            self.fc3 = nn.Linear(32, 1)
                            self.relu = nn.ReLU()
                            
                        def forward(self, x):
                            x = self.relu(self.fc1(x))
                            x = self.relu(self.fc2(x))
                            x = self.fc3(x)
                            return x.squeeze()
                    
                    model = ImpactNN(input_dim)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    
                    self.impact_model = model
                    self.impact_model_type = 'nn'
                    logger.info(f"Neural network impact model loaded from {model_path}")
                    return True
            
            logger.warning(f"No {model_type} impact model found")
            return False
            
        except Exception as e:
            logger.error(f"Error loading impact model: {str(e)}")
            return False
    
    def preprocess_stock_data(self, stock_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess stock data for impact analysis.
        
        Args:
            stock_df: DataFrame containing stock price data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Create a copy to avoid modifying the original
            df = stock_df.copy()
            
            # Ensure Date column is datetime
            if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Ignore Time column as it's always 0 (daily data)
            # Treat Start as 09:00 opening price and End as 15:00 closing price
            
            # Calculate daily returns
            df['daily_return'] = df['End'].pct_change() * 100
            
            # Calculate volatility (high-low range)
            df['volatility'] = ((df['High'] - df['Low']) / df['End']) * 100
            
            # Calculate open-close delta
            df['open_close_delta'] = ((df['End'] - df['Start']) / df['Start']) * 100
            
            # Calculate volume change
            df['volume_change'] = df['Volume'].pct_change() * 100
            
            # Calculate moving averages
            df['ma_5'] = df['End'].rolling(window=5).mean()
            df['ma_20'] = df['End'].rolling(window=20).mean()
            
            # Calculate relative strength index (RSI)
            delta = df['End'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            return df
        except Exception as e:
            logger.error(f"Error preprocessing stock data: {str(e)}")
            return stock_df
    
    def calculate_market_correlation(self, stock_df: pd.DataFrame, 
                                   market_df: pd.DataFrame) -> float:
        """
        Calculate correlation between stock and market returns.
        
        Args:
            stock_df: DataFrame containing stock price data
            market_df: DataFrame containing market index data
            
        Returns:
            Correlation coefficient
        """
        try:
            # Ensure Date columns are datetime
            stock = stock_df.copy()
            market = market_df.copy()
            
            if 'Date' in stock.columns and not pd.api.types.is_datetime64_any_dtype(stock['Date']):
                stock['Date'] = pd.to_datetime(stock['Date'], format='%Y%m%d')
                
            if 'Date' in market.columns and not pd.api.types.is_datetime64_any_dtype(market['Date']):
                market['Date'] = pd.to_datetime(market['Date'], format='%Y%m%d')
            
            # Calculate returns
            stock['return'] = stock['End'].pct_change()
            market['return'] = market['End'].pct_change()
            
            # Merge on date
            merged = pd.merge(stock, market, on='Date', suffixes=('_stock', '_market'))
            
            # Calculate correlation
            correlation = merged['return_stock'].corr(merged['return_market'])
            
            return correlation
        except Exception as e:
            logger.error(f"Error calculating market correlation: {str(e)}")
            return 0.0
