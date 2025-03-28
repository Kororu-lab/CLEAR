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
                 use_market_hours: bool = True,
                 company_ticker_map: Dict[str, str] = None):
        """
        Initialize the stock impact analyzer.
        
        Args:
            time_windows: List of time windows for impact analysis
            impact_thresholds: Thresholds for impact categorization
            use_gpu: Whether to use GPU for neural network models
            models_dir: Directory to save/load models
            use_market_hours: Whether to consider market opening/closing hours
            company_ticker_map: Mapping of company names to ticker symbols
        """
        # Default time windows
        self.time_windows = time_windows or [
            {"name": "immediate", "days": 1},  # Changed from hours to days to match stock data format
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
        
        # Use market hours for analysis
        self.use_market_hours = use_market_hours
        
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
                        
                        # Calculate volatility (high-low range)
                        volatilities = [data.get('volatility', 0) for data in ticker_impacts[window_name] if 'volatility' in data]
                        
                        avg_price_change = np.mean(price_changes)
                        avg_volume_change = np.mean(volume_changes)
                        avg_volatility = np.mean(volatilities) if volatilities else 0
                        
                        # Calculate impact score (-5 to +5 scale)
                        impact_score = self._calculate_impact_score(avg_price_change, avg_volume_change, avg_volatility)
                        
                        # Store in DataFrame
                        df.at[idx, f'impact_{window_name}'] = impact_score
                        df.at[idx, f'price_change_{window_name}'] = avg_price_change
                        df.at[idx, f'volume_change_{window_name}'] = avg_volume_change
                        df.at[idx, f'volatility_{window_name}'] = avg_volatility
            
            # Calculate overall impact
            df['impact_overall'] = df.apply(self._calculate_overall_impact, axis=1)
            
            # Add sentiment analysis if available
            if 'Emotion' in df.columns:
                df['sentiment_score'] = df['Emotion'].apply(self._map_emotion_to_score)
                
                # Combine sentiment with impact
                df['combined_score'] = df.apply(
                    lambda x: self._combine_sentiment_and_impact(
                        x['sentiment_score'] if pd.notna(x.get('sentiment_score')) else 0,
                        x['impact_overall'] if pd.notna(x.get('impact_overall')) else 0
                    ), 
                    axis=1
                )
            
            logger.info(f"Impact analysis completed for {len(df)} articles")
            return df
            
        except Exception as e:
            logger.error(f"Error in impact analysis: {str(e)}")
            return news_df
    
    def train_impact_model(self, news_df: pd.DataFrame, 
                          stock_data: Dict[str, pd.DataFrame],
                          model_type: str = 'random_forest',
                          features: List[str] = None,
                          target: str = 'price_change_short_term',
                          test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train a model to predict stock price impact from news features.
        
        Args:
            news_df: DataFrame containing news articles with impact scores
            stock_data: Dictionary mapping tickers to stock price DataFrames
            model_type: Type of model to train ('linear', 'random_forest', 'neural')
            features: List of feature columns to use
            target: Target column to predict
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_type} model for impact prediction")
        
        try:
            # Analyze impact if not already done
            if target not in news_df.columns:
                news_df = self.analyze_news_impact(news_df, stock_data)
            
            # Default features if not specified
            if features is None:
                features = ['sentiment_score', 'Num_comment']
                
                # Add time window features if available
                for window in self.time_windows:
                    window_name = window['name']
                    if f'impact_{window_name}' in news_df.columns:
                        features.append(f'impact_{window_name}')
            
            # Filter rows with valid target values
            valid_df = news_df.dropna(subset=[target])
            
            if len(valid_df) < 10:
                logger.warning(f"Insufficient data for training: {len(valid_df)} valid rows")
                return {"success": False, "error": "Insufficient data for training"}
            
            # Prepare features and target
            X = valid_df[features].fillna(0)
            y = valid_df[target]
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train model based on type
            if model_type == 'linear':
                model = LinearRegression()
                model.fit(X_train, y_train)
                
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
            elif model_type == 'neural':
                # Simple neural network for regression
                if self.use_gpu and torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train.values).to(device)
                y_train_tensor = torch.FloatTensor(y_train.values).to(device)
                X_test_tensor = torch.FloatTensor(X_test.values).to(device)
                y_test_tensor = torch.FloatTensor(y_test.values).to(device)
                
                # Create datasets and dataloaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                # Define model
                input_size = X_train.shape[1]
                model = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                ).to(device)
                
                # Train model
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                self._train_neural_model(model, train_loader, epochs=100)
            
            # Evaluate model
            if model_type in ['linear', 'random_forest']:
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Save model
                model_path = os.path.join(self.models_dir, f"impact_{model_type}.joblib")
                joblib.dump(model, model_path)
                
                self.impact_model = model
                self.impact_model_type = model_type
                
                logger.info(f"Model trained and saved: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
                
                return {
                    "success": True,
                    "model_type": model_type,
                    "train_score": train_score,
                    "test_score": test_score,
                    "model_path": model_path,
                    "features": features
                }
                
            elif model_type == 'neural':
                # Evaluate neural model
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_test_tensor)
                    test_loss = criterion(y_pred.squeeze(), y_test_tensor).item()
                
                # Save model
                model_path = os.path.join(self.models_dir, "impact_neural.pt")
                torch.save(model.state_dict(), model_path)
                
                self.impact_model = model
                self.impact_model_type = model_type
                
                logger.info(f"Neural model trained and saved: Test Loss = {test_loss:.4f}")
                
                return {
                    "success": True,
                    "model_type": model_type,
                    "test_loss": test_loss,
                    "model_path": model_path,
                    "features": features
                }
            
        except Exception as e:
            logger.error(f"Error training impact model: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def predict_impact(self, article_features: Dict[str, Any]) -> float:
        """
        Predict the impact of a news article on stock prices.
        
        Args:
            article_features: Dictionary of article features
            
        Returns:
            Predicted impact score
        """
        if self.impact_model is None:
            logger.warning("No impact model loaded, cannot predict")
            return 0.0
        
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([article_features])
            
            # Make prediction based on model type
            if self.impact_model_type in ['linear', 'random_forest']:
                prediction = self.impact_model.predict(features_df)[0]
                return prediction
                
            elif self.impact_model_type == 'neural':
                # Convert to tensor
                features_tensor = torch.FloatTensor(features_df.values)
                
                # Move to GPU if available
                if self.use_gpu and torch.cuda.is_available():
                    features_tensor = features_tensor.cuda()
                
                # Make prediction
                self.impact_model.eval()
                with torch.no_grad():
                    prediction = self.impact_model(features_tensor).item()
                
                return prediction
                
        except Exception as e:
            logger.error(f"Error predicting impact: {str(e)}")
            return 0.0
    
    def visualize_impact(self, news_df: pd.DataFrame, 
                        stock_data: Dict[str, pd.DataFrame],
                        ticker: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Visualize the impact of news articles on stock prices.
        
        Args:
            news_df: DataFrame containing news articles with impact scores
            stock_data: Dictionary mapping tickers to stock price DataFrames
            ticker: Stock ticker to visualize
            start_date: Start date for visualization
            end_date: End date for visualization
            save_path: Path to save the visualization
        """
        if ticker not in stock_data:
            logger.warning(f"Ticker {ticker} not found in stock data")
            return
        
        try:
            # Get stock data for ticker
            ticker_df = stock_data[ticker].copy()
            
            # Ensure Date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(ticker_df['Date']):
                ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], format='%Y%m%d')
            
            # Filter by date range if specified
            if start_date:
                ticker_df = ticker_df[ticker_df['Date'] >= start_date]
            if end_date:
                ticker_df = ticker_df[ticker_df['Date'] <= end_date]
            
            # Filter news for this ticker
            ticker_news = news_df[
                news_df.apply(
                    lambda row: ticker in self._extract_tickers(row), axis=1
                )
            ].copy()
            
            # Ensure Date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(ticker_news['Date']):
                ticker_news['Date'] = ticker_news['Date'].apply(
                    lambda x: pd.to_datetime(str(x).split()[0], format='%Y%m%d')
                )
            
            # Filter by date range if specified
            if start_date:
                ticker_news = ticker_news[ticker_news['Date'] >= start_date]
            if end_date:
                ticker_news = ticker_news[ticker_news['Date'] <= end_date]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot stock price
            plt.subplot(2, 1, 1)
            plt.plot(ticker_df['Date'], ticker_df['End'], label='Closing Price')
            
            # Plot news impact
            for _, row in ticker_news.iterrows():
                if 'impact_overall' in row and pd.notna(row['impact_overall']):
                    impact = row['impact_overall']
                    color = 'green' if impact > 0 else 'red'
                    alpha = min(abs(impact) / 5, 1.0)  # Scale alpha by impact magnitude
                    plt.axvline(x=row['Date'], color=color, alpha=alpha, linestyle='--')
            
            plt.title(f'Stock Price and News Impact for {ticker}')
            plt.ylabel('Price')
            plt.legend()
            
            # Plot volume
            plt.subplot(2, 1, 2)
            plt.bar(ticker_df['Date'], ticker_df['Volume'], color='blue', alpha=0.6)
            plt.ylabel('Volume')
            plt.xlabel('Date')
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing impact: {str(e)}")
    
    def get_top_impactful_news(self, news_df: pd.DataFrame, 
                              top_n: int = 10, 
                              impact_col: str = 'impact_overall',
                              include_negative: bool = True) -> pd.DataFrame:
        """
        Get the top most impactful news articles.
        
        Args:
            news_df: DataFrame containing news articles with impact scores
            top_n: Number of articles to return
            impact_col: Column to use for impact scores
            include_negative: Whether to include negative impact articles
            
        Returns:
            DataFrame with top impactful articles
        """
        if impact_col not in news_df.columns:
            logger.warning(f"Impact column {impact_col} not found in DataFrame")
            return pd.DataFrame()
        
        try:
            # Filter articles with valid impact scores
            valid_df = news_df.dropna(subset=[impact_col])
            
            if include_negative:
                # Sort by absolute impact
                sorted_df = valid_df.iloc[
                    valid_df[impact_col].abs().argsort()[::-1]
                ]
            else:
                # Sort by positive impact only
                sorted_df = valid_df.sort_values(impact_col, ascending=False)
            
            # Return top N
            return sorted_df.head(top_n)
            
        except Exception as e:
            logger.error(f"Error getting top impactful news: {str(e)}")
            return pd.DataFrame()
    
    def _extract_tickers(self, row: pd.Series) -> List[str]:
        """
        Extract stock tickers from an article row.
        
        Args:
            row: Series representing a news article
                
        Returns:
            List of stock tickers
        """
        tickers = []
        
        # Check for explicit tickers column
        if 'tickers' in row and row['tickers']:
            if isinstance(row['tickers'], list):
                tickers = row['tickers']
            elif isinstance(row['tickers'], str):
                # Handle string representation of list
                if row['tickers'].startswith('[') and row['tickers'].endswith(']'):
                    try:
                        tickers = eval(row['tickers'])
                    except:
                        tickers = [row['tickers']]
                else:
                    tickers = [row['tickers']]
        
        # Extract from the Title field to determine the company
        elif 'Title' in row and row['Title']:
            title = row['Title']
            # Use the company to ticker mapping
            for company, ticker in self.company_ticker_map.items():
                if company in title:
                    tickers.append(ticker)
        
        # Also check Body if available
        elif 'Body' in row and row['Body']:
            body = row['Body']
            # Use the company to ticker mapping
            for company, ticker in self.company_ticker_map.items():
                if company in body:
                    tickers.append(ticker)
        
        return tickers
    
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
            
            # Calculate price change
            price_change_pct = (end_row['End'] - start_row['End']) / start_row['End'] * 100
            
            # Calculate volume change
            volume_change_pct = (end_row['Volume'] - start_row['Volume']) / start_row['Volume'] * 100 if start_row['Volume'] > 0 else 0
            
            # Calculate volatility (high-low range)
            volatility = ((end_row['High'] - end_row['Low']) / end_row['End']) * 100
            
            # Calculate additional metrics using open/close data
            open_close_delta = ((end_row['End'] - end_row['Start']) / end_row['Start']) * 100
            
            return {
                'price_change': price_change_pct,
                'volume_change': volume_change_pct,
                'volatility': volatility,
                'open_close_delta': open_close_delta,
                'start_date': start_row['Date'],
                'end_date': end_row['Date'],
                'start_price': start_row['End'],
                'end_price': end_row['End'],
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
        total_weight = 0
        weighted_sum = 0
        
        for window in self.time_windows:
            window_name = window['name']
            impact_col = f'impact_{window_name}'
            
            if impact_col in row and pd.notna(row[impact_col]):
                weight = weights.get(window_name, 1.0 / len(self.time_windows))
                weighted_sum += row[impact_col] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _train_neural_model(self, model: nn.Module, dataloader: DataLoader, 
                          epochs: int = 100, learning_rate: float = 0.001) -> None:
        """
        Train a neural network model.
        
        Args:
            model: PyTorch model to train
            dataloader: DataLoader for training data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        # Set device
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}')
    
    def _map_emotion_to_score(self, emotion: str) -> float:
        """
        Map emotion string to numerical sentiment score.
        
        Args:
            emotion: Emotion string from news data
            
        Returns:
            Sentiment score from -1 to 1
        """
        if pd.isna(emotion) or not emotion:
            return 0.0
            
        # Convert to lowercase for case-insensitive matching
        emotion = emotion.lower()
        
        # Mapping of emotions to scores
        emotion_map = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
            'mixed': 0.0,
            '긍정': 1.0,  # Korean for positive
            '부정': -1.0,  # Korean for negative
            '중립': 0.0,   # Korean for neutral
            '혼합': 0.0    # Korean for mixed
        }
        
        # Return mapped score or default to 0
        return emotion_map.get(emotion, 0.0)
    
    def _combine_sentiment_and_impact(self, sentiment_score: float, impact_score: float) -> float:
        """
        Combine sentiment and impact scores.
        
        Args:
            sentiment_score: Sentiment score from -1 to 1
            impact_score: Impact score from -5 to 5
            
        Returns:
            Combined score
        """
        # Scale sentiment to match impact range
        scaled_sentiment = sentiment_score * 2.5
        
        # Weighted combination (60% impact, 40% sentiment)
        combined = 0.6 * impact_score + 0.4 * scaled_sentiment
        
        return combined
    
    def load_model(self, model_path: str, model_type: str) -> bool:
        """
        Load a trained impact model.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model ('linear', 'random_forest', 'neural')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_type in ['linear', 'random_forest']:
                self.impact_model = joblib.load(model_path)
                self.impact_model_type = model_type
                logger.info(f"Loaded {model_type} model from {model_path}")
                return True
                
            elif model_type == 'neural':
                # Create a simple model architecture (must match the saved one)
                input_size = 5  # Default, will need to be adjusted based on actual features
                model = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
                
                # Load state dict
                if self.use_gpu and torch.cuda.is_available():
                    model.load_state_dict(torch.load(model_path))
                    model = model.cuda()
                else:
                    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                
                self.impact_model = model
                self.impact_model_type = model_type
                logger.info(f"Loaded neural model from {model_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
