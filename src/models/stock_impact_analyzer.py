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
                 models_dir: str = None):
        """
        Initialize the stock impact analyzer.
        
        Args:
            time_windows: List of time windows for impact analysis
            impact_thresholds: Thresholds for impact categorization
            use_gpu: Whether to use GPU for neural network models
            models_dir: Directory to save/load models
        """
        # Default time windows
        self.time_windows = time_windows or [
            {"name": "immediate", "hours": 1},
            {"name": "short_term", "hours": 24},
            {"name": "medium_term", "days": 3}
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
                        window_hours = window.get('hours', 0)
                        window_days = window.get('days', 0)
                        
                        # Calculate time delta
                        time_delta = timedelta(hours=window_hours, days=window_days)
                        
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
                        
                        avg_price_change = np.mean(price_changes)
                        avg_volume_change = np.mean(volume_changes)
                        
                        # Calculate impact score (-5 to +5 scale)
                        impact_score = self._calculate_impact_score(avg_price_change, avg_volume_change)
                        
                        # Store in DataFrame
                        df.at[idx, f'impact_{window_name}'] = impact_score
                        df.at[idx, f'price_change_{window_name}'] = avg_price_change
                        df.at[idx, f'volume_change_{window_name}'] = avg_volume_change
            
            # Calculate overall impact score (weighted average of time windows)
            df['impact_score'] = df.apply(
                lambda row: self._calculate_overall_impact(row), axis=1
            )
            
            logger.info(f"Completed impact analysis for {len(df)} articles")
            
            # Save results
            self._save_impact_results(df)
            
            return df
        except Exception as e:
            logger.error(f"Error analyzing news impact: {str(e)}")
            raise
    
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
    
    def visualize_impact(self, articles_df: pd.DataFrame, 
                        stock_data: Dict[str, pd.DataFrame],
                        ticker: str,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> str:
        """
        Visualize the impact of news articles on stock prices.
        
        Args:
            articles_df: DataFrame containing news articles with impact scores
            stock_data: Dictionary mapping tickers to stock price DataFrames
            ticker: Stock ticker to visualize
            start_date: Start date for visualization
            end_date: End date for visualization
            
        Returns:
            Path to the saved visualization
        """
        if ticker not in stock_data:
            logger.warning(f"Stock data not found for ticker: {ticker}")
            return ""
            
        if len(articles_df) == 0 or 'impact_score' not in articles_df.columns:
            logger.warning("No impact scores found in articles DataFrame")
            return ""
            
        logger.info(f"Visualizing impact for ticker: {ticker}")
        
        try:
            # Get stock data for the ticker
            stock_df = stock_data[ticker].copy()
            
            # Ensure Date column is datetime
            if 'Date' in stock_df.columns and not pd.api.types.is_datetime64_any_dtype(stock_df['Date']):
                stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y%m%d')
            
            # Filter by date range if provided
            if start_date:
                stock_df = stock_df[stock_df['Date'] >= start_date]
            if end_date:
                stock_df = stock_df[stock_df['Date'] <= end_date]
            
            # Filter articles for this ticker
            ticker_articles = articles_df[articles_df.apply(
                lambda row: ticker in self._extract_tickers(row), axis=1
            )].copy()
            
            # Ensure Date column is datetime
            if 'Date' in ticker_articles.columns and not pd.api.types.is_datetime64_any_dtype(ticker_articles['Date']):
                ticker_articles['Date'] = ticker_articles['Date'].apply(
                    lambda x: pd.to_datetime(str(x).split()[0], format='%Y%m%d')
                )
            
            # Filter by date range if provided
            if start_date:
                ticker_articles = ticker_articles[ticker_articles['Date'] >= start_date]
            if end_date:
                ticker_articles = ticker_articles[ticker_articles['Date'] <= end_date]
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot stock price
            ax1 = plt.subplot(211)
            ax1.plot(stock_df['Date'], stock_df['End'], label='Stock Price')
            ax1.set_ylabel('Price')
            ax1.set_title(f'Stock Price and News Impact for {ticker}')
            
            # Add markers for news articles
            for _, article in ticker_articles.iterrows():
                impact = article.get('impact_score', 0)
                color = 'green' if impact > 0 else 'red' if impact < 0 else 'gray'
                size = min(100, abs(impact) * 20 + 20)  # Scale marker size by impact
                ax1.scatter(article['Date'], stock_df.loc[stock_df['Date'] == article['Date'], 'End'].values[0],
                           color=color, s=size, alpha=0.7)
            
            # Plot impact scores
            ax2 = plt.subplot(212, sharex=ax1)
            for _, article in ticker_articles.iterrows():
                impact = article.get('impact_score', 0)
                color = 'green' if impact > 0 else 'red' if impact < 0 else 'gray'
                ax2.bar(article['Date'], impact, color=color, alpha=0.7, width=1)
            
            ax2.set_ylabel('Impact Score')
            ax2.set_xlabel('Date')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.models_dir, f"impact_visualization_{ticker}_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved impact visualization to {plot_path}")
            return plot_path
        except Exception as e:
            logger.error(f"Error visualizing impact: {str(e)}")
            return ""
    
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
        
        # Instead of extracting from the Link, use the Title field to determine the company
        elif 'Title' in row and row['Title']:
            title = row['Title']
            # Create a mapping of company names to their corresponding tickers
            company_to_ticker = {
                "삼성전자": "005930",
                "SK그룹": "000660",   # Replace with the correct ticker if different
                # Add more mappings as needed
            }
            for company, ticker in company_to_ticker.items():
                if company in title:
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
            
            return {
                'price_change': price_change_pct,
                'volume_change': volume_change_pct,
                'start_date': start_row['Date'],
                'end_date': end_row['Date'],
                'start_price': start_row['End'],
                'end_price': end_row['End']
            }
        except Exception as e:
            logger.error(f"Error calculating impact: {str(e)}")
            return None
    
    def _calculate_impact_score(self, price_change: float, volume_change: float) -> float:
        """
        Calculate an impact score on a scale from -5 to +5.
        
        Args:
            price_change: Percentage change in price
            volume_change: Percentage change in volume
            
        Returns:
            Impact score from -5 to +5
        """
        # Weight price change more heavily than volume change
        weighted_change = 0.7 * price_change + 0.3 * volume_change
        
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
            model: Neural network model
            dataloader: DataLoader with training data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            
            for inputs, targets in dataloader:
                # Move to GPU if available
                if self.use_gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
    
    def _save_impact_model(self) -> None:
        """
        Save the trained impact model.
        """
        try:
            # Create model path
            model_path = os.path.join(self.models_dir, f"impact_model_{self.impact_model_type}.joblib")
            
            if self.impact_model_type in ['random_forest', 'linear']:
                # Save scikit-learn model
                joblib.dump(self.impact_model, model_path)
                
                # Save features
                features_path = os.path.join(self.models_dir, "impact_model_features.joblib")
                joblib.dump(self.model_features, features_path)
                
            elif self.impact_model_type == 'neural':
                # Save PyTorch model
                torch.save(self.impact_model.state_dict(), model_path)
                
                # Save features
                features_path = os.path.join(self.models_dir, "impact_model_features.joblib")
                joblib.dump(self.model_features, features_path)
                
                # Save model architecture info
                info_path = os.path.join(self.models_dir, "impact_model_info.joblib")
                model_info = {
                    'input_size': self.impact_model.input_size,
                    'hidden_size': self.impact_model.hidden_size,
                    'output_size': self.impact_model.output_size
                }
                joblib.dump(model_info, info_path)
            
            logger.info(f"Saved impact model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving impact model: {str(e)}")
    
    def load_impact_model(self, model_type: str = None) -> bool:
        """
        Load a previously saved impact model.
        
        Args:
            model_type: Type of model to load ('random_forest', 'linear', 'neural')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine model type if not provided
            if model_type is None:
                # Try to find any saved model
                for mt in ['random_forest', 'linear', 'neural']:
                    model_path = os.path.join(self.models_dir, f"impact_model_{mt}.joblib")
                    if os.path.exists(model_path):
                        model_type = mt
                        break
                
                if model_type is None:
                    logger.warning("No saved impact model found")
                    return False
            
            # Load model based on type
            model_path = os.path.join(self.models_dir, f"impact_model_{model_type}.joblib")
            features_path = os.path.join(self.models_dir, "impact_model_features.joblib")
            
            if not os.path.exists(model_path):
                logger.warning(f"Impact model not found at {model_path}")
                return False
            
            if not os.path.exists(features_path):
                logger.warning(f"Model features not found at {features_path}")
                return False
            
            # Load features
            self.model_features = joblib.load(features_path)
            
            if model_type in ['random_forest', 'linear']:
                # Load scikit-learn model
                self.impact_model = joblib.load(model_path)
                
            elif model_type == 'neural':
                # Load model architecture info
                info_path = os.path.join(self.models_dir, "impact_model_info.joblib")
                if not os.path.exists(info_path):
                    logger.warning(f"Model info not found at {info_path}")
                    return False
                
                model_info = joblib.load(info_path)
                
                # Create model with same architecture
                self.impact_model = NeuralImpactModel(
                    input_size=model_info['input_size'],
                    hidden_size=model_info['hidden_size'],
                    output_size=model_info['output_size']
                )
                
                # Load weights
                self.impact_model.load_state_dict(torch.load(model_path))
                
                # Move to GPU if available
                if self.use_gpu:
                    self.impact_model = self.impact_model.cuda()
                
                # Set to evaluation mode
                self.impact_model.eval()
            
            self.impact_model_type = model_type
            logger.info(f"Loaded {model_type} impact model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading impact model: {str(e)}")
            return False
    
    def _save_impact_results(self, df: pd.DataFrame) -> None:
        """
        Save impact analysis results for later use.
        
        Args:
            df: DataFrame with impact scores
        """
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save results
            results_path = os.path.join(self.models_dir, f"impact_results_{timestamp}.csv")
            df.to_csv(results_path, index=False)
            
            logger.info(f"Saved impact results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving impact results: {str(e)}")


class NeuralImpactModel(nn.Module):
    """
    Neural network model for predicting news impact on stock prices.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        """
        Initialize the neural impact model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            output_size: Number of output values (typically 1 for regression)
        """
        super(NeuralImpactModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_news = pd.DataFrame({
        'Title': ['Samsung reports record profits', 'Samsung stock falls on weak guidance'],
        'Date': [pd.to_datetime('20250101'), pd.to_datetime('20250110')],
        'tickers': [['005930'], ['005930']]
    })
    
    # Sample stock data
    sample_stock_data = {
        '005930': pd.DataFrame({
            'Date': pd.date_range(start='20250101', periods=20),
            'End': [100, 102, 105, 103, 101, 99, 98, 96, 95, 97, 99, 100, 102, 104, 105, 107, 106, 105, 103, 102],
            'Volume': [1000000] * 20
        })
    }
    
    # Create analyzer
    analyzer = StockImpactAnalyzer()
    
    # Analyze impact
    results = analyzer.analyze_news_impact(sample_news, sample_stock_data)
    
    print("Impact analysis results:")
    print(results[['Title', 'Date', 'impact_score']])
    
    # Train impact model
    analyzer.train_impact_model(results, sample_stock_data, model_type='random_forest')
    
    # Predict impact
    predictions = analyzer.predict_impact(sample_news)
    
    print("\nImpact predictions:")
    print(predictions[['Title', 'Date', 'predicted_impact']])
    
    # Visualize impact
    viz_path = analyzer.visualize_impact(results, sample_stock_data, '005930')
    print(f"\nVisualization saved to: {viz_path}")
