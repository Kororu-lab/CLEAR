"""
Advanced stock impact analyzer module for analyzing the impact of news on stock prices.
Extends the base stock impact analyzer with KO-finbert sentiment analysis and advanced metrics.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import joblib
import json
import re
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedStockImpactAnalyzer:
    """
    Advanced analyzer for measuring the impact of news articles on stock prices.
    Extends the base StockImpactAnalyzer with more sophisticated analysis techniques.
    """
    
    def __init__(self, 
                 stock_data_dir: str = None,
                 models_dir: str = None,
                 company_map_path: str = None,
                 use_finbert: bool = True,
                 use_volatility: bool = True,
                 use_market_trend: bool = True,
                 time_window_days: int = 3,
                 impact_threshold: float = 0.02,
                 sentiment_weight: float = 0.3):
        """
        Initialize the advanced stock impact analyzer.
        
        Args:
            stock_data_dir: Directory containing stock price data
            models_dir: Directory to save/load models
            company_map_path: Path to company-to-ticker mapping file
            use_finbert: Whether to use KO-finbert for sentiment analysis
            use_volatility: Whether to incorporate volatility in impact calculation
            use_market_trend: Whether to consider market trends in impact calculation
            time_window_days: Number of days to look for price changes after news
            impact_threshold: Threshold for significant price changes (percentage)
            sentiment_weight: Weight of sentiment in impact calculation
        """
        self.stock_data_dir = stock_data_dir or os.path.join(os.getcwd(), "data", "stock")
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "stock_impact")
        self.company_map_path = company_map_path or os.path.join(os.getcwd(), "data", "company_map.json")
        self.use_finbert = use_finbert
        self.use_volatility = use_volatility
        self.use_market_trend = use_market_trend
        self.time_window_days = time_window_days
        self.impact_threshold = impact_threshold
        self.sentiment_weight = sentiment_weight
        
        # Create directories if they don't exist
        os.makedirs(self.stock_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load company-to-ticker mapping
        self.company_to_ticker = self._load_company_map()
        
        # Initialize finbert model if enabled
        self.finbert_model = None
        self.finbert_tokenizer = None
        if self.use_finbert:
            self._initialize_finbert()
        
        logger.info(f"Initialized AdvancedStockImpactAnalyzer with time_window_days={time_window_days}")
    
    def _initialize_finbert(self):
        """
        Initialize the KO-finbert model for sentiment analysis.
        """
        try:
            # Import here to avoid dependency if not using finbert
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Load KO-finbert model and tokenizer
            model_name = "snunlp/KR-FinBert-SC"
            
            logger.info(f"Loading KO-finbert model: {model_name}")
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.finbert_model.eval()
            
            logger.info("KO-finbert model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KO-finbert: {str(e)}")
            logger.warning("Falling back to rule-based sentiment analysis")
            self.use_finbert = False
    
    def _load_company_map(self) -> Dict[str, str]:
        """
        Load company-to-ticker mapping from file.
        
        Returns:
            Dictionary mapping company names to ticker symbols
        """
        # Default mapping for common Korean companies
        default_map = {
            "삼성전자": "005930",
            "SK하이닉스": "000660",
            "NAVER": "035420",
            "카카오": "035720",
            "현대자동차": "005380",
            "기아": "000270",
            "LG화학": "051910",
            "셀트리온": "068270",
            "삼성바이오로직스": "207940",
            "삼성SDI": "006400",
            "LG전자": "066570",
            "포스코": "005490",
            "신한지주": "055550",
            "KB금융": "105560",
            "삼성물산": "028260",
            "SK이노베이션": "096770",
            "LG생활건강": "051900",
            "SK텔레콤": "017670",
            "삼성생명": "032830",
            "한국전력": "015760"
        }
        
        try:
            # Check if mapping file exists
            if os.path.exists(self.company_map_path):
                with open(self.company_map_path, 'r', encoding='utf-8') as f:
                    company_map = json.load(f)
                logger.info(f"Loaded company map from {self.company_map_path}")
                return company_map
            else:
                # Save default mapping
                with open(self.company_map_path, 'w', encoding='utf-8') as f:
                    json.dump(default_map, f, ensure_ascii=False, indent=2)
                logger.info(f"Created default company map at {self.company_map_path}")
                return default_map
        except Exception as e:
            logger.error(f"Error loading company map: {str(e)}")
            return default_map
    
    def analyze_news_impact(self, news_df: pd.DataFrame,
                          stock_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Analyze the impact of news articles on stock prices.
        
        Args:
            news_df: DataFrame containing news articles
            stock_data: Optional dictionary mapping tickers to stock DataFrames
            
        Returns:
            DataFrame with impact scores
        """
        if len(news_df) == 0:
            logger.warning("Empty DataFrame provided for impact analysis")
            return news_df
            
        logger.info(f"Analyzing impact for {len(news_df)} news articles")
        
        try:
            # Create a copy to avoid modifying the original
            df = news_df.copy()
            
            # Convert Date column to datetime if needed
            if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
                # Handle the specific date format (20250101 18:56)
                df['Date'] = df['Date'].apply(self._parse_date)
            
            # Extract tickers from news articles
            df = self._extract_tickers(df)
            
            # Load stock data if not provided
            if stock_data is None:
                stock_data = self._load_stock_data(df['tickers'].explode().unique())
            
            # Calculate sentiment scores
            df = self._calculate_sentiment(df)
            
            # Calculate price changes
            df = self._calculate_price_changes(df, stock_data)
            
            # Calculate volatility if enabled
            if self.use_volatility:
                df = self._calculate_volatility(df, stock_data)
            
            # Calculate market trend correlation if enabled
            if self.use_market_trend:
                df = self._calculate_market_correlation(df, stock_data)
            
            # Calculate impact scores
            df = self._calculate_impact_scores(df)
            
            logger.info(f"Completed impact analysis for {len(df)} articles")
            
            return df
        except Exception as e:
            logger.error(f"Error analyzing news impact: {str(e)}")
            raise
    
    def _parse_date(self, date_str):
        """
        Parse date string in format '20250101 18:56' to datetime.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed datetime object
        """
        try:
            if isinstance(date_str, pd.Timestamp):
                return date_str
            
            # Convert to string if not already
            date_str = str(date_str).strip()
            
            # Handle format with time component (20250101 18:56)
            if ' ' in date_str:
                date_part, time_part = date_str.split(' ', 1)
                # Parse date part
                year = int(date_part[:4])
                month = int(date_part[4:6])
                day = int(date_part[6:8])
                
                # Parse time part if available
                if ':' in time_part:
                    hour, minute = map(int, time_part.split(':'))
                    return pd.Timestamp(year, month, day, hour, minute)
                else:
                    return pd.Timestamp(year, month, day)
            else:
                # Handle format without time component (20250101)
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                return pd.Timestamp(year, month, day)
        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {str(e)}")
            # Return current date as fallback
            return pd.Timestamp.now()
    
    def _extract_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract stock tickers from news articles.
        
        Args:
            df: DataFrame containing news articles
            
        Returns:
            DataFrame with extracted tickers
        """
        try:
            # Initialize tickers column
            df['tickers'] = None
            
            # Create reverse mapping (ticker to company)
            ticker_to_company = {v: k for k, v in self.company_to_ticker.items()}
            
            # Process each article
            for idx, row in df.iterrows():
                tickers = set()
                
                # Check title for company names
                if 'Title' in row and isinstance(row['Title'], str):
                    for company, ticker in self.company_to_ticker.items():
                        if company in row['Title']:
                            tickers.add(ticker)
                
                # Check body for company names
                if 'Body' in row and isinstance(row['Body'], str):
                    for company, ticker in self.company_to_ticker.items():
                        if company in row['Body']:
                            tickers.add(ticker)
                
                # Check for direct ticker mentions
                if 'Title' in row and isinstance(row['Title'], str):
                    for ticker in ticker_to_company.keys():
                        if ticker in row['Title'].split():
                            tickers.add(ticker)
                
                if 'Body' in row and isinstance(row['Body'], str):
                    for ticker in ticker_to_company.keys():
                        if ticker in row['Body'].split():
                            tickers.add(ticker)
                
                # If no tickers found but article is about Samsung Electronics (005930)
                # This is a fallback for the sample data which is focused on Samsung
                if not tickers and '005930' in df.iloc[0]['Link']:
                    tickers.add('005930')
                
                # Store tickers as list
                df.at[idx, 'tickers'] = list(tickers) if tickers else ['005930']  # Default to Samsung if no tickers found
            
            # Count articles with tickers
            ticker_count = df['tickers'].apply(lambda x: x is not None and len(x) > 0).sum()
            logger.info(f"Extracted tickers for {ticker_count} out of {len(df)} articles")
            
            return df
        except Exception as e:
            logger.error(f"Error extracting tickers: {str(e)}")
            return df
    
    def _load_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load stock price data for the specified tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping tickers to stock DataFrames
        """
        stock_data = {}
        
        for ticker in tickers:
            if ticker is None:
                continue
                
            try:
                # Construct file path
                file_path = os.path.join(self.stock_data_dir, f"stockprice_{ticker}.csv")
                
                if os.path.exists(file_path):
                    # Load stock data
                    stock_df = pd.read_csv(file_path)
                    
                    # Convert date column to datetime
                    if 'Date' in stock_df.columns:
                        stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y%m%d')
                    
                    # Ignore Time column as it contains zeros
                    if 'Time' in stock_df.columns:
                        stock_df = stock_df.drop('Time', axis=1)
                    
                    # Store in dictionary
                    stock_data[ticker] = stock_df
                    logger.info(f"Loaded stock data for {ticker}")
                else:
                    logger.warning(f"Stock data file not found for {ticker}: {file_path}")
            except Exception as e:
                logger.error(f"Error loading stock data for {ticker}: {str(e)}")
        
        logger.info(f"Loaded stock data for {len(stock_data)} tickers")
        return stock_data
    
    def _calculate_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sentiment scores for news articles.
        
        Args:
            df: DataFrame containing news articles
            
        Returns:
            DataFrame with sentiment scores
        """
        try:
            # Initialize sentiment columns
            df['sentiment_score'] = 0.0
            df['sentiment_label'] = 'neutral'
            
            # Use KO-finbert if enabled
            if self.use_finbert and self.finbert_model is not None:
                logger.info("Using KO-finbert for sentiment analysis")
                
                import torch
                
                # Process each article
                for idx, row in df.iterrows():
                    # Prepare text (prefer title for more reliable sentiment)
                    if 'Title' in row and isinstance(row['Title'], str):
                        text = row['Title']
                    elif 'AI Summary' in row and isinstance(row['AI Summary'], str) and row['AI Summary']:
                        text = row['AI Summary']
                    elif 'Body' in row and isinstance(row['Body'], str):
                        # Use first 512 characters of body if no title
                        text = row['Body'][:512]
                    else:
                        continue
                    
                    try:
                        # Tokenize text
                        inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                        
                        # Get sentiment prediction
                        with torch.no_grad():
                            outputs = self.finbert_model(**inputs)
                            predictions = outputs.logits.softmax(dim=1)
                        
                        # Get sentiment score and label
                        # KR-FinBert-SC has 3 classes: negative (0), neutral (1), positive (2)
                        probs = predictions[0].tolist()
                        
                        # Convert to -1 to 1 scale
                        sentiment_score = probs[2] - probs[0]  # positive - negative
                        
                        # Get label
                        label_idx = predictions[0].argmax().item()
                        labels = ['negative', 'neutral', 'positive']
                        sentiment_label = labels[label_idx]
                        
                        # Store results
                        df.at[idx, 'sentiment_score'] = sentiment_score
                        df.at[idx, 'sentiment_label'] = sentiment_label
                    except Exception as e:
                        logger.error(f"Error calculating sentiment for article {idx}: {str(e)}")
            else:
                # Fallback to rule-based sentiment analysis
                logger.info("Using rule-based sentiment analysis")
                
                # Define positive and negative keywords
                positive_keywords = [
                    '상승', '급등', '호조', '성장', '개선', '흑자', '최대', '신기록', '돌파',
                    '매출 증가', '이익 증가', '실적 개선', '호실적', '기대', '전망', '성공',
                    '계약', '수주', '인수', '합병', '협력', '파트너십', '출시', '개발 성공'
                ]
                
                negative_keywords = [
                    '하락', '급락', '부진', '감소', '악화', '적자', '손실', '하향', '부담',
                    '매출 감소', '이익 감소', '실적 악화', '저조', '우려', '리스크', '실패',
                    '소송', '제재', '벌금', '파산', '구조조정', '감원', '철수', '중단'
                ]
                
                # Process each article
                for idx, row in df.iterrows():
                    # Prepare text
                    text = ""
                    if 'Title' in row and isinstance(row['Title'], str):
                        text += row['Title'] + " "
                    if 'AI Summary' in row and isinstance(row['AI Summary'], str):
                        text += row['AI Summary'] + " "
                    
                    # Count positive and negative keywords
                    positive_count = sum(1 for keyword in positive_keywords if keyword in text)
                    negative_count = sum(1 for keyword in negative_keywords if keyword in text)
                    
                    # Calculate sentiment score
                    total_count = positive_count + negative_count
                    if total_count > 0:
                        sentiment_score = (positive_count - negative_count) / total_count
                    else:
                        sentiment_score = 0.0
                    
                    # Determine sentiment label
                    if sentiment_score > 0.2:
                        sentiment_label = 'positive'
                    elif sentiment_score < -0.2:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'
                    
                    # Store results
                    df.at[idx, 'sentiment_score'] = sentiment_score
                    df.at[idx, 'sentiment_label'] = sentiment_label
            
            # Count sentiment distribution
            sentiment_counts = df['sentiment_label'].value_counts()
            logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
            
            return df
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return df
    
    def _calculate_price_changes(self, df: pd.DataFrame, 
                               stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate price changes after news articles.
        
        Args:
            df: DataFrame containing news articles
            stock_data: Dictionary mapping tickers to stock DataFrames
            
        Returns:
            DataFrame with price change information
        """
        try:
            # Initialize price change columns
            df['price_change_1d'] = None
            df['price_change_3d'] = None
            df['price_change_5d'] = None
            df['price_change_pct_1d'] = None
            df['price_change_pct_3d'] = None
            df['price_change_pct_5d'] = None
            df['volume_change_pct_1d'] = None
            
            # Process each article
            for idx, row in df.iterrows():
                # Skip if no tickers or no date
                if not row.get('tickers') or 'Date' not in row:
                    continue
                
                # Convert date to datetime if needed
                article_date = row['Date']
                if not isinstance(article_date, pd.Timestamp):
                    article_date = self._parse_date(article_date)
                
                # Calculate price changes for each ticker
                ticker_changes = {}
                
                for ticker in row['tickers']:
                    if ticker not in stock_data:
                        continue
                    
                    stock_df = stock_data[ticker]
                    
                    # Find closest trading day on or after article date
                    trading_days = stock_df[stock_df['Date'] >= article_date]['Date'].sort_values()
                    
                    if len(trading_days) < 2:
                        # Not enough data
                        continue
                    
                    # Get starting price (first trading day on or after article date)
                    start_date = trading_days.iloc[0]
                    start_price = stock_df[stock_df['Date'] == start_date]['End'].iloc[0]
                    start_volume = stock_df[stock_df['Date'] == start_date]['Volume'].iloc[0]
                    
                    # Calculate price changes for different time windows
                    changes = {}
                    
                    # 1-day change
                    if len(trading_days) >= 2:
                        end_date_1d = trading_days.iloc[1]
                        end_price_1d = stock_df[stock_df['Date'] == end_date_1d]['End'].iloc[0]
                        end_volume_1d = stock_df[stock_df['Date'] == end_date_1d]['Volume'].iloc[0]
                        
                        price_change_1d = end_price_1d - start_price
                        price_change_pct_1d = price_change_1d / start_price * 100
                        volume_change_pct_1d = (end_volume_1d - start_volume) / start_volume * 100
                        
                        changes['1d'] = {
                            'price_change': price_change_1d,
                            'price_change_pct': price_change_pct_1d,
                            'volume_change_pct': volume_change_pct_1d
                        }
                    
                    # 3-day change
                    if len(trading_days) >= 4:
                        end_date_3d = trading_days.iloc[3]
                        end_price_3d = stock_df[stock_df['Date'] == end_date_3d]['End'].iloc[0]
                        
                        price_change_3d = end_price_3d - start_price
                        price_change_pct_3d = price_change_3d / start_price * 100
                        
                        changes['3d'] = {
                            'price_change': price_change_3d,
                            'price_change_pct': price_change_pct_3d
                        }
                    
                    # 5-day change
                    if len(trading_days) >= 6:
                        end_date_5d = trading_days.iloc[5]
                        end_price_5d = stock_df[stock_df['Date'] == end_date_5d]['End'].iloc[0]
                        
                        price_change_5d = end_price_5d - start_price
                        price_change_pct_5d = price_change_5d / start_price * 100
                        
                        changes['5d'] = {
                            'price_change': price_change_5d,
                            'price_change_pct': price_change_pct_5d
                        }
                    
                    ticker_changes[ticker] = changes
                
                # Aggregate changes across tickers
                if ticker_changes:
                    # Calculate average changes
                    avg_changes = {
                        'price_change_1d': np.mean([changes['1d']['price_change'] 
                                                  for ticker, changes in ticker_changes.items() 
                                                  if '1d' in changes]),
                        'price_change_3d': np.mean([changes['3d']['price_change'] 
                                                  for ticker, changes in ticker_changes.items() 
                                                  if '3d' in changes]),
                        'price_change_5d': np.mean([changes['5d']['price_change'] 
                                                  for ticker, changes in ticker_changes.items() 
                                                  if '5d' in changes]),
                        'price_change_pct_1d': np.mean([changes['1d']['price_change_pct'] 
                                                      for ticker, changes in ticker_changes.items() 
                                                      if '1d' in changes]),
                        'price_change_pct_3d': np.mean([changes['3d']['price_change_pct'] 
                                                      for ticker, changes in ticker_changes.items() 
                                                      if '3d' in changes]),
                        'price_change_pct_5d': np.mean([changes['5d']['price_change_pct'] 
                                                      for ticker, changes in ticker_changes.items() 
                                                      if '5d' in changes]),
                        'volume_change_pct_1d': np.mean([changes['1d']['volume_change_pct'] 
                                                       for ticker, changes in ticker_changes.items() 
                                                       if '1d' in changes])
                    }
                    
                    # Store in DataFrame
                    for key, value in avg_changes.items():
                        if not np.isnan(value):
                            df.at[idx, key] = value
            
            # Count articles with price changes
            price_change_count = df['price_change_1d'].notna().sum()
            logger.info(f"Calculated price changes for {price_change_count} out of {len(df)} articles")
            
            return df
        except Exception as e:
            logger.error(f"Error calculating price changes: {str(e)}")
            return df
    
    def _calculate_volatility(self, df: pd.DataFrame, 
                            stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate stock volatility metrics.
        
        Args:
            df: DataFrame containing news articles
            stock_data: Dictionary mapping tickers to stock DataFrames
            
        Returns:
            DataFrame with volatility metrics
        """
        try:
            # Initialize volatility columns
            df['pre_volatility'] = None
            df['post_volatility'] = None
            df['volatility_change'] = None
            
            # Process each article
            for idx, row in df.iterrows():
                # Skip if no tickers or no date
                if not row.get('tickers') or 'Date' not in row:
                    continue
                
                # Convert date to datetime if needed
                article_date = row['Date']
                if not isinstance(article_date, pd.Timestamp):
                    article_date = self._parse_date(article_date)
                
                # Calculate volatility for each ticker
                ticker_volatilities = {}
                
                for ticker in row['tickers']:
                    if ticker not in stock_data:
                        continue
                    
                    stock_df = stock_data[ticker]
                    
                    # Calculate pre-news volatility (10 trading days before)
                    pre_period_end = article_date
                    pre_period_start = article_date - timedelta(days=20)  # More days to ensure enough trading days
                    
                    pre_period_data = stock_df[(stock_df['Date'] >= pre_period_start) & 
                                             (stock_df['Date'] <= pre_period_end)]
                    
                    if len(pre_period_data) >= 5:
                        # Calculate daily returns
                        pre_period_data['daily_return'] = pre_period_data['End'].pct_change()
                        
                        # Calculate volatility (standard deviation of returns)
                        pre_volatility = pre_period_data['daily_return'].std() * 100  # as percentage
                    else:
                        pre_volatility = None
                    
                    # Calculate post-news volatility (10 trading days after)
                    post_period_start = article_date
                    post_period_end = article_date + timedelta(days=20)  # More days to ensure enough trading days
                    
                    post_period_data = stock_df[(stock_df['Date'] >= post_period_start) & 
                                              (stock_df['Date'] <= post_period_end)]
                    
                    if len(post_period_data) >= 5:
                        # Calculate daily returns
                        post_period_data['daily_return'] = post_period_data['End'].pct_change()
                        
                        # Calculate volatility (standard deviation of returns)
                        post_volatility = post_period_data['daily_return'].std() * 100  # as percentage
                    else:
                        post_volatility = None
                    
                    # Calculate volatility change
                    if pre_volatility is not None and post_volatility is not None:
                        volatility_change = post_volatility - pre_volatility
                    else:
                        volatility_change = None
                    
                    ticker_volatilities[ticker] = {
                        'pre_volatility': pre_volatility,
                        'post_volatility': post_volatility,
                        'volatility_change': volatility_change
                    }
                
                # Aggregate volatilities across tickers
                if ticker_volatilities:
                    # Calculate average volatilities
                    pre_volatilities = [v['pre_volatility'] for ticker, v in ticker_volatilities.items() if v['pre_volatility'] is not None]
                    post_volatilities = [v['post_volatility'] for ticker, v in ticker_volatilities.items() if v['post_volatility'] is not None]
                    volatility_changes = [v['volatility_change'] for ticker, v in ticker_volatilities.items() if v['volatility_change'] is not None]
                    
                    if pre_volatilities:
                        df.at[idx, 'pre_volatility'] = np.mean(pre_volatilities)
                    
                    if post_volatilities:
                        df.at[idx, 'post_volatility'] = np.mean(post_volatilities)
                    
                    if volatility_changes:
                        df.at[idx, 'volatility_change'] = np.mean(volatility_changes)
            
            # Count articles with volatility metrics
            volatility_count = df['volatility_change'].notna().sum()
            logger.info(f"Calculated volatility metrics for {volatility_count} out of {len(df)} articles")
            
            return df
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return df
    
    def _calculate_market_correlation(self, df: pd.DataFrame, 
                                    stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate correlation with market trends.
        
        Args:
            df: DataFrame containing news articles
            stock_data: Dictionary mapping tickers to stock DataFrames
            
        Returns:
            DataFrame with market correlation metrics
        """
        try:
            # Initialize market correlation columns
            df['market_correlation'] = None
            df['market_beta'] = None
            
            # Use KOSPI as market index (if available)
            market_ticker = None
            for ticker in ['KOSPI', 'KS11', '001', 'KRX']:
                if ticker in stock_data:
                    market_ticker = ticker
                    break
            
            if market_ticker is None:
                logger.warning("Market index data not found, skipping market correlation calculation")
                return df
            
            market_df = stock_data[market_ticker]
            
            # Process each article
            for idx, row in df.iterrows():
                # Skip if no tickers or no date
                if not row.get('tickers') or 'Date' not in row:
                    continue
                
                # Convert date to datetime if needed
                article_date = row['Date']
                if not isinstance(article_date, pd.Timestamp):
                    article_date = self._parse_date(article_date)
                
                # Calculate correlation for each ticker
                ticker_correlations = {}
                
                for ticker in row['tickers']:
                    if ticker not in stock_data or ticker == market_ticker:
                        continue
                    
                    stock_df = stock_data[ticker]
                    
                    # Define analysis period (30 days after news)
                    period_start = article_date
                    period_end = article_date + timedelta(days=30)
                    
                    # Get stock data for the period
                    stock_period = stock_df[(stock_df['Date'] >= period_start) & 
                                          (stock_df['Date'] <= period_end)]
                    
                    # Get market data for the period
                    market_period = market_df[(market_df['Date'] >= period_start) & 
                                            (market_df['Date'] <= period_end)]
                    
                    # Merge on Date
                    if len(stock_period) >= 5 and len(market_period) >= 5:
                        merged = pd.merge(
                            stock_period[['Date', 'End']].rename(columns={'End': 'stock_price'}),
                            market_period[['Date', 'End']].rename(columns={'End': 'market_price'}),
                            on='Date'
                        )
                        
                        if len(merged) >= 5:
                            # Calculate daily returns
                            merged['stock_return'] = merged['stock_price'].pct_change()
                            merged['market_return'] = merged['market_price'].pct_change()
                            
                            # Drop first row (NaN returns)
                            merged = merged.dropna()
                            
                            if len(merged) >= 4:
                                # Calculate correlation
                                correlation = merged['stock_return'].corr(merged['market_return'])
                                
                                # Calculate beta
                                covariance = merged['stock_return'].cov(merged['market_return'])
                                market_variance = merged['market_return'].var()
                                beta = covariance / market_variance if market_variance > 0 else None
                                
                                ticker_correlations[ticker] = {
                                    'correlation': correlation,
                                    'beta': beta
                                }
                
                # Aggregate correlations across tickers
                if ticker_correlations:
                    # Calculate average correlations
                    correlations = [c['correlation'] for ticker, c in ticker_correlations.items() if not np.isnan(c['correlation'])]
                    betas = [c['beta'] for ticker, c in ticker_correlations.items() if c['beta'] is not None and not np.isnan(c['beta'])]
                    
                    if correlations:
                        df.at[idx, 'market_correlation'] = np.mean(correlations)
                    
                    if betas:
                        df.at[idx, 'market_beta'] = np.mean(betas)
            
            # Count articles with market correlation
            correlation_count = df['market_correlation'].notna().sum()
            logger.info(f"Calculated market correlation for {correlation_count} out of {len(df)} articles")
            
            return df
        except Exception as e:
            logger.error(f"Error calculating market correlation: {str(e)}")
            return df
    
    def _calculate_impact_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall impact scores based on price changes, sentiment, and other metrics.
        
        Args:
            df: DataFrame containing news articles with price changes and sentiment
            
        Returns:
            DataFrame with impact scores
        """
        try:
            # Initialize impact score columns
            df['impact_price'] = None
            df['impact_volatility'] = None
            df['impact_sentiment'] = None
            df['impact_overall'] = None
            
            # Process each article
            for idx, row in df.iterrows():
                # Calculate price impact
                price_impact = None
                
                # Try different time windows for price impact
                if pd.notna(row.get('price_change_pct_1d')):
                    # Normalize to -5 to 5 scale
                    price_impact = row['price_change_pct_1d'] / 2  # 10% change -> 5 impact
                elif pd.notna(row.get('price_change_pct_3d')):
                    # Use 3-day change if 1-day not available
                    price_impact = row['price_change_pct_3d'] / 3  # 15% change -> 5 impact
                elif pd.notna(row.get('price_change_pct_5d')):
                    # Use 5-day change if others not available
                    price_impact = row['price_change_pct_5d'] / 5  # 25% change -> 5 impact
                
                # Store price impact if calculated
                if price_impact is not None:
                    # Clip to range
                    price_impact = max(min(price_impact, 5), -5)
                    df.at[idx, 'impact_price'] = price_impact
                
                # Calculate volatility impact
                if self.use_volatility and pd.notna(row.get('volatility_change')):
                    # Normalize to -5 to 5 scale (assuming max volatility change of 5%)
                    volatility_impact = row['volatility_change']
                    
                    # Clip to range
                    volatility_impact = max(min(volatility_impact, 5), -5)
                    
                    df.at[idx, 'impact_volatility'] = volatility_impact
                
                # Calculate sentiment impact
                if pd.notna(row.get('sentiment_score')):
                    # Convert from -1 to 1 scale to -5 to 5 scale
                    sentiment_impact = row['sentiment_score'] * 5
                    
                    df.at[idx, 'impact_sentiment'] = sentiment_impact
                
                # Calculate overall impact
                impact_components = []
                weights = []
                
                # Add price impact if available
                if pd.notna(row.get('impact_price')):
                    impact_components.append(row['impact_price'])
                    weights.append(1 - self.sentiment_weight)
                
                # Add sentiment impact if available
                if pd.notna(row.get('impact_sentiment')):
                    impact_components.append(row['impact_sentiment'])
                    weights.append(self.sentiment_weight)
                
                # Add volatility impact if available
                if pd.notna(row.get('impact_volatility')):
                    impact_components.append(row['impact_volatility'])
                    weights.append(0.1)
                
                # Calculate weighted average if components exist
                if impact_components:
                    # Normalize weights
                    total_weight = sum(weights)
                    normalized_weights = [w / total_weight for w in weights]
                    
                    # Calculate weighted average
                    overall_impact = sum(c * w for c, w in zip(impact_components, normalized_weights))
                    
                    # Clip to range
                    overall_impact = max(min(overall_impact, 5), -5)
                    
                    df.at[idx, 'impact_overall'] = overall_impact
                else:
                    # Fallback to sentiment only if no other components
                    if pd.notna(row.get('sentiment_score')):
                        df.at[idx, 'impact_overall'] = row['sentiment_score'] * 5
                    else:
                        # Default neutral impact if no data available
                        df.at[idx, 'impact_overall'] = 0.0
            
            # Count articles with impact scores
            impact_count = df['impact_overall'].notna().sum()
            logger.info(f"Calculated impact scores for {impact_count} out of {len(df)} articles")
            
            return df
        except Exception as e:
            logger.error(f"Error calculating impact scores: {str(e)}")
            return df
    
    def visualize_impact_distribution(self, df: pd.DataFrame, 
                                    save_path: Optional[str] = None) -> None:
        """
        Visualize the distribution of impact scores.
        
        Args:
            df: DataFrame containing news articles with impact scores
            save_path: Path to save the visualization
        """
        if 'impact_overall' not in df.columns or df['impact_overall'].isna().all():
            logger.warning("No impact scores available for visualization")
            return
            
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot impact distribution
            sns.histplot(df['impact_overall'].dropna(), bins=20, kde=True)
            
            plt.title('Distribution of News Impact Scores')
            plt.xlabel('Impact Score (-5 to 5)')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
            
            # Add vertical line at zero
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Add mean and median lines
            mean_impact = df['impact_overall'].mean()
            median_impact = df['impact_overall'].median()
            
            plt.axvline(x=mean_impact, color='green', linestyle='-', alpha=0.7, 
                       label=f'Mean: {mean_impact:.2f}')
            plt.axvline(x=median_impact, color='blue', linestyle='-', alpha=0.7,
                       label=f'Median: {median_impact:.2f}')
            
            plt.legend()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved impact distribution visualization to {save_path}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Error visualizing impact distribution: {str(e)}")
    
    def visualize_impact_over_time(self, df: pd.DataFrame, 
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize impact scores over time.
        
        Args:
            df: DataFrame containing news articles with impact scores
            save_path: Path to save the visualization
        """
        if 'impact_overall' not in df.columns or df['impact_overall'].isna().all() or 'Date' not in df.columns:
            logger.warning("No impact scores or dates available for visualization")
            return
            
        try:
            # Create a copy with datetime index
            plot_df = df.copy()
            
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(plot_df['Date']):
                plot_df['Date'] = plot_df['Date'].apply(self._parse_date)
            
            # Sort by date
            plot_df = plot_df.sort_values('Date')
            
            # Create figure
            plt.figure(figsize=(14, 8))
            
            # Plot impact over time
            plt.scatter(plot_df['Date'], plot_df['impact_overall'], alpha=0.7)
            
            # Add trend line
            if len(plot_df) > 1:
                try:
                    # Convert dates to ordinal for regression
                    x = np.array([d.toordinal() for d in plot_df['Date']])
                    y = plot_df['impact_overall'].values
                    
                    # Fit trend line
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Plot trend line
                    plt.plot(plot_df['Date'], p(x), "r--", alpha=0.7, 
                           label=f'Trend: {z[0]:.2e}x + {z[1]:.2f}')
                    
                    plt.legend()
                except Exception as e:
                    logger.warning(f"Could not calculate trend line: {str(e)}")
            
            plt.title('News Impact Scores Over Time')
            plt.xlabel('Date')
            plt.ylabel('Impact Score (-5 to 5)')
            plt.grid(True, alpha=0.3)
            
            # Add horizontal line at zero
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Format x-axis
            plt.gcf().autofmt_xdate()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved impact over time visualization to {save_path}")
            
            plt.close()
        except Exception as e:
            logger.error(f"Error visualizing impact over time: {str(e)}")
    
    def get_top_impactful_news(self, df: pd.DataFrame, n: int = 10, 
                             impact_type: str = 'overall') -> pd.DataFrame:
        """
        Get the top most impactful news articles.
        
        Args:
            df: DataFrame containing news articles with impact scores
            n: Number of articles to return
            impact_type: Type of impact to sort by ('overall', 'positive', 'negative')
            
        Returns:
            DataFrame with top impactful news
        """
        if 'impact_overall' not in df.columns or df['impact_overall'].isna().all():
            logger.warning("No impact scores available")
            return pd.DataFrame()
            
        try:
            # Create a copy
            result_df = df.copy()
            
            # Filter based on impact type
            if impact_type == 'positive':
                result_df = result_df[result_df['impact_overall'] > 0]
                sort_ascending = False
            elif impact_type == 'negative':
                result_df = result_df[result_df['impact_overall'] < 0]
                sort_ascending = True
            else:
                # Sort by absolute impact
                result_df['abs_impact'] = result_df['impact_overall'].abs()
                sort_ascending = False
                
            # Sort by impact
            if impact_type == 'overall' and 'abs_impact' in result_df.columns:
                result_df = result_df.sort_values('abs_impact', ascending=sort_ascending)
            else:
                result_df = result_df.sort_values('impact_overall', ascending=sort_ascending)
            
            # Select columns
            columns = ['Title', 'Date', 'Press', 'impact_overall', 'sentiment_label', 'price_change_pct_1d']
            
            # Add additional columns if available
            for col in ['Link', 'tickers', 'impact_price', 'impact_sentiment']:
                if col in result_df.columns:
                    columns.append(col)
            
            # Return top n
            return result_df[columns].head(n)
        except Exception as e:
            logger.error(f"Error getting top impactful news: {str(e)}")
            return pd.DataFrame()
