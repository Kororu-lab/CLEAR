"""
Stock data collector module for retrieving and processing stock price data.
"""

import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    Stock data collector for retrieving and processing stock price data.
    Ensures compatibility with the existing data format.
    """
    
    def __init__(self, output_dir: str = None, market: str = "KRX"):
        """
        Initialize the stock data collector.
        
        Args:
            output_dir: Directory to save stock data
            market: Stock market (default: KRX for Korean Exchange)
        """
        self.output_dir = output_dir or os.path.join(os.getcwd(), "data", "stock")
        self.market = market
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized StockDataCollector with output_dir={self.output_dir}, market={market}")
    
    def get_stock_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol (e.g., "005930.KS" for Samsung in KRX)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo, etc.)
            
        Returns:
            DataFrame with stock price data
        """
        # Add market suffix if not present
        if self.market == "KRX" and ".KS" not in ticker:
            ticker = f"{ticker}.KS"
            
        try:
            logger.info(f"Retrieving stock data for {ticker} from {start_date} to {end_date}")
            
            # Get data from Yahoo Finance
            stock_data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if stock_data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
                
            logger.info(f"Retrieved {len(stock_data)} records for {ticker}")
            return stock_data
            
        except Exception as e:
            logger.error(f"Error retrieving stock data: {str(e)}")
            return pd.DataFrame()
    
    def process_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process stock data to match the required format.
        
        Args:
            data: Raw stock data from Yahoo Finance
            
        Returns:
            Processed DataFrame with the required format
        """
        if data.empty:
            return pd.DataFrame()
            
        try:
            logger.info("Processing stock data")
            
            # Reset index to make Date a column
            df = data.reset_index()
            
            # Rename columns to match required format
            df = df.rename(columns={
                'Date': 'Date',
                'Open': 'Start',
                'High': 'High',
                'Low': 'Low',
                'Close': 'End',
                'Volume': 'Volume'
            })
            
            # Format Date as YYYYMMDD
            df['Date'] = df['Date'].dt.strftime('%Y%m%d')
            
            # Add Time column (0 for daily data)
            df['Time'] = 0
            
            # Select and reorder columns to match required format
            result_df = df[['Date', 'Time', 'Start', 'High', 'Low', 'End', 'Volume']]
            
            # Calculate additional metrics
            result_df = self.calculate_additional_metrics(result_df)
            
            logger.info(f"Processed {len(result_df)} records")
            return result_df
            
        except Exception as e:
            logger.error(f"Error processing stock data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_additional_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional metrics for stock data analysis.
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            DataFrame with additional metrics
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate price changes
            result_df['Price_Change'] = result_df['End'] - result_df['Start']
            result_df['Price_Change_Pct'] = (result_df['Price_Change'] / result_df['Start']) * 100
            
            # Calculate price range
            result_df['Price_Range'] = result_df['High'] - result_df['Low']
            result_df['Price_Range_Pct'] = (result_df['Price_Range'] / result_df['Start']) * 100
            
            # Calculate moving averages (5-day and 20-day)
            result_df['MA5'] = result_df['End'].rolling(window=5).mean()
            result_df['MA20'] = result_df['End'].rolling(window=20).mean()
            
            # Calculate volume changes
            result_df['Volume_Change'] = result_df['Volume'].pct_change() * 100
            
            # Calculate volatility (standard deviation of price changes over 5 days)
            result_df['Volatility'] = result_df['Price_Change_Pct'].rolling(window=5).std()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {str(e)}")
            return df
    
    def save_stock_data(self, data: pd.DataFrame, ticker: str) -> str:
        """
        Save stock data to CSV file.
        
        Args:
            data: Processed stock data DataFrame
            ticker: Stock ticker symbol
            
        Returns:
            Path to the saved file
        """
        if data.empty:
            logger.warning(f"No data to save for {ticker}")
            return ""
            
        # Clean ticker for filename (remove market suffix)
        clean_ticker = ticker.split('.')[0]
        
        # Create filename
        filename = f"stockprice_{clean_ticker}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Save to CSV
            data.to_csv(filepath, index=False)
            logger.info(f"Saved stock data to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving stock data: {str(e)}")
            return ""
    
    def collect_stock_data(self, ticker: str, years: int = 5) -> str:
        """
        Collect stock data for a specific ticker for the last N years.
        
        Args:
            ticker: Stock ticker symbol
            years: Number of years of historical data to collect
            
        Returns:
            Path to the saved file
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Collecting {years} years of stock data for {ticker}")
        
        # Get stock data
        raw_data = self.get_stock_data(ticker, start_date_str, end_date_str)
        
        if raw_data.empty:
            return ""
            
        # Process data
        processed_data = self.process_stock_data(raw_data)
        
        # Save data
        return self.save_stock_data(processed_data, ticker)
    
    def collect_multiple_stocks(self, tickers: List[str], years: int = 5) -> Dict[str, str]:
        """
        Collect stock data for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            years: Number of years of historical data to collect
            
        Returns:
            Dictionary mapping tickers to saved file paths
        """
        results = {}
        
        for ticker in tqdm(tickers, desc="Collecting stock data"):
            filepath = self.collect_stock_data(ticker, years)
            results[ticker] = filepath
            
        return results
    
    def update_stock_data(self, ticker: str) -> bool:
        """
        Update existing stock data with the latest data.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if update was successful, False otherwise
        """
        # Clean ticker for filename
        clean_ticker = ticker.split('.')[0]
        
        # Check if file exists
        filename = f"stockprice_{clean_ticker}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"No existing data found for {ticker}, collecting new data")
            return bool(self.collect_stock_data(ticker))
            
        try:
            # Load existing data
            existing_data = pd.read_csv(filepath)
            
            if existing_data.empty:
                logger.warning(f"Existing data is empty for {ticker}, collecting new data")
                return bool(self.collect_stock_data(ticker))
                
            # Get the latest date in the existing data
            latest_date = existing_data['Date'].max()
            
            # Convert to datetime
            latest_date = datetime.strptime(str(latest_date), '%Y%m%d')
            
            # Add one day to get the start date for new data
            start_date = latest_date + timedelta(days=1)
            end_date = datetime.now()
            
            # Check if we need to update
            if start_date >= end_date:
                logger.info(f"Data for {ticker} is already up to date")
                return True
                
            # Format dates
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"Updating stock data for {ticker} from {start_date_str} to {end_date_str}")
            
            # Get new data
            new_data = self.get_stock_data(ticker, start_date_str, end_date_str)
            
            if new_data.empty:
                logger.warning(f"No new data found for {ticker}")
                return True  # Still consider it a success as we tried to update
                
            # Process new data
            processed_new_data = self.process_stock_data(new_data)
            
            # Combine existing and new data
            combined_data = pd.concat([existing_data, processed_new_data], ignore_index=True)
            
            # Remove duplicates if any
            combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
            
            # Sort by date
            combined_data = combined_data.sort_values('Date')
            
            # Save combined data
            combined_data.to_csv(filepath, index=False)
            
            logger.info(f"Updated stock data for {ticker}, added {len(processed_new_data)} new records")
            return True
            
        except Exception as e:
            logger.error(f"Error updating stock data for {ticker}: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Create collector instance
    collector = StockDataCollector()
    
    # Collect data for Samsung (005930)
    filepath = collector.collect_stock_data("005930")
    print(f"Stock data saved to: {filepath}")
    
    # Update existing data
    success = collector.update_stock_data("005930")
    print(f"Update successful: {success}")
