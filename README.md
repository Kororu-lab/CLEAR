# CLEAR: Clustering and Stock Impact-based News Recommendation System

[![GitHub](https://img.shields.io/github/license/Kororu-lab/CLEAR)](https://github.com/Kororu-lab/CLEAR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

CLEAR (Clustering and Stock Impact-based News Recommendation System) is a news recommendation system focused on stock impact analysis. It is inspired by NAVER's AiRS architecture but replaces personalization components with stock impact analysis.

## Overview

This system analyzes Korean news articles to identify their potential impact on stock prices. It clusters similar news articles, analyzes their sentiment and correlation with stock price movements, and provides recommendations based on content similarity and stock impact.

## Key Features

- **Korean Text Processing**: Specialized preprocessing for Korean news content with configurable stopwords lists
- **News Vectorization**: Flexible vectorization with options to handle title and content differently
- **News Clustering**: Groups similar news articles to identify trends and topics
- **Stock Impact Analysis**: Measures how news articles affect stock prices
- **Advanced Stock Impact Analysis**: Uses KO-finbert for sentiment analysis and incorporates volatility metrics
- **News Recommendation**: Combines multiple scoring mechanisms (CBF, QE, SI, Latest) to recommend relevant news

## System Architecture

The system consists of the following components:

1. **Text Preprocessor**: Cleans and normalizes Korean text data
2. **News Vectorizer**: Converts text to numerical vectors
3. **News Clustering**: Groups similar news articles
4. **Stock Impact Analyzer**: Analyzes news impact on stock prices
5. **Advanced Stock Impact Analyzer**: Provides more sophisticated impact analysis
6. **News Recommender**: Recommends news based on multiple criteria

## Data Format

### News Data

The system works with news data in the following format:
- Title: News title
- Date: Publication date and time (format: YYYYMMDD HH:MM)
- Press: News source
- Link: URL to the original article
- Body: Article content
- Emotion: Emotion information
- Num_comment: Number of comments
- AI Summary: AI-generated summary (optional)

### Stock Data

The system works with stock data in the following format:
- Date: Trading date (format: YYYYMMDD)
- Time: Time column (ignored as it contains zeros)
- Start: Opening price (09:00)
- High: Highest price of the day
- Low: Lowest price of the day
- End: Closing price (15:00)
- Volume: Trading volume

## Recent Improvements

The system has been significantly improved with the following enhancements:

1. **Robust Date Parsing**: Fixed date parsing for Korean news format (YYYYMMDD HH:MM)
2. **Proper Stock Data Handling**: Explicitly ignores the Time column in stock data
3. **Enhanced Ticker Extraction**: Improved ticker extraction logic with fallback mechanisms
4. **Comprehensive Impact Score Calculation**: Ensures all articles receive impact scores
5. **Visualization Enhancements**: Better visualization of impact distributions and trends

## Usage

### Demo Notebook

A comprehensive demonstration notebook is provided at `notebooks/demo/CLEAR_Demo_Improved.ipynb`. This notebook shows:

- Step-by-step usage of each component
- Visualizations of results
- Explanations in both Korean and English
- Formulas used in the implementation

### Basic Usage Example

```python
# Import components
from src.data.text_preprocessor import TextPreprocessor
from src.models.news_vectorizer import NewsVectorizer
from src.models.news_clustering import NewsClustering
from src.models.advanced_stock_impact_analyzer import AdvancedStockImpactAnalyzer
from src.models.news_recommender import NewsRecommender

# Initialize components
text_preprocessor = TextPreprocessor(language='ko', use_mecab=True)
news_vectorizer = NewsVectorizer(title_weight=0.7, body_weight=0.3)
news_clustering = NewsClustering(method='kmeans', n_clusters=10)
stock_impact_analyzer = AdvancedStockImpactAnalyzer(use_finbert=True)
news_recommender = NewsRecommender(use_cbf=True, use_si=True)

# Process news data
processed_titles = news_df['Title'].apply(text_preprocessor.preprocess)
processed_bodies = news_df['Body'].apply(text_preprocessor.preprocess)
news_vectors = news_vectorizer.fit_transform(processed_titles, processed_bodies)
cluster_labels = news_clustering.fit_predict(news_vectors)
news_df['cluster_id'] = cluster_labels

# Analyze stock impact
impact_df = stock_impact_analyzer.analyze_news_impact(news_df, stock_data)

# Get recommendations
recommended_news = news_recommender.recommend(
    news_df=impact_df,
    query="삼성전자 실적",
    top_n=10,
    impact_weight=0.7
)
```

## Configuration Options

### Text Preprocessor

- `language`: Language of the text ('ko' for Korean)
- `use_mecab`: Whether to use Mecab for Korean morphological analysis
- `remove_stopwords`: Whether to remove stopwords
- `custom_stopwords`: List of custom stopwords to remove

### News Vectorizer

- `method`: Vectorization method ('tfidf', 'word2vec', 'bert')
- `title_weight`: Weight for title in combined vectorization
- `body_weight`: Weight for body in combined vectorization
- `use_title_only`: Whether to use only title for vectorization

### News Clustering

- `method`: Clustering method ('kmeans', 'dbscan', 'hierarchical')
- `n_clusters`: Number of clusters for k-means
- `random_state`: Random seed for reproducibility

### Advanced Stock Impact Analyzer

- `use_finbert`: Whether to use KO-finbert for sentiment analysis
- `use_volatility`: Whether to incorporate volatility in impact calculation
- `use_market_trend`: Whether to consider market trends
- `sentiment_weight`: Weight of sentiment in impact calculation

### News Recommender

- `use_cf`: Whether to use collaborative filtering
- `use_cbf`: Whether to use content-based filtering
- `use_qe`: Whether to use query expansion
- `use_si`: Whether to use stock impact
- `use_latest`: Whether to prioritize latest news
- `weights`: Weights for each recommendation mechanism

## Requirements

The system requires the following dependencies:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn
- transformers (optional, for KO-finbert)
- torch (optional, for neural network models)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
