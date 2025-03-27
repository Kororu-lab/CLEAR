# CLEAR: Clustering and Stock Impact-based News Recommendation System

[![GitHub](https://img.shields.io/github/license/Kororu-lab/CLEAR)](https://github.com/Kororu-lab/CLEAR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

## Overview

CLEAR (Clustering and Stock Impact-based News Recommendation System) is an advanced news recommendation system based on NAVER's AiRs architecture, specifically adapted for financial news analysis and stock impact assessment. Unlike the original AiRs system which focuses on personalization, CLEAR prioritizes the impact of news on stock prices, providing recommendations that highlight financially significant news articles.

The system implements all core mechanisms from NAVER's AiRs (Collaborative Filtering, Content-Based Filtering, Quality Estimation, Social Impact, and Latest news prioritization) while replacing personalization components with stock impact analysis. CLEAR processes Korean financial news articles, clusters related content, analyzes the impact on stock prices, and generates recommendations based on financial significance rather than personal preferences.

## Features

- **Korean Text Processing**: Advanced Korean language support with Mecab integration
- **News Clustering**: Hierarchical agglomerative clustering to group related financial articles
- **Stock Impact Analysis**: Multi-window impact analysis (immediate, short-term, medium-term)
- **Multi-factor Recommendation**: Combines impact, quality, content similarity, collaborative filtering, and recency
- **GPU Acceleration**: Optimized for CUDA-compatible GPUs
- **Scheduled Updates**: Automatic updates at market open/close times

## Repository Structure

```
CLEAR/
├── config/                 # Configuration files
│   ├── config.py           # Configuration utilities
│   └── config.yaml         # Default configuration
├── data/                   # Data directory
│   ├── news/               # News data
│   └── stock/              # Stock price data
├── docs/                   # Documentation
│   ├── technical_report.md        # Technical report (English)
│   └── technical_report_korean.md # Technical report (Korean)
├── models/                 # Trained models
├── results/                # Results and visualizations
├── src/                    # Source code
│   ├── data/               # Data collection and processing
│   │   ├── news_crawler.py         # News crawler
│   │   ├── stock_data_collector.py # Stock data collector
│   │   └── text_preprocessor.py    # Text preprocessor
│   ├── models/             # Core algorithms
│   │   ├── news_vectorizer.py      # News vectorization
│   │   ├── news_clustering.py      # News clustering
│   │   ├── news_recommender.py     # News recommendation
│   │   └── stock_impact_analyzer.py # Stock impact analysis
│   ├── evaluation.py       # Evaluation framework
│   └── pipeline.py         # Pipeline integration
├── CLEAR_demo.ipynb        # Demo notebook
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Kororu-lab/CLEAR.git
   cd CLEAR
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Korean language support:
   ```
   pip install konlpy mecab-python3
   ```

## Usage

### Running the Pipeline

To run the pipeline once with existing data:
```
python src/pipeline.py --use-existing
```

To run with new data collection:
```
python src/pipeline.py
```

To run in scheduled mode (at market open/close):
```
python src/pipeline.py --schedule
```

### Interactive Demo

For an interactive demonstration of the system, open the Jupyter notebook:
```
jupyter notebook CLEAR_demo.ipynb
```

## Configuration

The system is configured through `config/config.yaml`. Key configuration options include:

- `stock_tickers`: List of stock tickers to monitor
- `news_crawler.use_ai_summary`: Whether to use AI-generated summaries
- `text_preprocessor.use_mecab`: Whether to use Mecab for Korean tokenization
- `news_vectorizer.method`: Vectorization method (tfidf, word2vec, fasttext, openai)
- `news_clustering.distance_threshold`: Threshold for forming clusters
- `stock_impact.time_windows`: Time windows for impact analysis
- `news_recommender.weights`: Weights for different recommendation factors
- `schedule`: Market open/close times for scheduled execution

## Documentation

For detailed documentation, see:
- [Technical Report (English)](docs/technical_report.md)
- [Technical Report (Korean)](docs/technical_report_korean.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on NAVER's AiRs (AI Recommender system) architecture
- Inspired by "AiRS: A Large-scale Recommender System for News Service" by Kim, J., et al. (2019)
