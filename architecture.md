# CLEAR System Architecture Design

## Overview

The CLEAR (Clustering and Stock Impact News Recommendation) system is designed to adapt NAVER's AiRS (AI Recommender System) architecture for financial news analysis with a focus on stock impact rather than personalization. The system will maintain all core mechanisms from AiRS while shifting the focus to analyzing how news articles affect stock prices.

## System Components

### 1. Data Collection Layer

#### 1.1 News Crawler
- Collects financial news articles from YNA (Yonhap News Agency)
- Stores data in the format: Title, Date, Press, Link, Body, Emotion, Num_comment, AI Summary
- Date format: YYYYMMDD HH:MM (e.g., 20250101 18:56)
- Configurable to use AI Summary when available

#### 1.2 Stock Data Collector
- Collects stock price data in the format: Date, Time, Start, High, Low, End, Volume
- Date format: YYYYMMDD (e.g., 20250305)
- Calculates delta values for price and volume changes
- Focuses on minimal required data as specified in the AiRs paper

### 2. Data Processing Layer

#### 2.1 Text Preprocessor
- Optimized for Korean text processing
- Configurable stopword handling with Mecab integration
- Handles mixed Korean/English content
- Language detection and appropriate processing

#### 2.2 News Vectorizer
- TF-IDF vectorization with configurable parameters
- Separate configuration for title and content processing
- Weighting mechanism for title vs. content importance
- Multiple embedding options including:
  - TF-IDF (default)
  - Word2Vec
  - FastText
  - OpenAI API embeddings
  - KR-FinBERT for financial sentiment

### 3. Core Algorithm Layer

#### 3.1 News Clustering
- Hierarchical agglomerative clustering
- Cosine similarity for measuring article relationships
- Configurable distance threshold and cluster size parameters
- Topic extraction for cluster labeling

#### 3.2 Stock Impact Analyzer
- Analyzes news impact on stock prices across multiple time windows
- Calculates impact scores on a -5 to +5 scale
- Considers both price and volume changes
- Machine learning model for impact prediction
- GPU acceleration with PyTorch

#### 3.3 Recommendation Engine
- Implements all 5 AiRS recommendation models:
  - CF (Collaborative Filtering): Adapted for stock-related behavior patterns
  - CBF (Content-Based Filtering): Based on article content similarity
  - QE (Quality Estimation): Measures article quality and credibility
  - SI (Social Impact): Replaced with Stock Impact for financial relevance
  - Latest: Prioritizes recent content
- Weighted ensemble approach for final recommendations

### 4. Application Layer

#### 4.1 Scheduler
- Runs the system twice daily at market open/close times
- Processes approximately 300 news articles per day
- Configurable for one-time inference testing

#### 4.2 API
- RESTful API for accessing recommendations
- Endpoints for trending topics, clusters, and individual articles
- Query parameters for filtering and customization

#### 4.3 Visualization
- Cluster visualization
- Impact score trends
- Stock price correlation with news events

## Data Flow

1. News articles and stock data are collected from sources
2. Text is preprocessed and converted to vector representations
3. Articles are clustered into related groups
4. Stock impact is analyzed for each article and cluster
5. Recommendations are generated based on impact and other factors
6. Results are stored and made available through the API

## Configuration System

A comprehensive configuration system allows for:
- Toggling features on/off
- Adjusting weights for different recommendation factors
- Setting thresholds for clustering and impact analysis
- Selecting embedding methods
- Configuring GPU usage for deep learning components

## Evaluation Framework

The system includes an evaluation framework to assess:
- Clustering quality (silhouette score, topic coherence)
- Recommendation relevance
- Impact prediction accuracy
- System performance metrics

## Implementation Considerations

- All code will be implemented in Python
- PyTorch will be used for deep learning components to leverage GPU acceleration
- Mecab will be used for Korean text processing
- The system will be designed to be modular and extensible
- All AiRS mechanisms will be maintained except personalization aspects
- Focus will be on stock impact analysis rather than user preferences
