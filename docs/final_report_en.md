# CLEAR: NAVER AiRS-based News Recommendation and Stock Impact Analysis System

## Executive Summary

The CLEAR (Clustering and Learning Engine for Automated Recommendations) system is a comprehensive news recommendation and stock impact analysis platform based on NAVER's AiRS (AI Recommendation System) algorithm. This system focuses on non-personalized news clustering and recommendation with an emphasis on stock market impact analysis. 

CLEAR implements the core mechanisms of NAVER's AiRS while extending its capabilities with advanced Korean language processing models and stock impact analysis features. The system processes news articles and stock price data to identify relationships between news events and market movements, clusters similar news items, and provides recommendations based on multiple factors including content similarity, social impact, recency, and stock price impact.

This report documents the architecture, implementation details, and evaluation of the CLEAR system, highlighting both the core AiRS mechanisms and the advanced extensions developed specifically for Korean financial news analysis.

## 1. Introduction

### 1.1 Background

News recommendation systems play a crucial role in helping users navigate the overwhelming volume of information available online. NAVER, one of Korea's leading technology companies, developed the AiRS (AI Recommendation System) to provide personalized news recommendations to its users. The AiRS system employs various techniques including collaborative filtering, content-based filtering, and quality evaluation to deliver relevant news content.

The CLEAR system builds upon NAVER's AiRS framework, adapting it for financial news analysis with a focus on stock market impact. By analyzing the relationship between news articles and stock price movements, CLEAR provides insights into how news events affect market behavior and recommends relevant news articles based on their potential impact on specific stocks.

### 1.2 Objectives

The primary objectives of the CLEAR system are:

1. To implement the core mechanisms of NAVER's AiRS for news clustering and recommendation
2. To extend these mechanisms with stock impact analysis capabilities
3. To leverage advanced Korean language processing models for improved text analysis
4. To provide a non-personalized recommendation system focused on financial news
5. To identify relationships between news events and stock price movements

### 1.3 Scope

The CLEAR system encompasses the following components:

1. Data processing for news articles and stock price data
2. Text preprocessing with Korean language support
3. News vectorization using multiple embedding techniques
4. News clustering to group similar articles
5. Stock impact analysis to measure the relationship between news and stock prices
6. News recommendation based on multiple factors
7. Evaluation and visualization of system performance

The system is designed to work with Korean financial news articles and stock price data, with a focus on the Korean market. While the original AiRS system includes personalization features, these have been excluded from CLEAR as specified in the project requirements.

## 2. System Architecture

### 2.1 Overall Architecture

The CLEAR system follows a modular architecture with the following main components:

1. **Data Processing**: Handles the loading and preprocessing of news and stock data
2. **Text Preprocessing**: Cleans and normalizes Korean text for analysis
3. **News Vectorization**: Converts text into vector representations using various embedding methods
4. **News Clustering**: Groups similar news articles based on their vector representations
5. **Stock Impact Analysis**: Analyzes the relationship between news articles and stock price movements
6. **News Recommendation**: Recommends news articles based on multiple factors
7. **Evaluation**: Measures and visualizes system performance

The system is implemented in Python, leveraging various libraries for natural language processing, machine learning, and data analysis.

### 2.2 Data Flow

The data flow through the CLEAR system follows these steps:

1. News articles and stock price data are loaded from CSV files
2. News text is preprocessed to remove noise and normalize content
3. Preprocessed text is vectorized using various embedding methods
4. News vectors are clustered to identify groups of similar articles
5. Stock impact analysis is performed to measure the relationship between news and stock prices
6. News articles are scored based on multiple factors including content similarity, social impact, recency, and stock impact
7. Recommendations are generated based on these scores
8. Results are evaluated and visualized

### 2.3 Component Interactions

The components of the CLEAR system interact as follows:

- The **Data Processing** component provides clean data to the **Text Preprocessing** component
- The **Text Preprocessing** component feeds normalized text to the **News Vectorization** component
- The **News Vectorization** component generates vectors for the **News Clustering** and **News Recommendation** components
- The **News Clustering** component provides cluster information to the **News Recommendation** component
- The **Stock Impact Analysis** component calculates impact scores for the **News Recommendation** component
- The **News Recommendation** component combines all these inputs to generate final recommendations
- The **Evaluation** component measures the performance of all other components

## 3. Implementation Details

### 3.1 Data Processing

#### 3.1.1 News Data

The news data is stored in CSV format with the following columns:
- Title: The headline of the news article
- Date: The publication date and time (format: YYYYMMDD HH:MM)
- Press: The news source or publisher
- Link: URL to the original article
- Body: The main content of the article
- Emotion: Emotional tone of the article (if available)
- Num_comment: Number of comments on the article
- AI Summary: Automated summary of the article (if available)

The system processes this data by:
- Converting dates to a standardized datetime format
- Sorting articles by publication date
- Preparing text for preprocessing

#### 3.1.2 Stock Data

The stock data is stored in CSV format with the following columns:
- Date: Trading date
- Time: Trading time
- Start: Opening price
- High: Highest price during the period
- Low: Lowest price during the period
- End: Closing price
- Volume: Trading volume

The system processes this data by:
- Converting dates to a standardized datetime format
- Sorting by date and time
- Calculating price changes and percentage changes
- Computing volatility metrics

### 3.2 Text Preprocessing

The text preprocessing component handles Korean text with the following features:

- **Language Detection**: Automatically detects and processes Korean text
- **Stopword Removal**: Removes common Korean stopwords that don't contribute to meaning
- **Punctuation Removal**: Cleans text of punctuation marks
- **Tokenization**: Segments text into meaningful units using Korean-specific tokenizers
- **Mecab Integration**: Utilizes the Mecab Korean morphological analyzer for improved tokenization
- **Configurable Options**: Allows customization of preprocessing steps

Implementation details:
```python
class TextPreprocessor:
    def __init__(self, language='korean', remove_stopwords=True, 
                 remove_punctuation=True, remove_numbers=False, use_mecab=True):
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.use_mecab = use_mecab
        
        # Initialize Korean stopwords
        self.stopwords = self._load_stopwords()
        
        # Initialize Mecab tokenizer if enabled
        if self.use_mecab:
            self.mecab = self._initialize_mecab()
```

The preprocessor handles Korean-specific challenges such as:
- Proper tokenization of Korean sentences without spaces
- Handling of Korean particles and grammatical structures
- Preservation of meaningful compound words
- Special treatment for financial terminology

### 3.3 News Vectorization

The news vectorization component converts preprocessed text into vector representations using multiple methods:

#### 3.3.1 Basic Vectorization Methods

- **TF-IDF**: Term Frequency-Inverse Document Frequency vectorization
- **Word2Vec**: Word embedding using the Word2Vec algorithm
- **Doc2Vec**: Document embedding using the Doc2Vec algorithm

Implementation details:
```python
class NewsVectorizer:
    def __init__(self, method='tfidf', title_weight=0.7, content_weight=0.3, 
                 max_features=5000, vector_size=100):
        self.method = method
        self.title_weight = title_weight
        self.content_weight = content_weight
        self.max_features = max_features
        self.vector_size = vector_size
        self.vectorizer = None
```

The vectorizer supports configurable weighting between title and content, allowing for emphasis on headlines which often contain the most important information in news articles.

#### 3.3.2 Advanced Embedding Methods

The system implements advanced Korean language embedding models through the `KoreanEmbeddingEnhancer` class, which supports:

- **KoBERT**: Korean BERT model pre-trained on Korean text corpus
- **KLUE-RoBERTa**: Korean RoBERTa model pre-trained for the KLUE benchmark
- **KoSimCSE-roberta**: Korean sentence embedding model based on SimCSE approach
- **bge-m3-korean**: Korean fine-tuned version of BGE-M3 for semantic textual similarity
- **KPF-BERT**: BERT model pre-trained on Korean news articles from Korea Press Foundation
- **KoELECTRA**: Korean ELECTRA model pre-trained on Korean text corpus

Implementation details:
```python
class KoreanEmbeddingEnhancer:
    def __init__(self, models_dir=None, use_kobert=True, use_klue_roberta=True,
                 use_kosimcse=True, use_bge_korean=True, use_kpf_bert=False,
                 use_koelectra=False, cache_embeddings=True, device='cpu'):
        # Initialize models and configurations
```

The enhancer provides methods for:
- Getting embeddings for individual texts or batches
- Vectorizing dataframes with configurable text and content columns
- Creating ensemble embeddings using multiple models with configurable weights
- Efficient caching to improve performance

### 3.4 News Clustering

The news clustering component groups similar news articles based on their vector representations:

#### 3.4.1 Basic Clustering Methods

- **K-Means**: Clusters articles based on centroid proximity
- **DBSCAN**: Density-based clustering for identifying articles in dense regions
- **Agglomerative Clustering**: Hierarchical clustering for building a tree of article similarities

Implementation details:
```python
class NewsClustering:
    def __init__(self, method='kmeans', n_clusters=5, random_state=42, 
                 min_samples=5, eps=0.5, time_decay_factor=0.1):
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.min_samples = min_samples
        self.eps = eps
        self.time_decay_factor = time_decay_factor
        self.model = None
```

#### 3.4.2 Advanced Clustering Features

- **Time-Aware Clustering**: Incorporates publication time into similarity calculations
- **Stock Impact-Aware Clustering**: Considers stock impact when grouping articles
- **Cluster Evaluation**: Measures cluster quality using silhouette score and other metrics
- **Cluster Visualization**: Provides 2D visualizations of clusters using dimensionality reduction

The clustering component is designed to identify meaningful groups of news articles that discuss similar topics or events, which is essential for understanding news trends and their potential impact on stock prices.

### 3.5 Stock Impact Analysis

The stock impact analysis component measures the relationship between news articles and stock price movements:

#### 3.5.1 Basic Impact Analysis

- **Price Change Calculation**: Measures stock price changes before and after news publication
- **Volatility Analysis**: Analyzes stock price volatility around news events
- **Trading Volume Analysis**: Examines changes in trading volume related to news
- **Time Window Analysis**: Considers different time windows for measuring impact

Implementation details:
```python
class StockImpactAnalyzer:
    def __init__(self, lookback_window=3, lookahead_window=3, impact_threshold=0.5,
                 use_volume=True, use_volatility=True, ticker_mapping=None):
        self.lookback_window = lookback_window
        self.lookahead_window = lookahead_window
        self.impact_threshold = impact_threshold
        self.use_volume = use_volume
        self.use_volatility = use_volatility
        self.ticker_mapping = ticker_mapping or self._default_ticker_mapping()
```

#### 3.5.2 Advanced Impact Analysis

The system implements advanced stock impact analysis through the `AdvancedScoringMethods` class, which supports:

- **Embedding-Based Impact Scoring**: Uses text embeddings to measure semantic relationship with financial concepts
- **Sentiment-Enhanced Impact Analysis**: Incorporates sentiment analysis into impact calculations
- **Multi-Model Ensemble Scoring**: Combines scores from multiple models for improved accuracy
- **Visualization Tools**: Provides visualizations of impact scores and model comparisons

Implementation details:
```python
class AdvancedScoringMethods:
    def __init__(self, models_dir=None, use_kosimcse=True, use_bge_korean=True,
                 use_gte_korean=False, use_e5_korean=False, cache_embeddings=True,
                 device='cpu'):
        # Initialize models and configurations
```

The advanced impact analysis provides more nuanced understanding of how news articles affect stock prices, considering both the content of the articles and market reactions.

### 3.6 News Recommendation

The news recommendation component generates recommendations based on multiple factors:

#### 3.6.1 Core AiRS Mechanisms

Following NAVER's AiRS algorithm, the system implements:

- **CF-based Generation**: Collaborative filtering using normalized point-wise mutual information (NPMI)
- **CBF-based Generation**: Content-based filtering using article similarities
- **QE-based Generation**: Quality evaluation based on article metrics
- **SI-based Generation**: Social impact based on user interactions
- **Latest-based Generation**: Recency factor for prioritizing recent articles

Implementation details:
```python
class NewsRecommender:
    def __init__(self, cf_weight=0.3, cbf_weight=0.3, si_weight=0.2,
                 latest_weight=0.1, stock_impact_weight=0.1):
        self.cf_weight = cf_weight
        self.cbf_weight = cbf_weight
        self.si_weight = si_weight
        self.latest_weight = latest_weight
        self.stock_impact_weight = stock_impact_weight
```

#### 3.6.2 Stock Impact Extensions

The system extends the AiRS algorithm with stock impact considerations:

- **Stock Impact Weighting**: Incorporates stock impact scores into recommendation calculations
- **Stock-Specific Recommendations**: Provides recommendations relevant to specific stocks
- **Impact-Based Filtering**: Filters recommendations based on impact thresholds
- **Configurable Weights**: Allows adjustment of factor weights for different use cases

The recommendation component combines all the system's components to deliver relevant news articles based on content similarity, social impact, recency, and stock impact.

### 3.7 Korean Financial Text Analysis

The system includes specialized components for Korean financial text analysis:

#### 3.7.1 KO-FinBERT Integration

- **Sentiment Analysis**: Uses KO-FinBERT for financial sentiment analysis of Korean text
- **Polarity Detection**: Identifies positive, negative, or neutral sentiment in financial news
- **Confidence Scoring**: Provides confidence levels for sentiment predictions

Implementation details:
```python
class KoreanFinancialTextAnalyzer:
    def __init__(self, use_finbert=True, use_advanced_embeddings=True,
                 cache_results=True, models_dir=None):
        self.use_finbert = use_finbert
        self.use_advanced_embeddings = use_advanced_embeddings
        self.cache_results = cache_results
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "text_analysis")
```

#### 3.7.2 Advanced Text Analysis Features

- **Batch Processing**: Efficiently processes multiple texts
- **Visualization Tools**: Provides visualizations of sentiment analysis results
- **Caching Mechanism**: Improves performance through result caching
- **Ensemble Methods**: Combines multiple models for improved accuracy

The Korean financial text analysis component provides deeper insights into the sentiment and implications of financial news articles, which is crucial for understanding their potential impact on stock prices.

## 4. Evaluation

### 4.1 Clustering Evaluation

The clustering component is evaluated using:

- **Silhouette Score**: Measures how similar articles are to their own cluster compared to other clusters
- **Davies-Bouldin Index**: Evaluates cluster separation
- **Inertia**: Measures the compactness of clusters
- **Manual Inspection**: Qualitative assessment of cluster coherence

Results show that the advanced embedding methods (particularly ensemble approaches) produce more coherent clusters than basic TF-IDF vectorization.

### 4.2 Stock Impact Analysis Evaluation

The stock impact analysis component is evaluated using:

- **Correlation Analysis**: Measures correlation between predicted impact and actual price changes
- **Precision and Recall**: Evaluates accuracy in identifying high-impact news
- **Model Comparison**: Compares performance of different embedding models
- **Case Studies**: Detailed analysis of specific news events and their market impact

Results indicate that ensemble approaches combining multiple embedding models provide the most accurate impact predictions.

### 4.3 Recommendation Evaluation

The recommendation component is evaluated using:

- **Relevance Assessment**: Measures the relevance of recommendations to query articles
- **Diversity Analysis**: Evaluates the diversity of recommendations
- **Weight Sensitivity Analysis**: Examines how different factor weights affect recommendations
- **User Simulation**: Simulates user interactions with the recommendation system

Results show that incorporating stock impact into the recommendation process improves the relevance of recommendations for financial news analysis.

## 5. Advanced Models and Techniques

### 5.1 Korean Language Models

The system leverages several advanced Korean language models:

- **KoBERT**: Provides general Korean language understanding
- **KLUE-RoBERTa**: Offers improved performance on Korean language understanding tasks
- **KoSimCSE-roberta**: Specializes in Korean sentence embeddings for similarity tasks
- **bge-m3-korean**: Excels at semantic textual similarity for Korean
- **KPF-BERT**: Focuses on Korean news articles
- **KoELECTRA**: Provides efficient Korean language representation

These models are integrated through the `KoreanEmbeddingEnhancer` class, which provides a unified interface for using multiple models and creating ensemble embeddings.

### 5.2 Ensemble Methods

The system implements several ensemble methods:

- **Embedding Ensembles**: Combines embeddings from multiple models with configurable weights
- **Scoring Ensembles**: Merges impact scores from different models
- **Clustering Ensembles**: Integrates results from multiple clustering algorithms
- **Recommendation Ensembles**: Combines recommendations from different factors

These ensemble approaches improve the robustness and accuracy of the system by leveraging the strengths of multiple models and techniques.

### 5.3 Visualization Techniques

The system includes various visualization techniques:

- **Cluster Visualization**: Shows article clusters in 2D space using PCA or t-SNE
- **Impact Score Distribution**: Displays the distribution of impact scores
- **Model Comparison**: Visualizes performance differences between models
- **Correlation Heatmaps**: Shows relationships between different factors
- **Time Series Analysis**: Displays stock price movements in relation to news events

These visualizations help users understand the system's behavior and the relationships between news articles and stock prices.

## 6. Deployment and Usage

### 6.1 System Requirements

The CLEAR system requires:

- Python 3.8 or higher
- PyTorch for deep learning models
- Transformers library for Hugging Face models
- Pandas, NumPy, and Scikit-learn for data processing and analysis
- Matplotlib and Seaborn for visualization
- Mecab-ko for Korean text processing

### 6.2 Installation

The system can be installed by:

1. Cloning the repository
2. Installing dependencies using `pip install -r requirements.txt`
3. Downloading pre-trained models (if not using on-demand loading)

### 6.3 Configuration

The system is highly configurable through:

- Configuration files for system-wide settings
- Class parameters for component-specific settings
- Runtime parameters for execution-specific settings

### 6.4 Usage Examples

Basic usage:
```python
# Initialize components
preprocessor = TextPreprocessor(language='korean')
vectorizer = NewsVectorizer(method='tfidf')
clustering = NewsClustering(method='kmeans')
impact_analyzer = StockImpactAnalyzer()
recommender = NewsRecommender()

# Process data
processed_text = preprocessor.preprocess_text(news_df['Title'])
vectors = vectorizer.vectorize_dataframe(news_df)
clusters = clustering.cluster(vectors)
impact_scores = impact_analyzer.analyze_impact(news_df, stock_df)
recommendations = recommender.recommend(news_df, vectors=vectors)
```

Advanced usage with Korean embedding models:
```python
# Initialize advanced components
embedding_enhancer = KoreanEmbeddingEnhancer(
    use_kobert=True,
    use_kosimcse=True
)
scoring_methods = AdvancedScoringMethods(
    use_kosimcse=True,
    use_bge_korean=True
)

# Process data with advanced methods
embeddings = embedding_enhancer.vectorize_dataframe_ensemble(
    news_df,
    models=['kobert', 'kosimcse'],
    weights=[0.6, 0.4]
)
impact_results = scoring_methods.calculate_ensemble_impact_scores(
    news_df,
    models=['kosimcse', 'bge_korean']
)
```

## 7. Limitations and Future Work

### 7.1 Current Limitations

The current implementation has several limitations:

- **Data Dependency**: Relies on pre-collected news and stock data
- **Language Specificity**: Primarily designed for Korean financial news
- **Computational Requirements**: Advanced models require significant computational resources
- **Evaluation Challenges**: Lack of ground truth for impact assessment
- **Market Complexity**: Financial markets are influenced by many factors beyond news

### 7.2 Future Work

Potential areas for future development include:

- **Real-time Processing**: Implementing real-time news collection and analysis
- **Multi-language Support**: Extending to other languages beyond Korean
- **User Feedback Integration**: Incorporating user feedback for improved recommendations
- **Advanced Market Models**: Developing more sophisticated models of market behavior
- **Explainable AI**: Enhancing the explainability of impact predictions
- **Fine-tuning Models**: Further fine-tuning language models for financial domain

## 8. Conclusion

The CLEAR system successfully implements the core mechanisms of NAVER's AiRS algorithm while extending it with stock impact analysis capabilities and advanced Korean language processing. By leveraging multiple embedding models and ensemble techniques, the system provides robust news clustering, impact analysis, and recommendation for Korean financial news.

The system demonstrates the value of combining traditional recommendation approaches with domain-specific extensions for financial analysis. The integration of advanced Korean language models significantly improves the system's ability to understand and process Korean financial news, leading to more accurate impact predictions and recommendations.

The modular architecture of CLEAR allows for easy extension and customization, making it adaptable to different use cases and requirements. The comprehensive evaluation shows promising results, particularly for the advanced embedding and ensemble approaches.

Overall, the CLEAR system provides a solid foundation for news recommendation and stock impact analysis in the Korean financial domain, with potential for further development and application in real-world scenarios.

## 9. References

1. NAVER AiRS: AI Recommendation System - https://media.naver.com/algorithm
2. KoBERT - https://github.com/SKTBrain/KoBERT
3. KLUE-RoBERTa - https://huggingface.co/klue/roberta-base
4. KoSimCSE-roberta - https://huggingface.co/BM-K/KoSimCSE-roberta
5. bge-m3-korean - https://huggingface.co/upskyy/bge-m3-korean
6. KPF-BERT - https://huggingface.co/kpfbert/kpfbert
7. KoELECTRA - https://huggingface.co/monologg/koelectra-base-v3-discriminator
8. Mecab-ko - https://bitbucket.org/eunjeon/mecab-ko/
9. PyTorch - https://pytorch.org/
10. Hugging Face Transformers - https://huggingface.co/transformers/
