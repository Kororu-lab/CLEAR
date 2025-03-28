"""
Advanced Korean text analysis models for financial news.
Implements various models for Korean text processing, including KO-finbert and advanced embeddings.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KoreanFinancialTextAnalyzer:
    """
    Advanced analyzer for Korean financial text using various models.
    """
    
    def __init__(self, 
                 models_dir: str = None,
                 use_finbert: bool = True,
                 use_kobert: bool = False,
                 use_openai: bool = False,
                 openai_api_key: str = None,
                 cache_embeddings: bool = True,
                 device: str = 'cpu'):
        """
        Initialize the Korean financial text analyzer.
        
        Args:
            models_dir: Directory to save/load models and cache
            use_finbert: Whether to use KO-finbert for sentiment analysis
            use_kobert: Whether to use KoBERT for embeddings
            use_openai: Whether to use OpenAI API for embeddings
            openai_api_key: OpenAI API key (if using OpenAI)
            cache_embeddings: Whether to cache embeddings
            device: Device to use for model inference ('cpu' or 'cuda')
        """
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "text_analysis")
        self.use_finbert = use_finbert
        self.use_kobert = use_kobert
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        self.cache_embeddings = cache_embeddings
        self.device = device
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.kobert_model = None
        self.kobert_tokenizer = None
        
        # Initialize cache
        self.embedding_cache = {}
        self.sentiment_cache = {}
        
        # Load models if enabled
        if self.use_finbert:
            self._initialize_finbert()
        
        if self.use_kobert:
            self._initialize_kobert()
        
        if self.use_openai and not self.openai_api_key:
            logger.warning("OpenAI API key not provided, disabling OpenAI embeddings")
            self.use_openai = False
        
        # Load cache if available
        if self.cache_embeddings:
            self._load_cache()
        
        logger.info(f"Initialized KoreanFinancialTextAnalyzer with models: " +
                   f"finbert={self.use_finbert}, kobert={self.use_kobert}, openai={self.use_openai}")
    
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
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.finbert_model = self.finbert_model.to('cuda')
                logger.info("Using CUDA for KO-finbert")
            else:
                self.finbert_model = self.finbert_model.to('cpu')
                logger.info("Using CPU for KO-finbert")
            
            # Set to evaluation mode
            self.finbert_model.eval()
            
            logger.info("KO-finbert model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KO-finbert: {str(e)}")
            logger.warning("Falling back to rule-based sentiment analysis")
            self.use_finbert = False
    
    def _initialize_kobert(self):
        """
        Initialize the KoBERT model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using kobert
            from transformers import BertModel, BertTokenizer
            import torch
            
            # Load KoBERT model and tokenizer
            model_name = "monologg/kobert"
            
            logger.info(f"Loading KoBERT model: {model_name}")
            
            self.kobert_tokenizer = BertTokenizer.from_pretrained(model_name)
            self.kobert_model = BertModel.from_pretrained(model_name)
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.kobert_model = self.kobert_model.to('cuda')
                logger.info("Using CUDA for KoBERT")
            else:
                self.kobert_model = self.kobert_model.to('cpu')
                logger.info("Using CPU for KoBERT")
            
            # Set to evaluation mode
            self.kobert_model.eval()
            
            logger.info("KoBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KoBERT: {str(e)}")
            logger.warning("Disabling KoBERT embeddings")
            self.use_kobert = False
    
    def _load_cache(self):
        """
        Load embedding and sentiment cache from disk.
        """
        try:
            # Load embedding cache
            embedding_cache_path = os.path.join(self.models_dir, "embedding_cache.json")
            if os.path.exists(embedding_cache_path):
                with open(embedding_cache_path, 'r', encoding='utf-8') as f:
                    # JSON can't handle numpy arrays directly, so we stored them as lists
                    cache_data = json.load(f)
                    self.embedding_cache = {k: np.array(v) for k, v in cache_data.items()}
                logger.info(f"Loaded embedding cache with {len(self.embedding_cache)} entries")
            
            # Load sentiment cache
            sentiment_cache_path = os.path.join(self.models_dir, "sentiment_cache.json")
            if os.path.exists(sentiment_cache_path):
                with open(sentiment_cache_path, 'r', encoding='utf-8') as f:
                    self.sentiment_cache = json.load(f)
                logger.info(f"Loaded sentiment cache with {len(self.sentiment_cache)} entries")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.embedding_cache = {}
            self.sentiment_cache = {}
    
    def _save_cache(self):
        """
        Save embedding and sentiment cache to disk.
        """
        if not self.cache_embeddings:
            return
            
        try:
            # Save embedding cache
            if self.embedding_cache:
                embedding_cache_path = os.path.join(self.models_dir, "embedding_cache.json")
                
                # Convert numpy arrays to lists for JSON serialization
                cache_data = {k: v.tolist() for k, v in self.embedding_cache.items()}
                
                with open(embedding_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f)
                logger.info(f"Saved embedding cache with {len(self.embedding_cache)} entries")
            
            # Save sentiment cache
            if self.sentiment_cache:
                sentiment_cache_path = os.path.join(self.models_dir, "sentiment_cache.json")
                with open(sentiment_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.sentiment_cache, f)
                logger.info(f"Saved sentiment cache with {len(self.sentiment_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of Korean financial text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
        
        # Check cache
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        
        try:
            # Use KO-finbert if available
            if self.use_finbert and self.finbert_model is not None:
                import torch
                
                # Tokenize text
                inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                # Move to same device as model
                if self.device == 'cuda' and torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                # Get sentiment prediction
                with torch.no_grad():
                    outputs = self.finbert_model(**inputs)
                    predictions = outputs.logits.softmax(dim=1)
                
                # Get sentiment score and label
                # KR-FinBert-SC has 3 classes: negative (0), neutral (1), positive (2)
                probs = predictions[0].cpu().numpy()
                
                # Convert to -1 to 1 scale
                sentiment_score = probs[2] - probs[0]  # positive - negative
                
                # Get label
                label_idx = predictions[0].argmax().item()
                labels = ['negative', 'neutral', 'positive']
                sentiment_label = labels[label_idx]
                
                # Get confidence
                confidence = probs[label_idx]
                
                result = {
                    'score': float(sentiment_score),
                    'label': sentiment_label,
                    'confidence': float(confidence),
                    'probabilities': {
                        'negative': float(probs[0]),
                        'neutral': float(probs[1]),
                        'positive': float(probs[2])
                    }
                }
            else:
                # Fallback to rule-based sentiment analysis
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
                
                # Calculate confidence
                confidence = abs(sentiment_score) if total_count > 0 else 0.0
                
                result = {
                    'score': sentiment_score,
                    'label': sentiment_label,
                    'confidence': confidence,
                    'keyword_counts': {
                        'positive': positive_count,
                        'negative': negative_count
                    }
                }
            
            # Cache result
            if self.cache_embeddings:
                self.sentiment_cache[text] = result
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
    
    def get_embedding(self, text: str, model: str = 'default') -> np.ndarray:
        """
        Get embedding vector for Korean text.
        
        Args:
            text: Text to embed
            model: Model to use ('kobert', 'openai', or 'default')
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not isinstance(text, str):
            # Return zero vector of appropriate size
            if model == 'openai':
                return np.zeros(1536)  # OpenAI embeddings are 1536-dimensional
            else:
                return np.zeros(768)  # BERT embeddings are 768-dimensional
        
        # Create cache key
        cache_key = f"{model}:{text}"
        
        # Check cache
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Determine which model to use
            if model == 'openai' and self.use_openai:
                embedding = self._get_openai_embedding(text)
            elif model == 'kobert' and self.use_kobert:
                embedding = self._get_kobert_embedding(text)
            elif self.use_kobert:
                embedding = self._get_kobert_embedding(text)
            elif self.use_openai:
                embedding = self._get_openai_embedding(text)
            else:
                # Fallback to simple TF-IDF like embedding
                embedding = self._get_simple_embedding(text)
            
            # Cache embedding
            if self.cache_embeddings:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            
            # Return zero vector of appropriate size
            if model == 'openai':
                return np.zeros(1536)
            else:
                return np.zeros(768)
    
    def _get_kobert_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KoBERT model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        import torch
        
        # Tokenize text
        inputs = self.kobert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.kobert_model(**inputs)
            
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using OpenAI API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            import openai
            
            # Set API key
            openai.api_key = self.openai_api_key
            
            # Get embedding
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            
            # Extract embedding
            embedding = np.array(response['data'][0]['embedding'])
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {str(e)}")
            logger.warning("Falling back to simple embedding")
            return self._get_simple_embedding(text)
    
    def _get_simple_embedding(self, text: str) -> np.ndarray:
        """
        Get simple TF-IDF like embedding for Korean text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Try to use scikit-learn if available
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create vectorizer
            vectorizer = TfidfVectorizer(max_features=768)
            
            # Fit and transform
            X = vectorizer.fit_transform([text])
            
            # Convert to dense array
            embedding = X.toarray()[0]
            
            # Pad or truncate to 768 dimensions
            if len(embedding) < 768:
                embedding = np.pad(embedding, (0, 768 - len(embedding)))
            elif len(embedding) > 768:
                embedding = embedding[:768]
            
            return embedding
        except Exception as e:
            logger.error(f"Error creating simple embedding: {str(e)}")
            
            # Even simpler fallback - character frequency
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Create embedding from character frequencies
            embedding = np.zeros(768)
            for i, (char, count) in enumerate(char_counts.items()):
                if i >= 768:
                    break
                embedding[i] = count / len(text)
            
            return embedding
    
    def analyze_text_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of Korean financial texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with analysis results
        """
        results = []
        
        for text in texts:
            # Get sentiment
            sentiment = self.analyze_sentiment(text)
            
            # Get embedding
            embedding = self.get_embedding(text)
            
            # Combine results
            result = {
                'sentiment': sentiment,
                'embedding': embedding.tolist(),  # Convert to list for JSON serialization
                'text_length': len(text)
            }
            
            results.append(result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                        text_col: str = 'Title',
                        content_col: str = 'Body',
                        use_content: bool = False) -> pd.DataFrame:
        """
        Analyze texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts
            text_col: Column name for primary text (e.g., title)
            content_col: Column name for secondary text (e.g., body)
            use_content: Whether to use content for analysis
            
        Returns:
            DataFrame with analysis results
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for analysis")
            return df
            
        logger.info(f"Analyzing {len(df)} texts")
        
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Initialize columns
            result_df['sentiment_score'] = None
            result_df['sentiment_label'] = None
            result_df['sentiment_confidence'] = None
            
            # Process each row
            for idx, row in result_df.iterrows():
                # Determine text to analyze
                if text_col in row and isinstance(row[text_col], str):
                    text = row[text_col]
                    
                    # Add content if specified
                    if use_content and content_col in row and isinstance(row[content_col], str):
                        # Use first 512 characters of content
                        text += " " + row[content_col][:512]
                else:
                    # Skip if no text
                    continue
                
                # Analyze sentiment
                sentiment = self.analyze_sentiment(text)
                
                # Store results
                result_df.at[idx, 'sentiment_score'] = sentiment['score']
                result_df.at[idx, 'sentiment_label'] = sentiment['label']
                result_df.at[idx, 'sentiment_confidence'] = sentiment['confidence']
            
            # Save cache
            self._save_cache()
            
            logger.info(f"Completed analysis for {len(result_df)} texts")
            
            return result_df
        except Exception as e:
            logger.error(f"Error analyzing DataFrame: {str(e)}")
            return df
    
    def visualize_sentiment_distribution(self, df: pd.DataFrame, 
                                       save_path: Optional[str] = None) -> None:
        """
        Visualize sentiment distribution.
        
        Args:
            df: DataFrame with sentiment analysis results
            save_path: Path to save the visualization
        """
        if 'sentiment_label' not in df.columns or df['sentiment_label'].isna().all():
            logger.warning("No sentiment labels available for visualization")
            return
            
        try:
            # Count sentiment labels
            sentiment_counts = df['sentiment_label'].value_counts()
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create bar plot
            ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
            
            # Add value labels
            for i, v in enumerate(sentiment_counts.values):
                ax.text(i, v + 0.1, str(v), ha='center')
            
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved sentiment distribution visualization to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing sentiment distribution: {str(e)}")
    
    def visualize_sentiment_over_time(self, df: pd.DataFrame, 
                                    date_col: str = 'Date',
                                    save_path: Optional[str] = None) -> None:
        """
        Visualize sentiment trends over time.
        
        Args:
            df: DataFrame with sentiment analysis results
            date_col: Column name for dates
            save_path: Path to save the visualization
        """
        if ('sentiment_score' not in df.columns or df['sentiment_score'].isna().all() or
            date_col not in df.columns):
            logger.warning("No sentiment scores or dates available for visualization")
            return
            
        try:
            # Ensure date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df[date_col] = pd.to_datetime(df[date_col])
            
            # Group by date and calculate average sentiment
            daily_sentiment = df.groupby(df[date_col].dt.date)['sentiment_score'].mean().reset_index()
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Create line plot
            plt.plot(daily_sentiment[date_col], daily_sentiment['sentiment_score'], marker='o')
            
            # Add horizontal line at zero
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Add trend line
            try:
                from scipy import stats
                
                # Calculate trend line
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    range(len(daily_sentiment)), daily_sentiment['sentiment_score']
                )
                
                # Plot trend line
                trend_x = np.array([0, len(daily_sentiment) - 1])
                trend_y = slope * trend_x + intercept
                
                plt.plot(daily_sentiment[date_col].iloc[trend_x], trend_y, 'r--', alpha=0.7)
                
                # Add trend info
                trend_direction = "Improving" if slope > 0 else "Declining"
                plt.text(
                    0.02, 0.02, 
                    f"Trend: {trend_direction} (slope: {slope:.4f})", 
                    transform=plt.gca().transAxes
                )
            except Exception as e:
                logger.warning(f"Could not calculate trend line: {str(e)}")
            
            plt.title('Sentiment Trend Over Time')
            plt.xlabel('Date')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved sentiment trend visualization to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing sentiment trend: {str(e)}")
    
    def compare_sentiment_by_source(self, df: pd.DataFrame, 
                                  source_col: str = 'Press',
                                  save_path: Optional[str] = None) -> None:
        """
        Compare sentiment distribution by news source.
        
        Args:
            df: DataFrame with sentiment analysis results
            source_col: Column name for news sources
            save_path: Path to save the visualization
        """
        if ('sentiment_score' not in df.columns or df['sentiment_score'].isna().all() or
            source_col not in df.columns):
            logger.warning("No sentiment scores or sources available for visualization")
            return
            
        try:
            # Group by source and calculate average sentiment
            source_sentiment = df.groupby(source_col)['sentiment_score'].agg(['mean', 'count']).reset_index()
            
            # Sort by count (descending)
            source_sentiment = source_sentiment.sort_values('count', ascending=False)
            
            # Take top 10 sources by count
            top_sources = source_sentiment.head(10)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Create bar plot
            bars = plt.bar(top_sources[source_col], top_sources['mean'])
            
            # Color bars by sentiment
            for i, bar in enumerate(bars):
                if top_sources['mean'].iloc[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            # Add count labels
            for i, v in enumerate(top_sources['count']):
                plt.text(i, top_sources['mean'].iloc[i] + 0.02, f"n={v}", ha='center')
            
            plt.title('Average Sentiment by News Source')
            plt.xlabel('News Source')
            plt.ylabel('Average Sentiment Score (-1 to 1)')
            plt.grid(True, alpha=0.3)
            
            # Add horizontal line at zero
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved source sentiment visualization to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error visualizing source sentiment: {str(e)}")
    
    def get_most_positive_negative(self, df: pd.DataFrame, 
                                 top_n: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get most positive and negative texts.
        
        Args:
            df: DataFrame with sentiment analysis results
            top_n: Number of texts to return
            
        Returns:
            Dictionary with most positive and negative texts
        """
        if 'sentiment_score' not in df.columns or df['sentiment_score'].isna().all():
            logger.warning("No sentiment scores available")
            return {'positive': pd.DataFrame(), 'negative': pd.DataFrame()}
            
        try:
            # Filter out rows without sentiment scores
            filtered_df = df.dropna(subset=['sentiment_score'])
            
            if len(filtered_df) == 0:
                logger.warning("No texts with sentiment scores")
                return {'positive': pd.DataFrame(), 'negative': pd.DataFrame()}
            
            # Get most positive texts
            positive_df = filtered_df.sort_values('sentiment_score', ascending=False).head(top_n)
            
            # Get most negative texts
            negative_df = filtered_df.sort_values('sentiment_score', ascending=True).head(top_n)
            
            return {'positive': positive_df, 'negative': negative_df}
        except Exception as e:
            logger.error(f"Error getting most positive/negative texts: {str(e)}")
            return {'positive': pd.DataFrame(), 'negative': pd.DataFrame()}
