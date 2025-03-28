"""
Korean financial text analyzer module with KO-finbert support.
Provides sentiment analysis and financial impact scoring for Korean news.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KoreanFinancialTextAnalyzer:
    """
    Korean financial text analyzer with KO-finbert support.
    Provides sentiment analysis and financial impact scoring for Korean news.
    """
    
    def __init__(self, 
                 use_finbert: bool = True,
                 use_advanced_embeddings: bool = False,
                 cache_results: bool = True,
                 models_dir: str = None):
        """
        Initialize the Korean financial text analyzer.
        
        Args:
            use_finbert: Whether to use KO-finbert for sentiment analysis
            use_advanced_embeddings: Whether to use advanced embeddings
            cache_results: Whether to cache results
            models_dir: Directory to save/load models and cache
        """
        self.use_finbert = use_finbert
        self.use_advanced_embeddings = use_advanced_embeddings
        self.cache_results = cache_results
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "text_analyzer")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.finbert_model = None
        self.finbert_tokenizer = None
        
        # Initialize cache
        self.analysis_cache = {}
        
        # Load models if enabled
        if self.use_finbert:
            self._initialize_finbert()
        
        # Load cache if available
        if self.cache_results:
            self._load_cache()
        
        logger.info(f"Initialized KoreanFinancialTextAnalyzer with finbert={self.use_finbert}, " +
                   f"advanced_embeddings={self.use_advanced_embeddings}")
    
    def _initialize_finbert(self):
        """
        Initialize the KO-finbert model for sentiment analysis.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Load KO-finbert model and tokenizer
            model_name = "snunlp/KR-FinBert-SC"
            
            logger.info(f"Loading KO-finbert model: {model_name}")
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to CPU (for demo purposes)
            self.finbert_model = self.finbert_model.to('cpu')
            
            # Set to evaluation mode
            self.finbert_model.eval()
            
            logger.info("KO-finbert model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KO-finbert: {str(e)}")
            self.use_finbert = False
    
    def _load_cache(self):
        """
        Load analysis cache from disk.
        """
        try:
            # Load analysis cache
            analysis_cache_path = os.path.join(self.models_dir, "analysis_cache.json")
            if os.path.exists(analysis_cache_path):
                with open(analysis_cache_path, 'r', encoding='utf-8') as f:
                    self.analysis_cache = json.load(f)
                logger.info(f"Loaded analysis cache with {len(self.analysis_cache)} entries")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.analysis_cache = {}
    
    def _save_cache(self):
        """
        Save analysis cache to disk.
        """
        if not self.cache_results:
            return
            
        try:
            # Save analysis cache
            if self.analysis_cache:
                analysis_cache_path = os.path.join(self.models_dir, "analysis_cache.json")
                with open(analysis_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_cache, f)
                logger.info(f"Saved analysis cache with {len(self.analysis_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze Korean financial text for sentiment and impact.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Check cache first
        if self.cache_results and text in self.analysis_cache:
            return self.analysis_cache[text]
        
        # Analyze sentiment
        if self.use_finbert and self.finbert_model is not None:
            sentiment_result = self._analyze_sentiment_finbert(text)
        else:
            sentiment_result = self._analyze_sentiment_simple(text)
        
        # Extract financial entities
        financial_entities = self._extract_financial_entities(text)
        
        # Combine results
        result = {
            'text': text,
            'sentiment_score': sentiment_result['sentiment_score'],
            'sentiment_label': sentiment_result['sentiment_label'],
            'confidence': sentiment_result['confidence'],
            'financial_entities': financial_entities
        }
        
        # Cache result
        if self.cache_results:
            self.analysis_cache[text] = result
            # Periodically save cache
            if len(self.analysis_cache) % 100 == 0:
                self._save_cache()
        
        return result
    
    def _analyze_sentiment_finbert(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using KO-finbert model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            import torch
            import torch.nn.functional as F
            
            # Tokenize text
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to same device as model
            inputs = {k: v.to(self.finbert_model.device) for k, v in inputs.items()}
            
            # Get sentiment prediction
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
            
            # Convert to numpy
            probabilities = probabilities.cpu().numpy()[0]
            
            # KO-finbert has 3 classes: negative (0), neutral (1), positive (2)
            sentiment_score = (probabilities[2] - probabilities[0])  # Range: -1 to 1
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            sentiment_label = ['negative', 'neutral', 'positive'][predicted_class]
            
            # Get confidence
            confidence = probabilities[predicted_class]
            
            return {
                'sentiment_score': float(sentiment_score),
                'sentiment_label': sentiment_label,
                'confidence': float(confidence),
                'probabilities': {
                    'negative': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'positive': float(probabilities[2])
                }
            }
        except Exception as e:
            logger.error(f"Error in KO-finbert sentiment analysis: {str(e)}")
            # Fall back to simple sentiment analysis
            return self._analyze_sentiment_simple(text)
    
    def _analyze_sentiment_simple(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using simple keyword-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        positive_words = ['상승', '급등', '호조', '성장', '개선', '흑자', '최대', '신기록']
        negative_words = ['하락', '급락', '부진', '감소', '악화', '적자', '손실', '하향']
        
        if not isinstance(text, str):
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 1.0,
                'probabilities': {
                    'negative': 0.0,
                    'neutral': 1.0,
                    'positive': 0.0
                }
            }
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count + neg_count == 0:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
            confidence = 0.7  # Lower confidence for keyword-based approach
            probabilities = {'negative': 0.15, 'neutral': 0.7, 'positive': 0.15}
        else:
            sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)
            
            if sentiment_score > 0.2:
                sentiment_label = 'positive'
                confidence = 0.5 + (sentiment_score * 0.3)  # Range: 0.5 to 0.8
                probabilities = {
                    'negative': 0.1,
                    'neutral': 0.9 - confidence,
                    'positive': confidence
                }
            elif sentiment_score < -0.2:
                sentiment_label = 'negative'
                confidence = 0.5 + (abs(sentiment_score) * 0.3)  # Range: 0.5 to 0.8
                probabilities = {
                    'negative': confidence,
                    'neutral': 0.9 - confidence,
                    'positive': 0.1
                }
            else:
                sentiment_label = 'neutral'
                confidence = 0.6
                probabilities = {
                    'negative': (0.4 - (sentiment_score * 0.5)) / 2,
                    'neutral': 0.6,
                    'positive': (0.4 + (sentiment_score * 0.5)) / 2
                }
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def _extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract financial entities from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of financial entities
        """
        # Simple financial entity extraction
        companies = ['삼성전자', 'SK하이닉스', 'LG전자', '현대자동차', '카카오', '네이버']
        financial_terms = ['실적', '매출', '영업이익', '순이익', '주가', '시장', '투자']
        
        entities = []
        
        # Extract companies
        for company in companies:
            if company in text:
                entities.append({
                    'type': 'company',
                    'name': company,
                    'position': text.find(company)
                })
        
        # Extract financial terms
        for term in financial_terms:
            if term in text:
                entities.append({
                    'type': 'financial_term',
                    'name': term,
                    'position': text.find(term)
                })
        
        # Sort by position
        entities.sort(key=lambda x: x['position'])
        
        return entities
    
    def analyze_texts_batch(self, texts: List[str], show_progress: bool = False) -> List[Dict[str, Any]]:
        """
        Analyze a batch of texts.
        
        Args:
            texts: List of texts to analyze
            show_progress: Whether to show progress bar
            
        Returns:
            List of analysis results
        """
        results = []
        
        iterator = texts
        if show_progress:
            iterator = tqdm(texts, desc="Analyzing texts")
        
        for text in iterator:
            results.append(self.analyze_text(text))
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_col: str, result_prefix: str = 'sentiment_', 
                        show_progress: bool = False) -> pd.DataFrame:
        """
        Analyze texts in a dataframe.
        
        Args:
            df: Dataframe with texts
            text_col: Column name for texts
            result_prefix: Prefix for result columns
            show_progress: Whether to show progress bar
            
        Returns:
            Dataframe with analysis results
        """
        # Copy dataframe to avoid modifying original
        result_df = df.copy()
        
        # Analyze texts
        texts = df[text_col].tolist()
        results = self.analyze_texts_batch(texts, show_progress)
        
        # Add results to dataframe
        result_df[f'{result_prefix}score'] = [r['sentiment_score'] for r in results]
        result_df[f'{result_prefix}label'] = [r['sentiment_label'] for r in results]
        result_df[f'{result_prefix}confidence'] = [r['confidence'] for r in results]
        
        return result_df
    
    def visualize_sentiment_distribution(self, df: pd.DataFrame, label_col: str = 'sentiment_label', 
                                       title: str = "Sentiment Distribution"):
        """
        Visualize sentiment distribution in a dataframe.
        
        Args:
            df: Dataframe with sentiment labels
            label_col: Column name for sentiment labels
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=label_col, data=df, order=['negative', 'neutral', 'positive'])
        plt.title(title)
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    def visualize_sentiment_scores(self, df: pd.DataFrame, score_col: str = 'sentiment_score', 
                                 text_col: str = None, title: str = "Sentiment Scores"):
        """
        Visualize sentiment scores in a dataframe.
        
        Args:
            df: Dataframe with sentiment scores
            score_col: Column name for sentiment scores
            text_col: Column name for text labels (optional)
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        if text_col is not None:
            # Use text as labels
            labels = df[text_col].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
            plt.bar(labels, df[score_col])
            plt.xticks(rotation=45, ha='right')
        else:
            # Use indices as labels
            plt.bar(range(len(df)), df[score_col])
            plt.xticks(range(len(df)), [f"Text {i+1}" for i in range(len(df))])
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(title)
        plt.xlabel('Text')
        plt.ylabel('Sentiment Score')
        plt.tight_layout()
        plt.show()
"""
