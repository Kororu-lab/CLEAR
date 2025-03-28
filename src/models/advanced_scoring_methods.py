"""
Advanced scoring methods module for news recommendation and stock impact analysis.
Implements various scoring techniques using different embedding models.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedScoringMethods:
    """
    Advanced scoring methods for news recommendation and stock impact analysis.
    Implements various scoring techniques using different embedding models.
    """
    
    def __init__(self, 
                 use_kosimcse: bool = True,
                 use_bge_korean: bool = True,
                 cache_results: bool = True,
                 models_dir: str = None):
        """
        Initialize the advanced scoring methods.
        
        Args:
            use_kosimcse: Whether to use KoSimCSE-roberta for embeddings
            use_bge_korean: Whether to use bge-m3-korean for embeddings
            cache_results: Whether to cache results
            models_dir: Directory to save/load models and cache
        """
        self.use_kosimcse = use_kosimcse
        self.use_bge_korean = use_bge_korean
        self.cache_results = cache_results
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "scoring")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.bge_korean_model = None
        self.bge_korean_tokenizer = None
        
        # Initialize cache
        self.similarity_cache = {}
        self.embedding_cache = {}
        
        # Load models if enabled
        if self.use_kosimcse:
            self._initialize_kosimcse()
        
        if self.use_bge_korean:
            self._initialize_bge_korean()
        
        # Load cache if available
        if self.cache_results:
            self._load_cache()
        
        logger.info(f"Initialized AdvancedScoringMethods with models: " +
                   f"kosimcse={self.use_kosimcse}, bge_korean={self.use_bge_korean}")
    
    def _initialize_kosimcse(self):
        """
        Initialize the KoSimCSE-roberta model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load KoSimCSE-roberta model and tokenizer
            model_name = "BM-K/KoSimCSE-roberta"
            
            logger.info(f"Loading KoSimCSE-roberta model: {model_name}")
            
            self.kosimcse_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.kosimcse_model = AutoModel.from_pretrained(model_name)
            
            # Move to CPU (for demo purposes)
            self.kosimcse_model = self.kosimcse_model.to('cpu')
            
            # Set to evaluation mode
            self.kosimcse_model.eval()
            
            logger.info("KoSimCSE-roberta model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KoSimCSE-roberta: {str(e)}")
            self.use_kosimcse = False
    
    def _initialize_bge_korean(self):
        """
        Initialize the bge-m3-korean model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load bge-m3-korean model and tokenizer
            model_name = "upskyy/bge-m3-korean"
            
            logger.info(f"Loading bge-m3-korean model: {model_name}")
            
            self.bge_korean_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bge_korean_model = AutoModel.from_pretrained(model_name)
            
            # Move to CPU (for demo purposes)
            self.bge_korean_model = self.bge_korean_model.to('cpu')
            
            # Set to evaluation mode
            self.bge_korean_model.eval()
            
            logger.info("bge-m3-korean model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing bge-m3-korean: {str(e)}")
            self.use_bge_korean = False
    
    def _load_cache(self):
        """
        Load cache from disk.
        """
        try:
            # Load similarity cache
            similarity_cache_path = os.path.join(self.models_dir, "similarity_cache.json")
            if os.path.exists(similarity_cache_path):
                with open(similarity_cache_path, 'r', encoding='utf-8') as f:
                    self.similarity_cache = json.load(f)
                logger.info(f"Loaded similarity cache with {len(self.similarity_cache)} entries")
            
            # Load embedding cache
            embedding_cache_path = os.path.join(self.models_dir, "embedding_cache.json")
            if os.path.exists(embedding_cache_path):
                with open(embedding_cache_path, 'r', encoding='utf-8') as f:
                    # JSON can't handle numpy arrays directly, so we stored them as lists
                    cache_data = json.load(f)
                    self.embedding_cache = {k: np.array(v) for k, v in cache_data.items()}
                logger.info(f"Loaded embedding cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.similarity_cache = {}
            self.embedding_cache = {}
    
    def _save_cache(self):
        """
        Save cache to disk.
        """
        if not self.cache_results:
            return
            
        try:
            # Save similarity cache
            if self.similarity_cache:
                similarity_cache_path = os.path.join(self.models_dir, "similarity_cache.json")
                with open(similarity_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.similarity_cache, f, default=lambda x: float(x) if isinstance(x, np.float32) else x)
                logger.info(f"Saved similarity cache with {len(self.similarity_cache)} entries")
            
            # Save embedding cache
            if self.embedding_cache:
                embedding_cache_path = os.path.join(self.models_dir, "embedding_cache.json")
                
                # Convert numpy arrays to lists for JSON serialization
                cache_data = {k: v.tolist() for k, v in self.embedding_cache.items()}
                
                with open(embedding_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, default=lambda x: float(x) if isinstance(x, np.float32) else x)
                logger.info(f"Saved embedding cache with {len(self.embedding_cache)} entries")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def get_embedding(self, text: str, model: str = 'default') -> np.ndarray:
        """
        Get embedding for a text using the specified model.
        
        Args:
            text: Text to embed
            model: Model to use ('kosimcse', 'bge_korean', or 'default')
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        cache_key = f"{model}:{text}"
        if self.cache_results and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Get embedding based on model
        if model == 'kosimcse' and self.use_kosimcse and self.kosimcse_model is not None:
            embedding = self._get_kosimcse_embedding(text)
        elif model == 'bge_korean' and self.use_bge_korean and self.bge_korean_model is not None:
            embedding = self._get_bge_korean_embedding(text)
        elif model == 'default':
            # Use first available model
            if self.use_kosimcse and self.kosimcse_model is not None:
                embedding = self._get_kosimcse_embedding(text)
            elif self.use_bge_korean and self.bge_korean_model is not None:
                embedding = self._get_bge_korean_embedding(text)
            else:
                # Fall back to simple TF-IDF-like embedding
                embedding = self._get_simple_embedding(text)
        else:
            # Fall back to simple TF-IDF-like embedding
            embedding = self._get_simple_embedding(text)
        
        # Cache embedding
        if self.cache_results:
            self.embedding_cache[cache_key] = embedding
            # Periodically save cache
            if len(self.embedding_cache) % 100 == 0:
                self._save_cache()
        
        return embedding
    
    def _get_kosimcse_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KoSimCSE-roberta model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            import torch
            
            # Tokenize text
            inputs = self.kosimcse_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to same device as model
            inputs = {k: v.to(self.kosimcse_model.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.kosimcse_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Convert to numpy array
            embedding = embeddings.cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            logger.error(f"Error in KoSimCSE embedding: {str(e)}")
            # Fall back to simple embedding
            return self._get_simple_embedding(text)
    
    def _get_bge_korean_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using bge-m3-korean model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            import torch
            
            # Tokenize text
            inputs = self.bge_korean_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move to same device as model
            inputs = {k: v.to(self.bge_korean_model.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bge_korean_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Convert to numpy array
            embedding = embeddings.cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            logger.error(f"Error in bge-m3-korean embedding: {str(e)}")
            # Fall back to simple embedding
            return self._get_simple_embedding(text)
    
    def _get_simple_embedding(self, text: str) -> np.ndarray:
        """
        Get simple TF-IDF-like embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Simple character-level embedding
        chars = list(set(text))
        embedding = np.zeros(300)  # Use same dimension as other embeddings
        
        for i, char in enumerate(chars[:300]):
            embedding[i] = text.count(char) / len(text)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def calculate_similarity(self, text1: str, text2: str, model: str = 'default') -> float:
        """
        Calculate similarity between two texts using the specified model.
        
        Args:
            text1: First text
            text2: Second text
            model: Model to use ('kosimcse', 'bge_korean', or 'default')
            
        Returns:
            Similarity score between 0 and 1
        """
        # Check cache first
        cache_key = f"{model}:{text1}:{text2}"
        if self.cache_results and cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get embeddings
        embedding1 = self.get_embedding(text1, model)
        embedding2 = self.get_embedding(text2, model)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding1, embedding2)
        
        # Cache similarity
        if self.cache_results:
            self.similarity_cache[cache_key] = similarity
            # Also cache the reverse order
            self.similarity_cache[f"{model}:{text2}:{text1}"] = similarity
            # Periodically save cache
            if len(self.similarity_cache) % 100 == 0:
                self._save_cache()
        
        return similarity
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity between 0 and 1
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure similarity is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def calculate_similarity_matrix(self, texts: List[str], model: str = 'default') -> np.ndarray:
        """
        Calculate similarity matrix for a list of texts.
        
        Args:
            texts: List of texts
            model: Model to use ('kosimcse', 'bge_korean', or 'default')
            
        Returns:
            Similarity matrix as numpy array
        """
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self.calculate_similarity(texts[i], texts[j], model)
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def visualize_similarity_matrix(self, texts: List[str], labels: List[str] = None, model: str = 'default', title: str = "Text Similarity Matrix"):
        """
        Visualize similarity matrix for a list of texts.
        
        Args:
            texts: List of texts
            labels: List of labels for texts (optional)
            model: Model to use ('kosimcse', 'bge_korean', or 'default')
            title: Plot title
        """
        similarity_matrix = self.calculate_similarity_matrix(texts, model)
        
        if labels is None:
            labels = [f"Text {i+1}" for i in range(len(texts))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def calculate_ensemble_similarity(self, text1: str, text2: str, weights: Dict[str, float] = None) -> float:
        """
        Calculate ensemble similarity using multiple models with weights.
        
        Args:
            text1: First text
            text2: Second text
            weights: Dictionary mapping model names to weights
            
        Returns:
            Weighted ensemble similarity score between 0 and 1
        """
        if weights is None:
            # Default weights
            weights = {}
            if self.use_kosimcse:
                weights['kosimcse'] = 0.6
            if self.use_bge_korean:
                weights['bge_korean'] = 0.4
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            return self.calculate_similarity(text1, text2, 'default')
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted similarity
        weighted_similarity = 0.0
        for model, weight in normalized_weights.items():
            similarity = self.calculate_similarity(text1, text2, model)
            weighted_similarity += similarity * weight
        
        return weighted_similarity
    
    def calculate_impact_score(self, news_text: str, sentiment_score: float, price_change_pct: float, model: str = 'default') -> float:
        """
        Calculate impact score of news on stock price.
        
        Args:
            news_text: News text
            sentiment_score: Sentiment score between -1 and 1
            price_change_pct: Price change percentage
            model: Model to use for embedding
            
        Returns:
            Impact score between -1 and 1
        """
        # Extract key financial terms from news
        financial_terms = self._extract_financial_terms(news_text)
        
        # Calculate term importance
        term_importance = self._calculate_term_importance(financial_terms)
        
        # Calculate correlation factor
        correlation_factor = self._calculate_correlation_factor(sentiment_score, price_change_pct)
        
        # Calculate impact score
        impact_score = sentiment_score * term_importance * correlation_factor
        
        return impact_score
    
    def _extract_financial_terms(self, text: str) -> List[str]:
        """
        Extract key financial terms from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of financial terms
        """
        # Simple financial term extraction
        financial_terms = [
            '실적', '매출', '영업이익', '순이익', '증가', '감소', '성장', '하락',
            '투자', '주가', '시장', '경쟁', '전망', '계약', '개발', '출시',
            '인수', '합병', '파트너십', '협약', '기술', '특허', '연구', '혁신'
        ]
        
        return [term for term in financial_terms if term in text]
    
    def _calculate_term_importance(self, terms: List[str]) -> float:
        """
        Calculate importance of financial terms.
        
        Args:
            terms: List of financial terms
            
        Returns:
            Term importance score between 0 and 1
        """
        if not terms:
            return 0.5  # Neutral importance
        
        # Calculate term importance based on number of terms
        term_count_factor = min(1.0, len(terms) / 5)  # Cap at 1.0
        
        # Calculate importance
        importance = 0.5 + (0.5 * term_count_factor)
        
        return importance
    
    def _calculate_correlation_factor(self, sentiment_score: float, price_change_pct: float) -> float:
        """
        Calculate correlation factor between sentiment and price change.
        
        Args:
            sentiment_score: Sentiment score between -1 and 1
            price_change_pct: Price change percentage
            
        Returns:
            Correlation factor between 0 and 1
        """
        # Normalize price change to -1 to 1 range
        normalized_price_change = np.tanh(price_change_pct / 5.0)  # 5% change is significant
        
        # Calculate correlation
        if sentiment_score * normalized_price_change > 0:
            # Same direction (positive correlation)
            correlation = 0.8
        elif sentiment_score * normalized_price_change < 0:
            # Opposite direction (negative correlation)
            correlation = 0.2
        else:
            # No correlation
            correlation = 0.5
        
        return correlation
    
    def calculate_ensemble_impact_scores(self, df: pd.DataFrame, text_col: str, sentiment_col: str, price_change_col: str, models: List[str] = None, weights: List[float] = None) -> pd.DataFrame:
        """
        Calculate ensemble impact scores for a dataframe of news.
        
        Args:
            df: Dataframe with news and price data
            text_col: Column name for news text
            sentiment_col: Column name for sentiment score
            price_change_col: Column name for price change percentage
            models: List of models to use
            weights: List of weights for models
            
        Returns:
            Dataframe with impact scores
        """
        # Copy dataframe to avoid modifying original
        result_df = df.copy()
        
        # Default models and weights
        if models is None:
            models = []
            if self.use_kosimcse:
                models.append('kosimcse')
            if self.use_bge_korean:
                models.append('bge_korean')
            if not models:
                models = ['default']
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Ensure weights sum to 1
        weights = np.array(weights) / sum(weights)
        
        # Calculate impact scores for each model
        for i, model in enumerate(models):
            col_name = f"impact_{model}"
            result_df[col_name] = result_df.apply(
                lambda row: self.calculate_impact_score(
                    row[text_col], 
                    row[sentiment_col], 
                    row[price_change_col], 
                    model
                ), 
                axis=1
            )
        
        # Calculate ensemble impact score
        result_df['impact_ensemble'] = 0.0
        for i, model in enumerate(models):
            col_name = f"impact_{model}"
            result_df['impact_ensemble'] += result_df[col_name] * weights[i]
        
        return result_df
    
    def visualize_impact_scores(self, df: pd.DataFrame, text_col: str, impact_cols: List[str], title: str = "Impact Scores Comparison"):
        """
        Visualize impact scores for different models.
        
        Args:
            df: Dataframe with impact scores
            text_col: Column name for text labels
            impact_cols: List of column names for impact scores
            title: Plot title
        """
        # Prepare data for plotting
        plot_data = []
        for i, row in df.iterrows():
            for col in impact_cols:
                plot_data.append({
                    'Text': row[text_col][:30] + "..." if len(row[text_col]) > 30 else row[text_col],
                    'Model': col.replace('impact_', ''),
                    'Impact Score': row[col]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Text', y='Impact Score', hue='Model', data=plot_df)
        plt.title(title)
        plt.xlabel('News Text')
        plt.ylabel('Impact Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()
