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
                 models_dir: str = None,
                 use_kosimcse: bool = True,
                 use_bge_korean: bool = True,
                 use_gte_korean: bool = False,
                 use_e5_korean: bool = False,
                 cache_embeddings: bool = True,
                 device: str = 'cpu'):
        """
        Initialize the advanced scoring methods.
        
        Args:
            models_dir: Directory to save/load models and cache
            use_kosimcse: Whether to use KoSimCSE-roberta for embeddings
            use_bge_korean: Whether to use bge-m3-korean for embeddings
            use_gte_korean: Whether to use gte-base-korean for embeddings
            use_e5_korean: Whether to use e5-large-korean for embeddings
            cache_embeddings: Whether to cache embeddings
            device: Device to use for model inference ('cpu' or 'cuda')
        """
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "scoring")
        self.use_kosimcse = use_kosimcse
        self.use_bge_korean = use_bge_korean
        self.use_gte_korean = use_gte_korean
        self.use_e5_korean = use_e5_korean
        self.cache_embeddings = cache_embeddings
        self.device = device
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.bge_korean_model = None
        self.bge_korean_tokenizer = None
        self.gte_korean_model = None
        self.gte_korean_tokenizer = None
        self.e5_korean_model = None
        self.e5_korean_tokenizer = None
        
        # Initialize cache
        self.embedding_cache = {}
        
        # Load models if enabled
        if self.use_kosimcse:
            self._initialize_kosimcse()
        
        if self.use_bge_korean:
            self._initialize_bge_korean()
        
        if self.use_gte_korean:
            self._initialize_gte_korean()
        
        if self.use_e5_korean:
            self._initialize_e5_korean()
        
        # Load cache if available
        if self.cache_embeddings:
            self._load_cache()
        
        logger.info(f"Initialized AdvancedScoringMethods with models: " +
                   f"kosimcse={self.use_kosimcse}, bge_korean={self.use_bge_korean}, " +
                   f"gte_korean={self.use_gte_korean}, e5_korean={self.use_e5_korean}")
    
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
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.kosimcse_model = self.kosimcse_model.to('cuda')
                logger.info("Using CUDA for KoSimCSE-roberta")
            else:
                self.kosimcse_model = self.kosimcse_model.to('cpu')
                logger.info("Using CPU for KoSimCSE-roberta")
            
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
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.bge_korean_model = self.bge_korean_model.to('cuda')
                logger.info("Using CUDA for bge-m3-korean")
            else:
                self.bge_korean_model = self.bge_korean_model.to('cpu')
                logger.info("Using CPU for bge-m3-korean")
            
            # Set to evaluation mode
            self.bge_korean_model.eval()
            
            logger.info("bge-m3-korean model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing bge-m3-korean: {str(e)}")
            self.use_bge_korean = False
    
    def _initialize_gte_korean(self):
        """
        Initialize the gte-base-korean model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load gte-base-korean model and tokenizer
            model_name = "upskyy/gte-base-korean"
            
            logger.info(f"Loading gte-base-korean model: {model_name}")
            
            self.gte_korean_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.gte_korean_model = AutoModel.from_pretrained(model_name)
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.gte_korean_model = self.gte_korean_model.to('cuda')
                logger.info("Using CUDA for gte-base-korean")
            else:
                self.gte_korean_model = self.gte_korean_model.to('cpu')
                logger.info("Using CPU for gte-base-korean")
            
            # Set to evaluation mode
            self.gte_korean_model.eval()
            
            logger.info("gte-base-korean model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing gte-base-korean: {str(e)}")
            self.use_gte_korean = False
    
    def _initialize_e5_korean(self):
        """
        Initialize the e5-large-korean model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load e5-large-korean model and tokenizer
            model_name = "upskyy/e5-large-korean"
            
            logger.info(f"Loading e5-large-korean model: {model_name}")
            
            self.e5_korean_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.e5_korean_model = AutoModel.from_pretrained(model_name)
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.e5_korean_model = self.e5_korean_model.to('cuda')
                logger.info("Using CUDA for e5-large-korean")
            else:
                self.e5_korean_model = self.e5_korean_model.to('cpu')
                logger.info("Using CPU for e5-large-korean")
            
            # Set to evaluation mode
            self.e5_korean_model.eval()
            
            logger.info("e5-large-korean model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing e5-large-korean: {str(e)}")
            self.use_e5_korean = False
    
    def _load_cache(self):
        """
        Load embedding cache from disk.
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
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            self.embedding_cache = {}
    
    def _save_cache(self):
        """
        Save embedding cache to disk.
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
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def get_embedding(self, text: str, model: str = 'default') -> np.ndarray:
        """
        Get embedding vector for Korean text.
        
        Args:
            text: Text to embed
            model: Model to use ('kosimcse', 'bge_korean', 'gte_korean', 'e5_korean', or 'default')
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not isinstance(text, str):
            # Return zero vector of appropriate size
            if model == 'bge_korean':
                return np.zeros(1024)  # bge-m3-korean embeddings are 1024-dimensional
            elif model == 'e5_korean':
                return np.zeros(1024)  # e5-large-korean embeddings are 1024-dimensional
            else:
                return np.zeros(768)  # Most BERT-based embeddings are 768-dimensional
        
        # Determine which model to use if 'default'
        if model == 'default':
            if self.use_bge_korean:
                model = 'bge_korean'
            elif self.use_kosimcse:
                model = 'kosimcse'
            elif self.use_gte_korean:
                model = 'gte_korean'
            elif self.use_e5_korean:
                model = 'e5_korean'
            else:
                # Fallback to simple embedding
                return self._get_simple_embedding(text)
        
        # Create cache key
        cache_key = f"{model}:{text}"
        
        # Check cache
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            # Get embedding based on specified model
            if model == 'kosimcse' and self.use_kosimcse:
                embedding = self._get_kosimcse_embedding(text)
            elif model == 'bge_korean' and self.use_bge_korean:
                embedding = self._get_bge_korean_embedding(text)
            elif model == 'gte_korean' and self.use_gte_korean:
                embedding = self._get_gte_korean_embedding(text)
            elif model == 'e5_korean' and self.use_e5_korean:
                embedding = self._get_e5_korean_embedding(text)
            else:
                # Fallback to simple embedding
                embedding = self._get_simple_embedding(text)
            
            # Cache embedding
            if self.cache_embeddings:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding with model {model}: {str(e)}")
            
            # Return zero vector of appropriate size
            if model == 'bge_korean' or model == 'e5_korean':
                return np.zeros(1024)
            else:
                return np.zeros(768)
    
    def _get_kosimcse_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KoSimCSE-roberta model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize text
        inputs = self.kosimcse_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.kosimcse_model(**inputs, return_dict=False)
            
            # KoSimCSE returns a tuple of (embeddings, pooler_output)
            embeddings = outputs[0]
            
            # Use CLS token embedding (first token)
            embedding = embeddings[0, 0, :].cpu().numpy()
        
        return embedding
    
    def _get_bge_korean_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using bge-m3-korean model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Mean Pooling function for bge-m3-korean
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Tokenize text
        inputs = self.bge_korean_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bge_korean_model(**inputs)
            
            # Perform mean pooling
            embedding = mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()[0]
        
        return embedding
    
    def _get_gte_korean_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using gte-base-korean model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Mean Pooling function for gte-base-korean
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Tokenize text
        inputs = self.gte_korean_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.gte_korean_model(**inputs)
            
            # Perform mean pooling
            embedding = mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()[0]
        
        return embedding
    
    def _get_e5_korean_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using e5-large-korean model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Mean Pooling function for e5-large-korean
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Tokenize text
        inputs = self.e5_korean_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.e5_korean_model(**inputs)
            
            # Perform mean pooling
            embedding = mean_pooling(outputs, inputs['attention_mask']).cpu().numpy()[0]
        
        return embedding
    
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
    
    def calculate_similarity(self, text1: str, text2: str, model: str = 'default') -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            model: Model to use for embeddings
            
        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        embedding1 = self.get_embedding(text1, model)
        embedding2 = self.get_embedding(text2, model)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding1, embedding2)
        
        return similarity
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Normalize vectors
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        
        # Calculate cosine similarity
        similarity = np.dot(a_norm, b_norm)
        
        return similarity
    
    def calculate_batch_similarities(self, query_text: str, texts: List[str], 
                                   model: str = 'default') -> List[float]:
        """
        Calculate similarities between a query text and multiple texts.
        
        Args:
            query_text: Query text
            texts: List of texts to compare to
            model: Model to use for embeddings
            
        Returns:
            List of similarity scores (0-1)
        """
        # Get query embedding
        query_embedding = self.get_embedding(query_text, model)
        
        # Get embeddings for all texts
        embeddings = [self.get_embedding(text, model) for text in texts]
        
        # Calculate similarities
        similarities = [self._cosine_similarity(query_embedding, embedding) for embedding in embeddings]
        
        return similarities
    
    def calculate_similarity_matrix(self, texts: List[str], model: str = 'default') -> np.ndarray:
        """
        Calculate similarity matrix for a list of texts.
        
        Args:
            texts: List of texts
            model: Model to use for embeddings
            
        Returns:
            Similarity matrix (n x n)
        """
        # Get embeddings for all texts
        embeddings = [self.get_embedding(text, model) for text in texts]
        
        # Calculate similarity matrix
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def calculate_ensemble_similarity(self, text1: str, text2: str, 
                                    models: List[str] = None,
                                    weights: List[float] = None) -> float:
        """
        Calculate ensemble similarity using multiple models.
        
        Args:
            text1: First text
            text2: Second text
            models: List of models to use
            weights: List of weights for each model
            
        Returns:
            Ensemble similarity score (0-1)
        """
        # Default to all available models
        if models is None:
            models = []
            if self.use_kosimcse:
                models.append('kosimcse')
            if self.use_bge_korean:
                models.append('bge_korean')
            if self.use_gte_korean:
                models.append('gte_korean')
            if self.use_e5_korean:
                models.append('e5_korean')
            
            if not models:
                # Fallback to simple embedding
                return self.calculate_similarity(text1, text2, 'default')
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Ensure weights sum to 1
        weights = [w / sum(weights) for w in weights]
        
        # Calculate similarity for each model
        similarities = []
        for model in models:
            similarity = self.calculate_similarity(text1, text2, model)
            similarities.append(similarity)
        
        # Calculate weighted average
        ensemble_similarity = sum(s * w for s, w in zip(similarities, weights))
        
        return ensemble_similarity
    
    def vectorize_dataframe(self, df: pd.DataFrame, 
                          text_col: str = 'Title',
                          content_col: str = 'Body',
                          use_content: bool = True,
                          model: str = 'default') -> np.ndarray:
        """
        Vectorize texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts
            text_col: Column name for primary text (e.g., title)
            content_col: Column name for secondary text (e.g., body)
            use_content: Whether to use content for vectorization
            model: Model to use for embeddings
            
        Returns:
            Array of embeddings (n_samples, n_features)
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for vectorization")
            return np.array([])
            
        logger.info(f"Vectorizing {len(df)} texts with model {model}")
        
        try:
            # Initialize list to store embeddings
            embeddings = []
            
            # Process each row
            for idx, row in df.iterrows():
                # Determine text to vectorize
                if text_col in row and isinstance(row[text_col], str):
                    text = row[text_col]
                    
                    # Add content if specified
                    if use_content and content_col in row and isinstance(row[content_col], str):
                        text += " " + row[content_col]
                else:
                    # Skip if no text
                    continue
                
                # Get embedding
                embedding = self.get_embedding(text, model)
                
                # Add to list
                embeddings.append(embedding)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            # Save cache
            self._save_cache()
            
            logger.info(f"Vectorized {len(embeddings)} texts with model {model}")
            
            return embeddings_array
        except Exception as e:
            logger.error(f"Error vectorizing DataFrame: {str(e)}")
            return np.array([])
    
    def calculate_ensemble_vectors(self, df: pd.DataFrame,
                                 text_col: str = 'Title',
                                 content_col: str = 'Body',
                                 use_content: bool = True,
                                 models: List[str] = None,
                                 weights: List[float] = None) -> np.ndarray:
        """
        Calculate ensemble vectors using multiple models.
        
        Args:
            df: DataFrame containing texts
            text_col: Column name for primary text (e.g., title)
            content_col: Column name for secondary text (e.g., body)
            use_content: Whether to use content for vectorization
            models: List of models to use
            weights: List of weights for each model
            
        Returns:
            Array of ensemble embeddings (n_samples, n_features)
        """
        # Default to all available models
        if models is None:
            models = []
            if self.use_kosimcse:
                models.append('kosimcse')
            if self.use_bge_korean:
                models.append('bge_korean')
            if self.use_gte_korean:
                models.append('gte_korean')
            if self.use_e5_korean:
                models.append('e5_korean')
            
            if not models:
                # Fallback to simple embedding
                return self.vectorize_dataframe(df, text_col, content_col, use_content, 'default')
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Ensure weights sum to 1
        weights = [w / sum(weights) for w in weights]
        
        # Calculate vectors for each model
        model_vectors = []
        for model in models:
            vectors = self.vectorize_dataframe(df, text_col, content_col, use_content, model)
            
            # Normalize vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized_vectors = vectors / np.maximum(norms, 1e-10)
            
            model_vectors.append(normalized_vectors)
        
        # Calculate weighted average
        ensemble_vectors = np.zeros_like(model_vectors[0])
        for i, vectors in enumerate(model_vectors):
            ensemble_vectors += vectors * weights[i]
        
        # Normalize ensemble vectors
        norms = np.linalg.norm(ensemble_vectors, axis=1, keepdims=True)
        ensemble_vectors = ensemble_vectors / np.maximum(norms, 1e-10)
        
        return ensemble_vectors
    
    def calculate_stock_impact_score(self, news_text: str, 
                                   sentiment_score: float,
                                   price_change: float,
                                   model: str = 'default') -> float:
        """
        Calculate stock impact score using advanced embedding models.
        
        Args:
            news_text: News text
            sentiment_score: Sentiment score (-1 to 1)
            price_change: Price change percentage
            model: Model to use for embeddings
            
        Returns:
            Impact score (-5 to 5)
        """
        # Get embedding
        embedding = self.get_embedding(news_text, model)
        
        # Define positive and negative financial keywords
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
        
        # Calculate similarity to positive and negative keywords
        positive_similarities = []
        for keyword in positive_keywords:
            keyword_embedding = self.get_embedding(keyword, model)
            similarity = self._cosine_similarity(embedding, keyword_embedding)
            positive_similarities.append(similarity)
        
        negative_similarities = []
        for keyword in negative_keywords:
            keyword_embedding = self.get_embedding(keyword, model)
            similarity = self._cosine_similarity(embedding, keyword_embedding)
            negative_similarities.append(similarity)
        
        # Calculate average similarities
        avg_positive_similarity = sum(positive_similarities) / len(positive_similarities)
        avg_negative_similarity = sum(negative_similarities) / len(negative_similarities)
        
        # Calculate keyword impact
        keyword_impact = (avg_positive_similarity - avg_negative_similarity) * 2.5  # Scale to -2.5 to 2.5
        
        # Combine with sentiment and price change
        sentiment_impact = sentiment_score * 1.5  # Scale to -1.5 to 1.5
        price_impact = price_change / 2  # Scale to -5 to 5 (assuming price_change is -10 to 10)
        
        # Calculate overall impact
        impact_score = keyword_impact * 0.3 + sentiment_impact * 0.3 + price_impact * 0.4
        
        # Clip to range
        impact_score = max(min(impact_score, 5), -5)
        
        return impact_score
    
    def calculate_ensemble_impact_scores(self, df: pd.DataFrame,
                                       text_col: str = 'Title',
                                       sentiment_col: str = 'sentiment_score',
                                       price_change_col: str = 'price_change_pct_1d',
                                       models: List[str] = None,
                                       weights: List[float] = None) -> pd.DataFrame:
        """
        Calculate ensemble impact scores using multiple models.
        
        Args:
            df: DataFrame containing news articles
            text_col: Column name for news text
            sentiment_col: Column name for sentiment scores
            price_change_col: Column name for price changes
            models: List of models to use
            weights: List of weights for each model
            
        Returns:
            DataFrame with impact scores
        """
        # Default to all available models
        if models is None:
            models = []
            if self.use_kosimcse:
                models.append('kosimcse')
            if self.use_bge_korean:
                models.append('bge_korean')
            if self.use_gte_korean:
                models.append('gte_korean')
            if self.use_e5_korean:
                models.append('e5_korean')
            
            if not models:
                models = ['default']
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Ensure weights sum to 1
        weights = [w / sum(weights) for w in weights]
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Initialize impact score columns
        for model in models:
            result_df[f'impact_{model}'] = None
        
        result_df['impact_ensemble'] = None
        
        # Calculate impact scores for each model
        for idx, row in result_df.iterrows():
            # Skip if missing required data
            if (text_col not in row or not isinstance(row[text_col], str) or
                sentiment_col not in row or pd.isna(row[sentiment_col]) or
                price_change_col not in row or pd.isna(row[price_change_col])):
                continue
            
            # Get text, sentiment, and price change
            text = row[text_col]
            sentiment = row[sentiment_col]
            price_change = row[price_change_col]
            
            # Calculate impact scores for each model
            model_scores = {}
            for model in models:
                score = self.calculate_stock_impact_score(text, sentiment, price_change, model)
                result_df.at[idx, f'impact_{model}'] = score
                model_scores[model] = score
            
            # Calculate ensemble score
            ensemble_score = sum(model_scores[model] * weight for model, weight in zip(models, weights))
            result_df.at[idx, 'impact_ensemble'] = ensemble_score
        
        return result_df
    
    def visualize_model_comparison(self, df: pd.DataFrame,
                                 models: List[str] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize comparison of impact scores from different models.
        
        Args:
            df: DataFrame with impact scores
            models: List of models to compare
            save_path: Path to save the visualization
        """
        # Default to all available models
        if models is None:
            models = []
            for col in df.columns:
                if col.startswith('impact_') and col != 'impact_ensemble':
                    models.append(col.replace('impact_', ''))
        
        if not models:
            logger.warning("No impact score columns found for visualization")
            return
            
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create subplots
            fig, axes = plt.subplots(len(models), 1, figsize=(12, 4 * len(models)), sharex=True)
            
            # If only one model, axes is not a list
            if len(models) == 1:
                axes = [axes]
            
            # Plot histograms for each model
            for i, model in enumerate(models):
                col = f'impact_{model}'
                if col in df.columns:
                    sns.histplot(df[col].dropna(), bins=20, kde=True, ax=axes[i])
                    axes[i].set_title(f'Impact Score Distribution - {model}')
                    axes[i].set_xlabel('Impact Score (-5 to 5)')
                    axes[i].set_ylabel('Count')
                    axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved model comparison visualization to {save_path}")
            else:
                plt.show()
                
            # Create correlation matrix figure
            plt.figure(figsize=(10, 8))
            
            # Get impact score columns
            impact_cols = [f'impact_{model}' for model in models]
            impact_cols = [col for col in impact_cols if col in df.columns]
            
            if len(impact_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = df[impact_cols].corr()
                
                # Plot correlation matrix
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Correlation Matrix of Impact Scores')
                plt.tight_layout()
                
                # Save or show
                if save_path:
                    correlation_path = save_path.replace('.png', '_correlation.png')
                    plt.savefig(correlation_path)
                    logger.info(f"Saved correlation matrix visualization to {correlation_path}")
                else:
                    plt.show()
        except Exception as e:
            logger.error(f"Error visualizing model comparison: {str(e)}")
    
    def visualize_ensemble_weights(self, weights: List[float],
                                 models: List[str] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize ensemble weights for different models.
        
        Args:
            weights: List of weights
            models: List of model names
            save_path: Path to save the visualization
        """
        # Default model names
        if models is None:
            models = []
            if self.use_kosimcse:
                models.append('kosimcse')
            if self.use_bge_korean:
                models.append('bge_korean')
            if self.use_gte_korean:
                models.append('gte_korean')
            if self.use_e5_korean:
                models.append('e5_korean')
            
            if not models:
                models = ['default']
        
        # Ensure weights and models have the same length
        if len(weights) != len(models):
            logger.warning("Number of weights and models must be the same")
            return
            
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create bar plot
            plt.bar(models, weights)
            
            plt.title('Ensemble Weights for Different Models')
            plt.xlabel('Model')
            plt.ylabel('Weight')
            plt.ylim(0, max(weights) * 1.2)
            
            # Add value labels
            for i, v in enumerate(weights):
                plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Saved ensemble weights visualization to {save_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Error visualizing ensemble weights: {str(e)}")
