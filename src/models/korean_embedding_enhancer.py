"""
Enhanced Korean text embedding module with multiple Hugging Face models.
Provides advanced embedding capabilities for Korean financial text analysis.
"""

import os
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KoreanEmbeddingEnhancer:
    """
    Enhanced Korean text embedding with multiple Hugging Face models.
    Provides advanced embedding capabilities for Korean financial text analysis.
    """
    
    def __init__(self, 
                 models_dir: str = None,
                 use_kobert: bool = True,
                 use_klue_roberta: bool = True,
                 use_kosimcse: bool = True,
                 use_bge_korean: bool = True,
                 use_kpf_bert: bool = False,
                 use_koelectra: bool = False,
                 cache_embeddings: bool = True,
                 device: str = 'cpu'):
        """
        Initialize the Korean embedding enhancer.
        
        Args:
            models_dir: Directory to save/load models and cache
            use_kobert: Whether to use KoBERT for embeddings
            use_klue_roberta: Whether to use KLUE-RoBERTa for embeddings
            use_kosimcse: Whether to use KoSimCSE-roberta for embeddings
            use_bge_korean: Whether to use bge-m3-korean for embeddings
            use_kpf_bert: Whether to use KPF-BERT for embeddings
            use_koelectra: Whether to use KoELECTRA for embeddings
            cache_embeddings: Whether to cache embeddings
            device: Device to use for model inference ('cpu' or 'cuda')
        """
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "embeddings")
        self.use_kobert = use_kobert
        self.use_klue_roberta = use_klue_roberta
        self.use_kosimcse = use_kosimcse
        self.use_bge_korean = use_bge_korean
        self.use_kpf_bert = use_kpf_bert
        self.use_koelectra = use_koelectra
        self.cache_embeddings = cache_embeddings
        self.device = device
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models
        self.kobert_model = None
        self.kobert_tokenizer = None
        self.klue_roberta_model = None
        self.klue_roberta_tokenizer = None
        self.kosimcse_model = None
        self.kosimcse_tokenizer = None
        self.bge_korean_model = None
        self.bge_korean_tokenizer = None
        self.kpf_bert_model = None
        self.kpf_bert_tokenizer = None
        self.koelectra_model = None
        self.koelectra_tokenizer = None
        
        # Initialize cache
        self.embedding_cache = {}
        
        # Load models if enabled
        if self.use_kobert:
            self._initialize_kobert()
        
        if self.use_klue_roberta:
            self._initialize_klue_roberta()
        
        if self.use_kosimcse:
            self._initialize_kosimcse()
        
        if self.use_bge_korean:
            self._initialize_bge_korean()
        
        if self.use_kpf_bert:
            self._initialize_kpf_bert()
        
        if self.use_koelectra:
            self._initialize_koelectra()
        
        # Load cache if available
        if self.cache_embeddings:
            self._load_cache()
        
        logger.info(f"Initialized KoreanEmbeddingEnhancer with models: " +
                   f"kobert={self.use_kobert}, klue_roberta={self.use_klue_roberta}, " +
                   f"kosimcse={self.use_kosimcse}, bge_korean={self.use_bge_korean}, " +
                   f"kpf_bert={self.use_kpf_bert}, koelectra={self.use_koelectra}")
    
    def _initialize_kobert(self):
        """
        Initialize the KoBERT model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load KoBERT model and tokenizer
            model_name = "monologg/kobert"
            
            logger.info(f"Loading KoBERT model: {model_name}")
            
            self.kobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.kobert_model = AutoModel.from_pretrained(model_name)
            
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
            self.use_kobert = False
    
    def _initialize_klue_roberta(self):
        """
        Initialize the KLUE-RoBERTa model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load KLUE-RoBERTa model and tokenizer
            model_name = "klue/roberta-base"
            
            logger.info(f"Loading KLUE-RoBERTa model: {model_name}")
            
            self.klue_roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.klue_roberta_model = AutoModel.from_pretrained(model_name)
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.klue_roberta_model = self.klue_roberta_model.to('cuda')
                logger.info("Using CUDA for KLUE-RoBERTa")
            else:
                self.klue_roberta_model = self.klue_roberta_model.to('cpu')
                logger.info("Using CPU for KLUE-RoBERTa")
            
            # Set to evaluation mode
            self.klue_roberta_model.eval()
            
            logger.info("KLUE-RoBERTa model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KLUE-RoBERTa: {str(e)}")
            self.use_klue_roberta = False
    
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
    
    def _initialize_kpf_bert(self):
        """
        Initialize the KPF-BERT model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load KPF-BERT model and tokenizer
            model_name = "kpfbert/kpfbert"
            
            logger.info(f"Loading KPF-BERT model: {model_name}")
            
            self.kpf_bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.kpf_bert_model = AutoModel.from_pretrained(model_name)
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.kpf_bert_model = self.kpf_bert_model.to('cuda')
                logger.info("Using CUDA for KPF-BERT")
            else:
                self.kpf_bert_model = self.kpf_bert_model.to('cpu')
                logger.info("Using CPU for KPF-BERT")
            
            # Set to evaluation mode
            self.kpf_bert_model.eval()
            
            logger.info("KPF-BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KPF-BERT: {str(e)}")
            self.use_kpf_bert = False
    
    def _initialize_koelectra(self):
        """
        Initialize the KoELECTRA model for embeddings.
        """
        try:
            # Import here to avoid dependency if not using the model
            from transformers import AutoModel, AutoTokenizer
            
            # Load KoELECTRA model and tokenizer
            model_name = "monologg/koelectra-base-v3-discriminator"
            
            logger.info(f"Loading KoELECTRA model: {model_name}")
            
            self.koelectra_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.koelectra_model = AutoModel.from_pretrained(model_name)
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.koelectra_model = self.koelectra_model.to('cuda')
                logger.info("Using CUDA for KoELECTRA")
            else:
                self.koelectra_model = self.koelectra_model.to('cpu')
                logger.info("Using CPU for KoELECTRA")
            
            # Set to evaluation mode
            self.koelectra_model.eval()
            
            logger.info("KoELECTRA model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing KoELECTRA: {str(e)}")
            self.use_koelectra = False
    
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
            model: Model to use ('kobert', 'klue_roberta', 'kosimcse', 'bge_korean', 
                                'kpf_bert', 'koelectra', or 'default')
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not isinstance(text, str):
            # Return zero vector of appropriate size
            if model == 'bge_korean':
                return np.zeros(1024)  # bge-m3-korean embeddings are 1024-dimensional
            else:
                return np.zeros(768)  # Most BERT-based embeddings are 768-dimensional
        
        # Determine which model to use if 'default'
        if model == 'default':
            if self.use_kosimcse:
                model = 'kosimcse'
            elif self.use_bge_korean:
                model = 'bge_korean'
            elif self.use_klue_roberta:
                model = 'klue_roberta'
            elif self.use_kobert:
                model = 'kobert'
            elif self.use_kpf_bert:
                model = 'kpf_bert'
            elif self.use_koelectra:
                model = 'koelectra'
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
            if model == 'kobert' and self.use_kobert:
                embedding = self._get_kobert_embedding(text)
            elif model == 'klue_roberta' and self.use_klue_roberta:
                embedding = self._get_klue_roberta_embedding(text)
            elif model == 'kosimcse' and self.use_kosimcse:
                embedding = self._get_kosimcse_embedding(text)
            elif model == 'bge_korean' and self.use_bge_korean:
                embedding = self._get_bge_korean_embedding(text)
            elif model == 'kpf_bert' and self.use_kpf_bert:
                embedding = self._get_kpf_bert_embedding(text)
            elif model == 'koelectra' and self.use_koelectra:
                embedding = self._get_koelectra_embedding(text)
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
            if model == 'bge_korean':
                return np.zeros(1024)
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
        # Tokenize text
        inputs = self.kobert_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.kobert_model(**inputs)
            
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def _get_klue_roberta_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KLUE-RoBERTa model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize text
        inputs = self.klue_roberta_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.klue_roberta_model(**inputs)
            
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
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
    
    def _get_kpf_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KPF-BERT model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize text
        inputs = self.kpf_bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.kpf_bert_model(**inputs)
            
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    
    def _get_koelectra_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KoELECTRA model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize text
        inputs = self.koelectra_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Move to same device as model
        if self.device == 'cuda' and torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.koelectra_model(**inputs)
            
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
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
    
    def get_embeddings_batch(self, texts: List[str], model: str = 'default', 
                           batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use for embeddings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings (n_samples, n_features)
        """
        if not texts:
            return np.array([])
        
        # Determine embedding dimension
        if model == 'bge_korean':
            embedding_dim = 1024
        else:
            embedding_dim = 768
        
        # Initialize array to store embeddings
        embeddings = np.zeros((len(texts), embedding_dim))
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Embedding with {model}")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Get embeddings for batch
            for j, text in enumerate(batch_texts):
                embeddings[i+j] = self.get_embedding(text, model)
        
        # Save cache
        if self.cache_embeddings:
            self._save_cache()
        
        return embeddings
    
    def get_ensemble_embeddings(self, texts: List[str], 
                              models: List[str] = None,
                              weights: List[float] = None,
                              batch_size: int = 32,
                              show_progress: bool = True) -> np.ndarray:
        """
        Get ensemble embeddings using multiple models.
        
        Args:
            texts: List of texts to embed
            models: List of models to use
            weights: List of weights for each model
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of ensemble embeddings (n_samples, n_features)
        """
        if not texts:
            return np.array([])
        
        # Default to all available models
        if models is None:
            models = []
            if self.use_kobert:
                models.append('kobert')
            if self.use_klue_roberta:
                models.append('klue_roberta')
            if self.use_kosimcse:
                models.append('kosimcse')
            if self.use_bge_korean:
                models.append('bge_korean')
            if self.use_kpf_bert:
                models.append('kpf_bert')
            if self.use_koelectra:
                models.append('koelectra')
            
            if not models:
                # Fallback to simple embedding
                return self.get_embeddings_batch(texts, 'default', batch_size, show_progress)
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Ensure weights sum to 1
        weights = [w / sum(weights) for w in weights]
        
        # Determine embedding dimension
        if 'bge_korean' in models:
            embedding_dim = 1024
        else:
            embedding_dim = 768
        
        # Initialize array to store ensemble embeddings
        ensemble_embeddings = np.zeros((len(texts), embedding_dim))
        
        # Get embeddings for each model
        for model, weight in zip(models, weights):
            logger.info(f"Getting embeddings with model {model} (weight: {weight:.2f})")
            
            # Get embeddings for current model
            model_embeddings = self.get_embeddings_batch(texts, model, batch_size, show_progress)
            
            # Normalize embeddings
            norms = np.linalg.norm(model_embeddings, axis=1, keepdims=True)
            normalized_embeddings = model_embeddings / np.maximum(norms, 1e-10)
            
            # Add to ensemble with weight
            ensemble_embeddings += normalized_embeddings * weight
        
        # Normalize ensemble embeddings
        norms = np.linalg.norm(ensemble_embeddings, axis=1, keepdims=True)
        ensemble_embeddings = ensemble_embeddings / np.maximum(norms, 1e-10)
        
        return ensemble_embeddings
    
    def vectorize_dataframe(self, df: pd.DataFrame, 
                          text_col: str = 'Title',
                          content_col: str = 'Body',
                          use_content: bool = True,
                          model: str = 'default',
                          batch_size: int = 32) -> np.ndarray:
        """
        Vectorize texts in a DataFrame.
        
        Args:
            df: DataFrame containing texts
            text_col: Column name for primary text (e.g., title)
            content_col: Column name for secondary text (e.g., body)
            use_content: Whether to use content for vectorization
            model: Model to use for embeddings
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings (n_samples, n_features)
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for vectorization")
            return np.array([])
            
        logger.info(f"Vectorizing {len(df)} texts with model {model}")
        
        try:
            # Prepare texts for vectorization
            texts = []
            
            # Process each row
            for idx, row in df.iterrows():
                # Determine text to vectorize
                if text_col in row and isinstance(row[text_col], str):
                    text = row[text_col]
                    
                    # Add content if specified
                    if use_content and content_col in row and isinstance(row[content_col], str):
                        text += " " + row[content_col]
                    
                    texts.append(text)
                else:
                    # Skip if no text
                    texts.append("")
            
            # Get embeddings
            embeddings = self.get_embeddings_batch(texts, model, batch_size)
            
            logger.info(f"Vectorized {len(texts)} texts with model {model}")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error vectorizing DataFrame: {str(e)}")
            return np.array([])
    
    def vectorize_dataframe_ensemble(self, df: pd.DataFrame,
                                   text_col: str = 'Title',
                                   content_col: str = 'Body',
                                   use_content: bool = True,
                                   models: List[str] = None,
                                   weights: List[float] = None,
                                   batch_size: int = 32) -> np.ndarray:
        """
        Vectorize texts in a DataFrame using ensemble of models.
        
        Args:
            df: DataFrame containing texts
            text_col: Column name for primary text (e.g., title)
            content_col: Column name for secondary text (e.g., body)
            use_content: Whether to use content for vectorization
            models: List of models to use
            weights: List of weights for each model
            batch_size: Batch size for processing
            
        Returns:
            Array of ensemble embeddings (n_samples, n_features)
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for vectorization")
            return np.array([])
            
        logger.info(f"Vectorizing {len(df)} texts with ensemble of models")
        
        try:
            # Prepare texts for vectorization
            texts = []
            
            # Process each row
            for idx, row in df.iterrows():
                # Determine text to vectorize
                if text_col in row and isinstance(row[text_col], str):
                    text = row[text_col]
                    
                    # Add content if specified
                    if use_content and content_col in row and isinstance(row[content_col], str):
                        text += " " + row[content_col]
                    
                    texts.append(text)
                else:
                    # Skip if no text
                    texts.append("")
            
            # Get ensemble embeddings
            embeddings = self.get_ensemble_embeddings(texts, models, weights, batch_size)
            
            logger.info(f"Vectorized {len(texts)} texts with ensemble of models")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error vectorizing DataFrame with ensemble: {str(e)}")
            return np.array([])
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        models = []
        
        if self.use_kobert:
            models.append('kobert')
        
        if self.use_klue_roberta:
            models.append('klue_roberta')
        
        if self.use_kosimcse:
            models.append('kosimcse')
        
        if self.use_bge_korean:
            models.append('bge_korean')
        
        if self.use_kpf_bert:
            models.append('kpf_bert')
        
        if self.use_koelectra:
            models.append('koelectra')
        
        return models
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models.
        
        Returns:
            Dictionary with model information
        """
        model_info = {}
        
        if self.use_kobert:
            model_info['kobert'] = {
                'name': 'KoBERT',
                'source': 'monologg/kobert',
                'embedding_dim': 768,
                'description': 'Korean BERT model pre-trained on Korean text corpus',
                'available': True
            }
        
        if self.use_klue_roberta:
            model_info['klue_roberta'] = {
                'name': 'KLUE-RoBERTa',
                'source': 'klue/roberta-base',
                'embedding_dim': 768,
                'description': 'Korean RoBERTa model pre-trained on Korean text corpus for KLUE benchmark',
                'available': True
            }
        
        if self.use_kosimcse:
            model_info['kosimcse'] = {
                'name': 'KoSimCSE-roberta',
                'source': 'BM-K/KoSimCSE-roberta',
                'embedding_dim': 768,
                'description': 'Korean sentence embedding model based on SimCSE approach',
                'available': True
            }
        
        if self.use_bge_korean:
            model_info['bge_korean'] = {
                'name': 'BGE-M3-Korean',
                'source': 'upskyy/bge-m3-korean',
                'embedding_dim': 1024,
                'description': 'Korean fine-tuned version of BGE-M3 for semantic textual similarity',
                'available': True
            }
        
        if self.use_kpf_bert:
            model_info['kpf_bert'] = {
                'name': 'KPF-BERT',
                'source': 'kpfbert/kpfbert',
                'embedding_dim': 768,
                'description': 'BERT model pre-trained on Korean news articles from Korea Press Foundation',
                'available': True
            }
        
        if self.use_koelectra:
            model_info['koelectra'] = {
                'name': 'KoELECTRA',
                'source': 'monologg/koelectra-base-v3-discriminator',
                'embedding_dim': 768,
                'description': 'Korean ELECTRA model pre-trained on Korean text corpus',
                'available': True
            }
        
        return model_info
