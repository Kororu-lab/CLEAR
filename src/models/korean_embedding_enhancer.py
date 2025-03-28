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
                 use_kobert: bool = True,
                 use_klue_roberta: bool = True,
                 use_kosimcse: bool = False,
                 use_bge_korean: bool = False,
                 cache_embeddings: bool = True,
                 models_dir: str = None):
        """
        Initialize the Korean embedding enhancer.
        
        Args:
            use_kobert: Whether to use KoBERT for embeddings
            use_klue_roberta: Whether to use KLUE-RoBERTa for embeddings
            use_kosimcse: Whether to use KoSimCSE-roberta for embeddings
            use_bge_korean: Whether to use bge-m3-korean for embeddings
            cache_embeddings: Whether to cache embeddings
            models_dir: Directory to save/load models and cache
        """
        self.use_kobert = use_kobert
        self.use_klue_roberta = use_klue_roberta
        self.use_kosimcse = use_kosimcse
        self.use_bge_korean = use_bge_korean
        self.cache_embeddings = cache_embeddings
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "embeddings")
        
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
        
        # Load cache if available
        if self.cache_embeddings:
            self._load_cache()
        
        logger.info(f"Initialized KoreanEmbeddingEnhancer with models: " +
                   f"kobert={self.use_kobert}, klue_roberta={self.use_klue_roberta}, " +
                   f"kosimcse={self.use_kosimcse}, bge_korean={self.use_bge_korean}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        available_models = []
        if self.use_kobert and self.kobert_model is not None:
            available_models.append('kobert')
        if self.use_klue_roberta and self.klue_roberta_model is not None:
            available_models.append('klue_roberta')
        if self.use_kosimcse and self.kosimcse_model is not None:
            available_models.append('kosimcse')
        if self.use_bge_korean and self.bge_korean_model is not None:
            available_models.append('bge_korean')
        
        # Add default as an option if any model is available
        if available_models:
            available_models.append('default')
        else:
            available_models.append('simple')
        
        return available_models
    
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
            
            # Move to CPU (for demo purposes)
            self.kobert_model = self.kobert_model.to('cpu')
            
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
            
            # Move to CPU (for demo purposes)
            self.klue_roberta_model = self.klue_roberta_model.to('cpu')
            
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
        Get embedding for a text using the specified model.
        
        Args:
            text: Text to embed
            model: Model to use ('kobert', 'klue_roberta', 'kosimcse', 'bge_korean', or 'default')
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        cache_key = f"{model}:{text}"
        if self.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Get embedding based on model
        if model == 'kobert' and self.use_kobert and self.kobert_model is not None:
            embedding = self._get_kobert_embedding(text)
        elif model == 'klue_roberta' and self.use_klue_roberta and self.klue_roberta_model is not None:
            embedding = self._get_klue_roberta_embedding(text)
        elif model == 'kosimcse' and self.use_kosimcse and self.kosimcse_model is not None:
            embedding = self._get_kosimcse_embedding(text)
        elif model == 'bge_korean' and self.use_bge_korean and self.bge_korean_model is not None:
            embedding = self._get_bge_korean_embedding(text)
        elif model == 'default':
            # Use first available model
            if self.use_kobert and self.kobert_model is not None:
                embedding = self._get_kobert_embedding(text)
            elif self.use_klue_roberta and self.klue_roberta_model is not None:
                embedding = self._get_klue_roberta_embedding(text)
            elif self.use_kosimcse and self.kosimcse_model is not None:
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
        if self.cache_embeddings:
            self.embedding_cache[cache_key] = embedding
            # Periodically save cache
            if len(self.embedding_cache) % 100 == 0:
                self._save_cache()
        
        return embedding
    
    def _get_kobert_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KoBERT model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            import torch
            
            # Tokenize text
            inputs = self.kobert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.kobert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Convert to numpy array
            embedding = embeddings.cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            logger.error(f"Error in KoBERT embedding: {str(e)}")
            # Fall back to simple embedding
            return self._get_simple_embedding(text)
    
    def _get_klue_roberta_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding using KLUE-RoBERTa model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            import torch
            
            # Tokenize text
            inputs = self.klue_roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.klue_roberta_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Convert to numpy array
            embedding = embeddings.cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            logger.error(f"Error in KLUE-RoBERTa embedding: {str(e)}")
            # Fall back to simple embedding
            return self._get_simple_embedding(text)
    
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
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.kosimcse_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Convert to numpy array
            embedding = embeddings.cpu().numpy()[0]
            
            return embedding
        except Exception as e:
            logger.error(f"Error in KoSimCSE-roberta embedding: {str(e)}")
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
        embedding = np.zeros(768)  # Use same dimension as BERT models
        
        for i, char in enumerate(chars[:768]):
            embedding[i] = text.count(char) / len(text)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str], model: str = 'default', 
                           batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use ('kobert', 'klue_roberta', 'kosimcse', 'bge_korean', or 'default')
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        n = len(texts)
        
        # Get embedding dimension from a sample text
        sample_embedding = self.get_embedding(texts[0] if texts else "", model)
        embedding_dim = sample_embedding.shape[0]
        
        # Initialize embeddings array
        embeddings = np.zeros((n, embedding_dim))
        
        # Process in batches
        iterator = range(0, n, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Getting {model} embeddings")
        
        for i in iterator:
            batch_texts = texts[i:min(i+batch_size, n)]
            batch_size_actual = len(batch_texts)
            
            # Process each text in the batch
            for j, text in enumerate(batch_texts):
                embeddings[i+j] = self.get_embedding(text, model)
        
        return embeddings
    
    def get_ensemble_embeddings(self, texts: List[str], weights: Dict[str, float] = None, 
                              batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Get ensemble embeddings using multiple models with weights.
        
        Args:
            texts: List of texts to embed
            weights: Dictionary mapping model names to weights
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of ensemble embeddings
        """
        if weights is None:
            # Default weights
            weights = {}
            if self.use_kobert:
                weights['kobert'] = 0.5
            if self.use_klue_roberta:
                weights['klue_roberta'] = 0.5
            if self.use_kosimcse:
                weights['kosimcse'] = 0.0
            if self.use_bge_korean:
                weights['bge_korean'] = 0.0
        
        # Filter out models with zero weight or not available
        weights = {k: v for k, v in weights.items() if v > 0}
        
        # If no weights, use default model
        if not weights:
            return self.get_embeddings_batch(texts, 'default', batch_size, show_progress)
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Get embeddings for each model
        model_embeddings = {}
        for model in normalized_weights.keys():
            model_embeddings[model] = self.get_embeddings_batch(texts, model, batch_size, show_progress)
        
        # Get embedding dimension from the first model
        first_model = next(iter(model_embeddings.keys()))
        embedding_dim = model_embeddings[first_model].shape[1]
        
        # Initialize ensemble embeddings
        ensemble_embeddings = np.zeros((len(texts), embedding_dim))
        
        # Add weighted embeddings
        for model, weight in normalized_weights.items():
            ensemble_embeddings += model_embeddings[model] * weight
        
        # Normalize embeddings
        for i in range(len(ensemble_embeddings)):
            norm = np.linalg.norm(ensemble_embeddings[i])
            if norm > 0:
                ensemble_embeddings[i] = ensemble_embeddings[i] / norm
        
        return ensemble_embeddings
    
    def compare_models(self, texts: List[str], reference_text: str, models: List[str] = None, 
                     batch_size: int = 32) -> Dict[str, List[float]]:
        """
        Compare similarity scores from different models.
        
        Args:
            texts: List of texts to compare
            reference_text: Reference text to compare against
            models: List of models to use
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping model names to lists of similarity scores
        """
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
        
        # Get reference embedding for each model
        reference_embeddings = {}
        for model in models:
            reference_embeddings[model] = self.get_embedding(reference_text, model)
        
        # Get embeddings for each model
        model_embeddings = {}
        for model in models:
            embeddings = self.get_embeddings_batch(texts, model, batch_size)
            model_embeddings[model] = embeddings
        
        # Calculate similarity scores
        similarity_scores = {}
        for model in models:
            ref_embedding = reference_embeddings[model]
            embeddings = model_embeddings[model]
            
            scores = []
            for embedding in embeddings:
                # Calculate cosine similarity
                dot_product = np.dot(ref_embedding, embedding)
                norm1 = np.linalg.norm(ref_embedding)
                norm2 = np.linalg.norm(embedding)
                
                if norm1 == 0 or norm2 == 0:
                    scores.append(0.0)
                else:
                    similarity = dot_product / (norm1 * norm2)
                    scores.append(max(0.0, min(1.0, similarity)))
            
            similarity_scores[model] = scores
        
        return similarity_scores
    
    def visualize_model_comparison(self, texts: List[str], reference_text: str, models: List[str] = None, 
                                 batch_size: int = 32, title: str = "Model Comparison"):
        """
        Visualize comparison of similarity scores from different models.
        
        Args:
            texts: List of texts to compare
            reference_text: Reference text to compare against
            models: List of models to use
            batch_size: Batch size for processing
            title: Plot title
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get similarity scores
        similarity_scores = self.compare_models(texts, reference_text, models, batch_size)
        
        # Prepare data for plotting
        data = []
        for model, scores in similarity_scores.items():
            for i, score in enumerate(scores):
                data.append({
                    'Model': model,
                    'Text Index': i,
                    'Similarity': score
                })
        
        df = pd.DataFrame(data)
        
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Text Index', y='Similarity', hue='Model', data=df)
        plt.title(title)
        plt.xlabel('Text Index')
        plt.ylabel('Similarity Score')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()
