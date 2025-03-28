"""
News vectorizer module for converting news articles into vector representations.
Based on NAVER's TF-IDF approach for news clustering with additional embedding options.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import torch
from gensim.models import Word2Vec, FastText
import nltk
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsVectorizer:
    """
    Vectorizer for converting news articles into vector representations.
    Supports multiple embedding methods with configurable parameters.
    """
    
    def __init__(self, 
                 method: str = 'tfidf',
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 embedding_dim: int = 300,
                 use_gpu: bool = True,
                 title_weight: float = 2.0,
                 models_dir: str = None,
                 korean_support: bool = True):
        """
        Initialize the news vectorizer.
        
        Args:
            method: Vectorization method ('tfidf', 'word2vec', 'fasttext', 'openai', 'kobert')
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider
            embedding_dim: Dimension of embeddings for neural methods
            use_gpu: Whether to use GPU for neural methods
            title_weight: Weight multiplier for title vs. content
            models_dir: Directory to save/load models
            korean_support: Whether to enable specific Korean language support
        """
        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.embedding_dim = embedding_dim
        self.title_weight = title_weight
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "vectorizers")
        self.korean_support = korean_support
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check GPU availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if use_gpu and not torch.cuda.is_available():
            logger.warning("GPU requested but not available, falling back to CPU")
        
        # Initialize vectorizer based on method
        self._initialize_vectorizer()
        
        logger.info(f"Initialized NewsVectorizer with method={method}, max_features={max_features}")
    
    def _initialize_vectorizer(self) -> None:
        """
        Initialize the appropriate vectorizer based on the selected method.
        """
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
            )
            self.svd = None
            logger.info(f"Initialized TF-IDF vectorizer with max_features={self.max_features}")
            
        elif self.method == 'word2vec':
            self.vectorizer = None  # Will be trained on fit
            logger.info(f"Word2Vec vectorization will be initialized during training")
            
        elif self.method == 'fasttext':
            self.vectorizer = None  # Will be trained on fit
            logger.info(f"FastText vectorization will be initialized during training")
            
        elif self.method == 'openai':
            try:
                import openai
                self.vectorizer = "openai"
                logger.info("OpenAI API vectorization initialized")
            except ImportError:
                logger.error("OpenAI package not installed, falling back to TF-IDF")
                self.method = 'tfidf'
                self._initialize_vectorizer()
        
        elif self.method == 'kobert':
            try:
                # We'll check if transformers is available but won't load the model yet
                # to avoid disk space issues
                import importlib.util
                if importlib.util.find_spec("transformers") is not None:
                    self.vectorizer = "kobert"
                    logger.info("KoBERT vectorization initialized (model will be loaded on demand)")
                else:
                    logger.error("Transformers package not installed, falling back to TF-IDF")
                    self.method = 'tfidf'
                    self._initialize_vectorizer()
            except Exception as e:
                logger.error(f"Error initializing KoBERT: {str(e)}, falling back to TF-IDF")
                self.method = 'tfidf'
                self._initialize_vectorizer()
                
        else:
            logger.warning(f"Unknown method: {self.method}, falling back to TF-IDF")
            self.method = 'tfidf'
            self._initialize_vectorizer()
    
    def fit(self, texts: List[str], tokenized_texts: List[List[str]] = None) -> 'NewsVectorizer':
        """
        Fit the vectorizer on a corpus of texts.
        
        Args:
            texts: List of preprocessed text documents
            tokenized_texts: List of tokenized texts (required for Word2Vec/FastText)
            
        Returns:
            Self for method chaining
        """
        if not texts:
            logger.warning("Empty text list provided for fitting")
            return self
            
        logger.info(f"Fitting {self.method} vectorizer on {len(texts)} documents")
        
        try:
            if self.method == 'tfidf':
                self.vectorizer.fit(texts)
                logger.info(f"TF-IDF vectorizer fitted with vocabulary size: {len(self.vectorizer.vocabulary_)}")
                
            elif self.method in ['word2vec', 'fasttext']:
                if not tokenized_texts:
                    logger.warning(f"{self.method} requires tokenized texts, attempting to tokenize")
                    tokenized_texts = [nltk.word_tokenize(text) for text in texts]
                
                if self.method == 'word2vec':
                    self.vectorizer = Word2Vec(
                        sentences=tokenized_texts,
                        vector_size=self.embedding_dim,
                        window=5,
                        min_count=1,
                        workers=4,
                        sg=1  # Skip-gram model
                    )
                    logger.info(f"Word2Vec model fitted with vocabulary size: {len(self.vectorizer.wv.key_to_index)}")
                    
                else:  # FastText
                    self.vectorizer = FastText(
                        sentences=tokenized_texts,
                        vector_size=self.embedding_dim,
                        window=5,
                        min_count=1,
                        workers=4,
                        sg=1  # Skip-gram model
                    )
                    logger.info(f"FastText model fitted with vocabulary size: {len(self.vectorizer.wv.key_to_index)}")
            
            # Save the fitted vectorizer
            self._save_model()
            
            return self
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {str(e)}")
            raise
    
    def transform(self, texts: List[str], tokenized_texts: List[List[str]] = None) -> np.ndarray:
        """
        Transform texts into vector representations.
        
        Args:
            texts: List of preprocessed text documents
            tokenized_texts: List of tokenized texts (required for Word2Vec/FastText)
            
        Returns:
            Array of vector representations
        """
        if not texts:
            logger.warning("Empty text list provided for transformation")
            return np.array([])
            
        logger.info(f"Transforming {len(texts)} documents using {self.method}")
        
        try:
            if self.method == 'tfidf':
                vectors = self.vectorizer.transform(texts)
                logger.info(f"Transformed {len(texts)} documents to shape {vectors.shape}")
                return vectors
                
            elif self.method in ['word2vec', 'fasttext']:
                if not tokenized_texts:
                    logger.warning(f"{self.method} requires tokenized texts, attempting to tokenize")
                    tokenized_texts = [nltk.word_tokenize(text) for text in texts]
                
                # Create document vectors by averaging word vectors
                doc_vectors = []
                for tokens in tokenized_texts:
                    token_vectors = []
                    for token in tokens:
                        try:
                            token_vectors.append(self.vectorizer.wv[token])
                        except KeyError:
                            # Skip tokens not in vocabulary
                            continue
                    
                    if token_vectors:
                        # Average the token vectors
                        doc_vector = np.mean(token_vectors, axis=0)
                    else:
                        # Use zeros for empty documents
                        doc_vector = np.zeros(self.embedding_dim)
                    
                    doc_vectors.append(doc_vector)
                
                vectors = np.array(doc_vectors)
                logger.info(f"Transformed {len(texts)} documents to shape {vectors.shape}")
                return vectors
                
            elif self.method == 'openai':
                try:
                    import openai
                    # Check if API key is set
                    if not openai.api_key:
                        logger.warning("OpenAI API key not set, returning placeholder vectors")
                        return np.zeros((len(texts), self.embedding_dim))
                    
                    # Get embeddings in batches to avoid API limits
                    batch_size = 100
                    all_embeddings = []
                    
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        response = openai.Embedding.create(
                            input=batch_texts,
                            model="text-embedding-ada-002"
                        )
                        batch_embeddings = [item["embedding"] for item in response["data"]]
                        all_embeddings.extend(batch_embeddings)
                    
                    vectors = np.array(all_embeddings)
                    logger.info(f"Transformed {len(texts)} documents using OpenAI API to shape {vectors.shape}")
                    return vectors
                except Exception as e:
                    logger.error(f"Error using OpenAI API: {str(e)}, returning placeholder")
                    return np.zeros((len(texts), self.embedding_dim))
            
            elif self.method == 'kobert':
                try:
                    # Lazy load KoBERT to save memory
                    from transformers import AutoTokenizer, AutoModel
                    import torch
                    
                    # Load KoBERT tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
                    model = AutoModel.from_pretrained("monologg/kobert")
                    
                    if self.use_gpu and torch.cuda.is_available():
                        model = model.cuda()
                    
                    # Process in batches to avoid memory issues
                    batch_size = 8
                    all_embeddings = []
                    
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        
                        # Tokenize
                        encoded_input = tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt'
                        )
                        
                        # Move to GPU if available
                        if self.use_gpu and torch.cuda.is_available():
                            encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                        
                        # Get embeddings
                        with torch.no_grad():
                            model_output = model(**encoded_input)
                            
                        # Use [CLS] token embedding as document representation
                        sentence_embeddings = model_output[0][:, 0, :].cpu().numpy()
                        all_embeddings.extend(sentence_embeddings)
                    
                    vectors = np.array(all_embeddings)
                    logger.info(f"Transformed {len(texts)} documents using KoBERT to shape {vectors.shape}")
                    return vectors
                    
                except Exception as e:
                    logger.error(f"Error using KoBERT: {str(e)}, falling back to TF-IDF")
                    # Temporarily switch to TF-IDF for this transformation
                    temp_vectorizer = TfidfVectorizer(max_features=self.max_features)
                    temp_vectorizer.fit(texts)
                    vectors = temp_vectorizer.transform(texts)
                    return vectors
                
            else:
                logger.error(f"Unknown method: {self.method}")
                return np.array([])
                
        except Exception as e:
            logger.error(f"Error transforming texts: {str(e)}")
            raise
    
    def fit_transform(self, texts: List[str], tokenized_texts: List[List[str]] = None) -> np.ndarray:
        """
        Fit the vectorizer and transform texts in one step.
        
        Args:
            texts: List of preprocessed text documents
            tokenized_texts: List of tokenized texts (required for Word2Vec/FastText)
            
        Returns:
            Array of vector representations
        """
        self.fit(texts, tokenized_texts)
        return self.transform(texts, tokenized_texts)
    
    def reduce_dimensions(self, vectors: np.ndarray, n_components: int = 100) -> np.ndarray:
        """
        Reduce dimensionality of vectors using SVD.
        
        Args:
            vectors: Vectors to reduce
            n_components: Number of components to keep
            
        Returns:
            Reduced vectors
        """
        if vectors.shape[0] == 0:
            logger.warning("Empty vector array provided for dimension reduction")
            return np.array([])
            
        logger.info(f"Reducing dimensions from {vectors.shape[1]} to {n_components}")
        
        try:
            if self.svd is None:
                self.svd = TruncatedSVD(n_components=n_components, random_state=42)
                reduced_vectors = self.svd.fit_transform(vectors)
                
                # Save the SVD model
                svd_path = os.path.join(self.models_dir, f"{self.method}_svd_model.joblib")
                joblib.dump(self.svd, svd_path)
                logger.info(f"Saved SVD model to {svd_path}")
            else:
                reduced_vectors = self.svd.transform(vectors)
                
            logger.info(f"Reduced vectors to shape {reduced_vectors.shape}")
            return reduced_vectors
        except Exception as e:
            logger.error(f"Error reducing dimensions: {str(e)}")
            raise
    
    def vectorize_articles(self, articles_df: pd.DataFrame, 
                          content_col: str = 'processed_content',
                          title_col: str = 'processed_title',
                          tokenized_content_col: str = None,
                          tokenized_title_col: str = None,
                          combine_title_content: bool = True,
                          reduce_dims: bool = False,
                          n_components: int = 100,
                          title_content_ratio: float = None) -> pd.DataFrame:
        """
        Vectorize a DataFrame of news articles.
        
        Args:
            articles_df: DataFrame containing news articles
            content_col: Column name for processed content
            title_col: Column name for processed title
            tokenized_content_col: Column name for tokenized content (for Word2Vec/FastText)
            tokenized_title_col: Column name for tokenized title (for Word2Vec/FastText)
            combine_title_content: Whether to combine title and content with title weighting
            reduce_dims: Whether to reduce dimensions
            n_components: Number of components if reducing dimensions
            title_content_ratio: Optional ratio to weight title vs content (overrides title_weight)
            
        Returns:
            DataFrame with added vector representations
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for vectorization")
            return articles_df
            
        logger.info(f"Vectorizing {len(articles_df)} articles using {self.method}")
        
        # Create a copy to avoid modifying the original
        df = articles_df.copy()
        
        try:
            # Use provided title_content_ratio if specified
            title_weight = title_content_ratio if title_content_ratio is not None else self.title_weight
            
            # Prepare texts for vectorization
            if combine_title_content and title_col in df.columns and content_col in df.columns:
                # Combine title and content with title having more weight
                combined_texts = []
                title_only_texts = []
                content_only_texts = []
                
                for _, row in df.iterrows():
                    title = row.get(title_col, "")
                    content = row.get(content_col, "")
                    
                    # Store separate title and content for potential separate vectorization
                    title_only_texts.append(title)
                    content_only_texts.append(content)
                    
                    # Weight title by repeating it
                    title_part = " ".join([title] * int(title_weight))
                    combined_text = f"{title_part} {content}"
                    combined_texts.append(combined_text)
                
                texts_to_vectorize = combined_texts
                logger.info(f"Combined {title_col} and {content_col} with title weight {title_weight}")
                
                # Store separate vectors for title and content if requested
                if title_content_ratio is not None:
                    # Vectorize title and content separately
                    title_vectors = self.transform(title_only_texts)
                    content_vectors = self.transform(content_only_texts)
                    
                    # Store separate vectors
                    if self.method == 'tfidf':
                        df['title_vector'] = list(title_vectors.toarray())
                        df['content_vector'] = list(content_vectors.toarray())
                    else:
                        df['title_vector'] = list(title_vectors)
                        df['content_vector'] = list(content_vectors)
                    
                    logger.info(f"Created separate vectors for title and content")
            else:
                # Use only content
                if content_col in df.columns:
                    texts_to_vectorize = df[content_col].tolist()
                    logger.info(f"Using only {content_col} for vectorization")
                else:
                    logger.error(f"Content column '{content_col}' not found in DataFrame")
                    return df
            
            # Prepare tokenized texts if needed
            tokenized_texts = None
            if self.method in ['word2vec', 'fasttext']:
                if tokenized_content_col in df.columns:
                    tokenized_texts = df[tokenized_content_col].tolist()
                else:
                    # Tokenize texts
                    tokenized_texts = [nltk.word_tokenize(text) for text in texts_to_vectorize]
            
            # Fit and transform
            vectors = self.fit_transform(texts_to_vectorize, tokenized_texts)
            
            # Reduce dimensions if requested
            if reduce_dims:
                vectors = self.reduce_dimensions(vectors, n_components)
                
                # Store as numpy arrays in DataFrame
                df['vector'] = list(vectors)
            else:
                # For sparse matrices, convert to dense for storage
                if self.method == 'tfidf':
                    df['vector'] = list(vectors.toarray())
                else:
                    df['vector'] = list(vectors)
                
            logger.info(f"Vectorization complete for {len(df)} articles")
            return df
        except Exception as e:
            logger.error(f"Error vectorizing articles: {str(e)}")
            raise
    
    def get_top_terms(self, n: int = 20) -> Dict[str, float]:
        """
        Get the top terms by importance value.
        
        Args:
            n: Number of top terms to return
            
        Returns:
            Dictionary mapping terms to their importance values
        """
        if self.method == 'tfidf':
            if not hasattr(self.vectorizer, 'vocabulary_'):
                logger.warning("TF-IDF vectorizer not fitted yet")
                return {}
                
            try:
                # Get feature names and IDF values
                feature_names = self.vectorizer.get_feature_names_out()
                idf_values = self.vectorizer.idf_
                
                # Create a dictionary of term -> IDF value
                term_idf = {feature_names[i]: idf_values[i] for i in range(len(feature_names))}
                
                # Sort by IDF value (descending) and take top n
                top_terms = dict(sorted(term_idf.items(), key=lambda x: x[1], reverse=True)[:n])
                
                return top_terms
            except Exception as e:
                logger.error(f"Error getting top terms: {str(e)}")
                return {}
        else:
            logger.warning(f"get_top_terms not implemented for {self.method}")
            return {}
    
    def _save_model(self) -> None:
        """
        Save the vectorizer model to disk.
        """
        try:
            if self.method == 'tfidf':
                model_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
                joblib.dump(self.vectorizer, model_path)
                logger.info(f"Saved TF-IDF vectorizer to {model_path}")
                
            elif self.method in ['word2vec', 'fasttext']:
                model_path = os.path.join(self.models_dir, f"{self.method}_model.bin")
                self.vectorizer.save(model_path)
                logger.info(f"Saved {self.method} model to {model_path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load a saved vectorizer model.
        
        Args:
            model_path: Path to the model file (if None, uses default path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path is None:
                if self.method == 'tfidf':
                    model_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
                elif self.method in ['word2vec', 'fasttext']:
                    model_path = os.path.join(self.models_dir, f"{self.method}_model.bin")
                else:
                    logger.error(f"No default path for method: {self.method}")
                    return False
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            if self.method == 'tfidf':
                self.vectorizer = joblib.load(model_path)
                logger.info(f"Loaded TF-IDF vectorizer from {model_path}")
                
            elif self.method == 'word2vec':
                from gensim.models import Word2Vec
                self.vectorizer = Word2Vec.load(model_path)
                logger.info(f"Loaded Word2Vec model from {model_path}")
                
            elif self.method == 'fasttext':
                from gensim.models import FastText
                self.vectorizer = FastText.load(model_path)
                logger.info(f"Loaded FastText model from {model_path}")
                
            # Load SVD model if it exists
            svd_path = os.path.join(self.models_dir, f"{self.method}_svd_model.joblib")
            if os.path.exists(svd_path):
                self.svd = joblib.load(svd_path)
                logger.info(f"Loaded SVD model from {svd_path}")
                
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Convert to numpy arrays if not already
            if isinstance(vec1, list):
                vec1 = np.array(vec1)
            if isinstance(vec2, list):
                vec2 = np.array(vec2)
                
            # Calculate cosine similarity
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def calculate_similarity_matrix(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Calculate pairwise cosine similarity matrix for a list of vectors.
        
        Args:
            vectors: List of vectors
            
        Returns:
            Similarity matrix
        """
        try:
            n = len(vectors)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i, n):
                    sim = self.calculate_similarity(vectors[i], vectors[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric
                    
            return similarity_matrix
        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {str(e)}")
            return np.array([])
