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
                 models_dir: str = None):
        """
        Initialize the news vectorizer.
        
        Args:
            method: Vectorization method ('tfidf', 'word2vec', 'fasttext', 'openai')
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider
            embedding_dim: Dimension of embeddings for neural methods
            use_gpu: Whether to use GPU for neural methods
            title_weight: Weight multiplier for title vs. content
            models_dir: Directory to save/load models
        """
        self.method = method.lower()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.embedding_dim = embedding_dim
        self.title_weight = title_weight
        self.models_dir = models_dir or os.path.join(os.getcwd(), "models", "vectorizers")
        
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
                # This would use the OpenAI API to get embeddings
                # For now, we'll just return a placeholder
                logger.warning("OpenAI embeddings not implemented, returning placeholder")
                return np.zeros((len(texts), self.embedding_dim))
                
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
                          n_components: int = 100) -> pd.DataFrame:
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
            # Prepare texts for vectorization
            if combine_title_content and title_col in df.columns and content_col in df.columns:
                # Combine title and content with title having more weight
                combined_texts = []
                
                for _, row in df.iterrows():
                    title = row.get(title_col, "")
                    content = row.get(content_col, "")
                    
                    # Weight title by repeating it
                    title_part = " ".join([title] * int(self.title_weight))
                    combined_text = f"{title_part} {content}"
                    combined_texts.append(combined_text)
                
                texts_to_vectorize = combined_texts
                logger.info(f"Combined {title_col} and {content_col} with title weight {self.title_weight}")
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
        
        elif self.method in ['word2vec', 'fasttext']:
            if not self.vectorizer:
                logger.warning(f"{self.method} model not fitted yet")
                return {}
                
            try:
                # Get most frequent words and their vectors
                vocab = self.vectorizer.wv.key_to_index
                top_words = sorted(vocab.items(), key=lambda x: x[1])[:n]
                
                # Create a dictionary of word -> vector norm (as importance)
                top_terms = {}
                for word, idx in top_words:
                    vector = self.vectorizer.wv[word]
                    importance = np.linalg.norm(vector)
                    top_terms[word] = importance
                
                return top_terms
            except Exception as e:
                logger.error(f"Error getting top terms: {str(e)}")
                return {}
        
        else:
            logger.warning(f"get_top_terms not implemented for {self.method}")
            return {}
    
    def _save_model(self) -> None:
        """
        Save the fitted vectorizer model.
        """
        try:
            model_path = os.path.join(self.models_dir, f"{self.method}_vectorizer.joblib")
            
            if self.method == 'tfidf':
                joblib.dump(self.vectorizer, model_path)
            elif self.method in ['word2vec', 'fasttext']:
                self.vectorizer.save(model_path)
            
            logger.info(f"Saved {self.method} vectorizer model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving vectorizer model: {str(e)}")
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Load a previously saved vectorizer model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, f"{self.method}_vectorizer.joblib")
            
        try:
            if os.path.exists(model_path):
                if self.method == 'tfidf':
                    self.vectorizer = joblib.load(model_path)
                elif self.method == 'word2vec':
                    self.vectorizer = Word2Vec.load(model_path)
                elif self.method == 'fasttext':
                    self.vectorizer = FastText.load(model_path)
                
                logger.info(f"Loaded {self.method} vectorizer model from {model_path}")
                
                # Also load SVD model if it exists
                svd_path = os.path.join(self.models_dir, f"{self.method}_svd_model.joblib")
                if os.path.exists(svd_path):
                    self.svd = joblib.load(svd_path)
                    logger.info(f"Loaded SVD model from {svd_path}")
                
                return True
            else:
                logger.warning(f"Model file not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Sample articles
    sample_articles = [
        "삼성전자가 반도체 패키징 역량 강화를 위해 영입한 대만 TSMC 출신 베테랑 엔지니어 린준청 부사장이 회사를 떠났다.",
        "SK하이닉스는 인공지능 반도체 시장 공략을 위한 HBM 생산 확대에 나선다.",
        "LG전자가 올레드 TV 신제품을 출시하며 프리미엄 TV 시장 공략을 강화한다.",
        "현대자동차는 전기차 판매 호조에 힘입어 분기 실적이 시장 예상치를 상회했다.",
        "카카오는 인공지능 기술 강화를 위해 연구개발 투자를 확대한다고 밝혔다."
    ]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'title': sample_articles,
        'processed_title': sample_articles,
        'processed_content': sample_articles
    })
    
    # Create a vectorizer and vectorize articles
    vectorizer = NewsVectorizer(method='tfidf', max_features=100)
    vectorized_df = vectorizer.vectorize_articles(df, reduce_dims=True, n_components=5)
    
    print(f"Vectorized DataFrame shape: {vectorized_df.shape}")
    print(f"First vector: {vectorized_df['vector'].iloc[0]}")
    
    # Get top terms
    top_terms = vectorizer.get_top_terms(5)
    print("Top terms:")
    for term, value in top_terms.items():
        print(f"  {term}: {value:.4f}")
