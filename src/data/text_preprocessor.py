"""
Text preprocessor module for Korean financial news articles.
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set
from konlpy.tag import Mecab, Okt
import nltk
from nltk.corpus import stopwords
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Text preprocessor for Korean financial news articles.
    Optimized for Korean text with support for mixed Korean/English content.
    """
    
    def __init__(self, 
                 language: str = 'ko',
                 use_mecab: bool = True,
                 remove_stopwords: bool = True,
                 custom_stopwords: List[str] = None,
                 min_token_length: int = 2,
                 max_token_length: int = 50):
        """
        Initialize the text preprocessor.
        
        Args:
            language: Primary language ('ko' for Korean, 'en' for English)
            use_mecab: Whether to use Mecab for Korean tokenization
            remove_stopwords: Whether to remove stopwords
            custom_stopwords: List of additional stopwords
            min_token_length: Minimum token length to keep
            max_token_length: Maximum token length to keep
        """
        self.language = language
        self.use_mecab = use_mecab
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Initialize tokenizers
        if self.use_mecab:
            try:
                self.tokenizer = Mecab()
                logger.info("Initialized Mecab tokenizer")
            except Exception as e:
                logger.warning(f"Failed to initialize Mecab: {str(e)}. Falling back to Okt.")
                self.tokenizer = Okt()
                self.use_mecab = False
        else:
            self.tokenizer = Okt()
            logger.info("Initialized Okt tokenizer")
        
        # Initialize stopwords
        self.stopwords = self._initialize_stopwords(custom_stopwords)
        
        logger.info(f"Initialized TextPreprocessor with language={language}, use_mecab={use_mecab}")
    
    def _initialize_stopwords(self, custom_stopwords: List[str] = None) -> Set[str]:
        """
        Initialize stopwords for text preprocessing.
        
        Args:
            custom_stopwords: List of additional stopwords
            
        Returns:
            Set of stopwords
        """
        stopword_set = set()
        
        # Download NLTK stopwords if needed
        try:
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK stopwords: {str(e)}")
        
        # Add Korean stopwords
        korean_stopwords = self._get_korean_stopwords()
        stopword_set.update(korean_stopwords)
        
        # Add English stopwords if available
        try:
            english_stopwords = set(stopwords.words('english'))
            stopword_set.update(english_stopwords)
        except Exception as e:
            logger.warning(f"Failed to load English stopwords: {str(e)}")
        
        # Add custom stopwords
        if custom_stopwords:
            stopword_set.update(custom_stopwords)
        
        logger.info(f"Initialized {len(stopword_set)} stopwords")
        return stopword_set
    
    def _get_korean_stopwords(self) -> Set[str]:
        """
        Get Korean stopwords.
        
        Returns:
            Set of Korean stopwords
        """
        # Common Korean stopwords
        korean_stopwords = {
            '이', '그', '저', '것', '수', '등', '들', '및', '에', '에서', '의', '을', '를', '이', '가',
            '은', '는', '이다', '있다', '하다', '이런', '저런', '그런', '한', '할', '하는', '했다', '된다',
            '일', '때', '년', '월', '일', '로', '으로', '에서', '까지', '부터', '에게', '께', '에게서',
            '께서', '처럼', '만큼', '같이', '도', '만', '라도', '커녕', '든지', '든가', '나', '이나',
            '거나', '또는', '혹은', '또한', '그리고', '따라서', '그러나', '하지만', '그래도', '그러면',
            '그러므로', '그런데', '그리하여', '하여', '해서', '때문에', '위해', '위하여', '에도', '에는',
            '에서는', '로는', '로서', '로써', '로부터', '이라고', '라고', '이라는', '라는', '이라면',
            '라면', '이었다', '였다', '더라도', '이든', '든', '이야', '야', '이건', '건', '이네요',
            '네요', '예요', '이에요', '군요', '는군요', '이군요', '까', '나요', '이나요', '까요', '이까요'
        }
        
        # Add financial/news specific stopwords
        financial_stopwords = {
            '주가', '주식', '증시', '코스피', '코스닥', '상승', '하락', '투자', '매수', '매도', '호가',
            '거래량', '시가', '종가', '고가', '저가', '기업', '회사', '그룹', '뉴스', '기자', '보도',
            '전망', '예상', '분석', '시장', '경제', '금융', '증권', '은행', '펀드', '채권', '외국인',
            '기관', '개인', '투자자', '실적', '실적발표', '공시', '공개', '발표', '전일', '어제', '오늘',
            '내일', '지난', '이번', '다음', '최근', '작년', '올해', '내년', '분기', '반기', '연간'
        }
        
        # Combine all Korean stopwords
        all_korean_stopwords = korean_stopwords.union(financial_stopwords)
        
        # Add Mecab-specific stopwords if using Mecab
        if self.use_mecab:
            mecab_stopwords = {
                'NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ',
                'IC', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN',
                'ETM', 'XPN', 'XSN', 'XSV', 'XSA', 'XR', 'SF', 'SE', 'SSO', 'SSC', 'SC', 'SY', 'SL', 'SH',
                'SN', 'VA+ETM'
            }
            all_korean_stopwords.update(mecab_stopwords)
        
        return all_korean_stopwords
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters, URLs, etc.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase for English parts
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters but keep Korean characters
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using the appropriate tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            if self.use_mecab:
                # Mecab returns (token, pos) pairs
                tokens = self.tokenizer.pos(text)
                # Extract only tokens with relevant POS tags (nouns, verbs, adjectives)
                filtered_tokens = [token for token, pos in tokens if pos.startswith('N') or pos.startswith('V') or pos.startswith('XR')]
            else:
                # Okt tokenization
                tokens = self.tokenizer.morphs(text)
                filtered_tokens = tokens
            
            return filtered_tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {str(e)}")
            return []
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Filter tokens by removing stopwords and short/long tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list of tokens
        """
        if not tokens:
            return []
        
        # Filter by token length
        length_filtered = [token for token in tokens if self.min_token_length <= len(token) <= self.max_token_length]
        
        # Filter stopwords if enabled
        if self.remove_stopwords:
            stopword_filtered = [token for token in length_filtered if token not in self.stopwords]
            return stopword_filtered
        else:
            return length_filtered
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning, tokenizing, and filtering.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text (tokens joined by space)
        """
        if not text:
            return ""
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize text
        tokens = self.tokenize_text(cleaned_text)
        
        # Filter tokens
        filtered_tokens = self.filter_tokens(tokens)
        
        # Join tokens back into text
        preprocessed_text = ' '.join(filtered_tokens)
        
        return preprocessed_text
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                            title_col: str = 'Title', 
                            body_col: str = 'Body',
                            summary_col: str = 'AI Summary',
                            use_summary: bool = True) -> pd.DataFrame:
        """
        Preprocess a DataFrame of news articles.
        
        Args:
            df: DataFrame containing news articles
            title_col: Column name for article titles
            body_col: Column name for article bodies
            summary_col: Column name for article summaries
            use_summary: Whether to use summaries when available
            
        Returns:
            DataFrame with added preprocessed columns
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided for preprocessing")
            return df
        
        logger.info(f"Preprocessing {len(df)} articles")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Preprocess titles
        if title_col in result_df.columns:
            result_df['processed_title'] = result_df[title_col].apply(self.preprocess_text)
            logger.info(f"Preprocessed {title_col} column")
        
        # Preprocess bodies or summaries
        if body_col in result_df.columns:
            # If using summaries and they are available, use them instead of full body
            if use_summary and summary_col in result_df.columns:
                # Use summary when available, otherwise use body
                result_df['content_to_process'] = result_df.apply(
                    lambda row: row[summary_col] if pd.notna(row[summary_col]) and row[summary_col] else row[body_col],
                    axis=1
                )
                logger.info(f"Using {summary_col} when available, falling back to {body_col}")
            else:
                result_df['content_to_process'] = result_df[body_col]
                logger.info(f"Using {body_col} for content processing")
            
            # Preprocess the selected content
            result_df['processed_content'] = result_df['content_to_process'].apply(self.preprocess_text)
            
            # Remove temporary column
            result_df = result_df.drop(columns=['content_to_process'])
        
        logger.info(f"Preprocessing complete for {len(result_df)} articles")
        return result_df
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract top keywords from text.
        
        Args:
            text: Input text
            top_n: Number of top keywords to extract
            
        Returns:
            List of top keywords
        """
        if not text:
            return []
        
        # Clean and tokenize text
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        filtered_tokens = self.filter_tokens(tokens)
        
        # Count token frequencies
        token_counts = Counter(filtered_tokens)
        
        # Get top N tokens
        top_tokens = [token for token, count in token_counts.most_common(top_n)]
        
        return top_tokens
    
    def extract_keywords_from_df(self, df: pd.DataFrame, 
                               text_col: str = 'processed_content',
                               top_n: int = 10) -> pd.DataFrame:
        """
        Extract keywords from a DataFrame of preprocessed articles.
        
        Args:
            df: DataFrame containing preprocessed articles
            text_col: Column name for preprocessed text
            top_n: Number of top keywords to extract per article
            
        Returns:
            DataFrame with added keywords column
        """
        if len(df) == 0 or text_col not in df.columns:
            logger.warning(f"Cannot extract keywords: DataFrame is empty or {text_col} column not found")
            return df
        
        logger.info(f"Extracting keywords from {len(df)} articles")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract keywords for each article
        result_df['keywords'] = result_df[text_col].apply(lambda x: self.extract_keywords(x, top_n))
        
        logger.info(f"Extracted keywords for {len(result_df)} articles")
        return result_df
    
    def add_custom_stopwords(self, new_stopwords: List[str]) -> None:
        """
        Add custom stopwords to the existing set.
        
        Args:
            new_stopwords: List of new stopwords to add
        """
        if not new_stopwords:
            return
            
        self.stopwords.update(new_stopwords)
        logger.info(f"Added {len(new_stopwords)} custom stopwords, total: {len(self.stopwords)}")
    
    def load_stopwords_from_file(self, filepath: str) -> None:
        """
        Load stopwords from a file.
        
        Args:
            filepath: Path to stopwords file (one word per line)
        """
        if not os.path.exists(filepath):
            logger.warning(f"Stopwords file not found: {filepath}")
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                new_stopwords = [line.strip() for line in f if line.strip()]
                
            self.add_custom_stopwords(new_stopwords)
            logger.info(f"Loaded {len(new_stopwords)} stopwords from {filepath}")
        except Exception as e:
            logger.error(f"Error loading stopwords from file: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create preprocessor instance
    preprocessor = TextPreprocessor(use_mecab=True)
    
    # Example text
    sample_text = "삼성전자가 반도체 패키징 역량 강화를 위해 영입한 대만 TSMC 출신 베테랑 엔지니어 린준청 부사장이 회사를 떠났다."
    
    # Preprocess text
    preprocessed = preprocessor.preprocess_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Preprocessed: {preprocessed}")
    
    # Extract keywords
    keywords = preprocessor.extract_keywords(sample_text, top_n=5)
    print(f"Keywords: {keywords}")
