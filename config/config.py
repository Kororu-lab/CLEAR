"""
Configuration file for the CLEAR system.
"""

import os
import yaml

# Default configuration
DEFAULT_CONFIG = {
    'data_dir': 'data',
    'models_dir': 'models',
    'results_dir': 'results',
    'use_gpu': True,
    'stock_tickers': ['005930'],  # Samsung Electronics
    'news_crawler': {
        'source': 'yna',
        'use_ai_summary': True
    },
    'text_preprocessor': {
        'language': 'ko',
        'use_mecab': True,
        'remove_stopwords': True,
        'min_token_length': 2
    },
    'news_vectorizer': {
        'method': 'tfidf',
        'max_features': 10000,
        'embedding_dim': 300,
        'title_weight': 2.0,
        'combine_title_content': True,
        'reduce_dims': True,
        'n_components': 100
    },
    'news_clustering': {
        'distance_threshold': 0.7,
        'min_cluster_size': 3,
        'max_cluster_size': 20,
        'linkage': 'average'
    },
    'stock_impact': {
        'time_windows': [
            {"name": "immediate", "hours": 1},
            {"name": "short_term", "hours": 24},
            {"name": "medium_term", "days": 3}
        ],
        'impact_thresholds': {
            "high": 0.02,    # 2% price change
            "medium": 0.01,  # 1% price change
            "low": 0.005     # 0.5% price change
        },
        'model_type': 'random_forest'
    },
    'news_recommender': {
        'weights': {
            'impact': 0.4,      # Stock Impact (SI)
            'quality': 0.2,     # Quality Estimation (QE)
            'content': 0.2,     # Content-Based Filtering (CBF)
            'collaborative': 0.1, # Collaborative Filtering (CF)
            'recency': 0.1      # Latest news prioritization
        }
    },
    'schedule': {
        'market_open': '09:00',
        'market_close': '15:30'
    }
}


def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        print("Using default configuration")
        return DEFAULT_CONFIG


def save_config(config, config_path='config/config.yaml'):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")


# Create default configuration file if it doesn't exist
if __name__ == "__main__":
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        save_config(DEFAULT_CONFIG, config_path)
