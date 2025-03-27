"""
Evaluation framework for the CLEAR system.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CLEAREvaluator:
    """
    Evaluation framework for the CLEAR system.
    Provides metrics for clustering quality, impact prediction accuracy, and recommendation relevance.
    """
    
    def __init__(self, results_dir: str = "results/evaluation"):
        """
        Initialize the evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("Initialized CLEAR evaluator")
    
    def evaluate_clustering(self, articles_df: pd.DataFrame, 
                           vector_col: str = 'vector',
                           cluster_col: str = 'cluster_id') -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            articles_df: DataFrame containing clustered articles
            vector_col: Column name for vector representations
            cluster_col: Column name for cluster assignments
            
        Returns:
            Dictionary with clustering quality metrics
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for clustering evaluation")
            return {}
            
        if vector_col not in articles_df.columns:
            logger.error(f"Vector column '{vector_col}' not found in DataFrame")
            return {}
            
        if cluster_col not in articles_df.columns:
            logger.error(f"Cluster column '{cluster_col}' not found in DataFrame")
            return {}
            
        logger.info(f"Evaluating clustering quality for {len(articles_df)} articles")
        
        try:
            # Filter out articles not in valid clusters
            valid_df = articles_df[articles_df[cluster_col] >= 0].copy()
            
            if len(valid_df) < 2:
                logger.warning("Not enough articles in valid clusters for evaluation")
                return {}
            
            # Extract vectors and cluster labels
            vectors = np.array(valid_df[vector_col].tolist())
            labels = valid_df[cluster_col].values
            
            # Calculate clustering metrics
            metrics = {}
            
            # Silhouette score (higher is better, range: -1 to 1)
            try:
                silhouette = silhouette_score(vectors, labels, metric='cosine')
                metrics['silhouette_score'] = silhouette
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {str(e)}")
            
            # Calinski-Harabasz index (higher is better)
            try:
                calinski = calinski_harabasz_score(vectors, labels)
                metrics['calinski_harabasz_score'] = calinski
            except Exception as e:
                logger.warning(f"Could not calculate Calinski-Harabasz score: {str(e)}")
            
            # Davies-Bouldin index (lower is better)
            try:
                davies = davies_bouldin_score(vectors, labels)
                metrics['davies_bouldin_score'] = davies
            except Exception as e:
                logger.warning(f"Could not calculate Davies-Bouldin score: {str(e)}")
            
            # Cluster statistics
            cluster_counts = valid_df[cluster_col].value_counts()
            metrics['num_clusters'] = len(cluster_counts)
            metrics['avg_cluster_size'] = cluster_counts.mean()
            metrics['max_cluster_size'] = cluster_counts.max()
            metrics['min_cluster_size'] = cluster_counts.min()
            
            logger.info(f"Calculated clustering metrics: {metrics}")
            
            # Save metrics
            self._save_metrics('clustering', metrics)
            
            # Create visualizations
            self._visualize_clustering_metrics(metrics, valid_df, vectors, labels)
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating clustering: {str(e)}")
            return {}
    
    def evaluate_impact_prediction(self, articles_df: pd.DataFrame,
                                 actual_col: str = 'impact_score',
                                 predicted_col: str = 'predicted_impact') -> Dict[str, float]:
        """
        Evaluate impact prediction accuracy.
        
        Args:
            articles_df: DataFrame containing articles with actual and predicted impact scores
            actual_col: Column name for actual impact scores
            predicted_col: Column name for predicted impact scores
            
        Returns:
            Dictionary with prediction accuracy metrics
        """
        if len(articles_df) == 0:
            logger.warning("Empty DataFrame provided for impact prediction evaluation")
            return {}
            
        if actual_col not in articles_df.columns:
            logger.error(f"Actual impact column '{actual_col}' not found in DataFrame")
            return {}
            
        if predicted_col not in articles_df.columns:
            logger.error(f"Predicted impact column '{predicted_col}' not found in DataFrame")
            return {}
            
        logger.info(f"Evaluating impact prediction for {len(articles_df)} articles")
        
        try:
            # Filter out rows with missing values
            valid_df = articles_df.dropna(subset=[actual_col, predicted_col]).copy()
            
            if len(valid_df) < 10:
                logger.warning("Not enough articles with impact scores for evaluation")
                return {}
            
            # Extract actual and predicted values
            y_true = valid_df[actual_col].values
            y_pred = valid_df[predicted_col].values
            
            # Calculate regression metrics
            metrics = {}
            
            # Mean Squared Error (lower is better)
            mse = mean_squared_error(y_true, y_pred)
            metrics['mean_squared_error'] = mse
            
            # Root Mean Squared Error (lower is better)
            rmse = np.sqrt(mse)
            metrics['root_mean_squared_error'] = rmse
            
            # Mean Absolute Error (lower is better)
            mae = mean_absolute_error(y_true, y_pred)
            metrics['mean_absolute_error'] = mae
            
            # R-squared (higher is better, range: 0 to 1)
            r2 = r2_score(y_true, y_pred)
            metrics['r2_score'] = r2
            
            # Direction accuracy (percentage of correct impact direction predictions)
            direction_match = np.sign(y_true) == np.sign(y_pred)
            direction_accuracy = np.mean(direction_match)
            metrics['direction_accuracy'] = direction_accuracy
            
            logger.info(f"Calculated impact prediction metrics: {metrics}")
            
            # Save metrics
            self._save_metrics('impact_prediction', metrics)
            
            # Create visualizations
            self._visualize_impact_metrics(metrics, valid_df, actual_col, predicted_col)
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating impact prediction: {str(e)}")
            return {}
    
    def evaluate_recommendations(self, recommendations: Dict[str, Any],
                               ground_truth: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate recommendation quality.
        
        Args:
            recommendations: Dictionary with recommendations
            ground_truth: Optional DataFrame with ground truth data
            
        Returns:
            Dictionary with recommendation quality metrics
        """
        if not recommendations:
            logger.warning("Empty recommendations provided for evaluation")
            return {}
            
        logger.info("Evaluating recommendation quality")
        
        try:
            metrics = {}
            
            # Basic recommendation statistics
            if 'top_articles' in recommendations:
                top_articles = recommendations['top_articles']
                if isinstance(top_articles, pd.DataFrame):
                    metrics['num_recommended_articles'] = len(top_articles)
                    
                    if 'impact_score' in top_articles.columns:
                        metrics['avg_impact_score'] = top_articles['impact_score'].mean()
                        metrics['max_impact_score'] = top_articles['impact_score'].max()
                        metrics['min_impact_score'] = top_articles['impact_score'].min()
                    
                    if 'cluster_id' in top_articles.columns:
                        unique_clusters = top_articles['cluster_id'].nunique()
                        metrics['unique_clusters_in_recommendations'] = unique_clusters
            
            if 'top_clusters' in recommendations:
                top_clusters = recommendations['top_clusters']
                if isinstance(top_clusters, dict):
                    metrics['num_recommended_clusters'] = len(top_clusters)
            
            if 'trending_topics' in recommendations:
                trending_topics = recommendations['trending_topics']
                if isinstance(trending_topics, list):
                    metrics['num_trending_topics'] = len(trending_topics)
            
            # Evaluate against ground truth if provided
            if ground_truth is not None and not ground_truth.empty:
                # This would compare recommendations against some ground truth data
                # For example, comparing against manually curated recommendations
                # or historical user interactions
                pass
            
            logger.info(f"Calculated recommendation metrics: {metrics}")
            
            # Save metrics
            self._save_metrics('recommendations', metrics)
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating recommendations: {str(e)}")
            return {}
    
    def evaluate_pipeline(self, pipeline_results: Dict[str, Any],
                        articles_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the complete pipeline.
        
        Args:
            pipeline_results: Results from running the pipeline
            articles_df: DataFrame with processed articles
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating complete pipeline")
        
        try:
            # Combine all evaluation metrics
            evaluation = {}
            
            # Evaluate clustering if possible
            if 'vector' in articles_df.columns and 'cluster_id' in articles_df.columns:
                clustering_metrics = self.evaluate_clustering(articles_df)
                evaluation['clustering'] = clustering_metrics
            
            # Evaluate impact prediction if possible
            if 'impact_score' in articles_df.columns and 'predicted_impact' in articles_df.columns:
                impact_metrics = self.evaluate_impact_prediction(articles_df)
                evaluation['impact_prediction'] = impact_metrics
            
            # Evaluate recommendations
            if 'recommendations' in pipeline_results:
                recommendation_metrics = self.evaluate_recommendations(pipeline_results['recommendations'])
                evaluation['recommendations'] = recommendation_metrics
            
            # Overall pipeline metrics
            overall_metrics = {
                'news_count': pipeline_results.get('news_count', 0),
                'cluster_count': pipeline_results.get('cluster_count', 0),
                'processing_time': pipeline_results.get('processing_time', 0),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            evaluation['overall'] = overall_metrics
            
            # Save complete evaluation
            self._save_evaluation(evaluation)
            
            logger.info("Completed pipeline evaluation")
            return evaluation
        except Exception as e:
            logger.error(f"Error evaluating pipeline: {str(e)}")
            return {}
    
    def _save_metrics(self, metric_type: str, metrics: Dict[str, Any]) -> None:
        """
        Save metrics to file.
        
        Args:
            metric_type: Type of metrics
            metrics: Dictionary with metrics
        """
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics to JSON
            metrics_path = os.path.join(self.results_dir, f"{metric_type}_metrics_{timestamp}.json")
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {metric_type} metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def _save_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """
        Save complete evaluation to file.
        
        Args:
            evaluation: Dictionary with evaluation metrics
        """
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save evaluation to JSON
            eval_path = os.path.join(self.results_dir, f"evaluation_{timestamp}.json")
            
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved complete evaluation to {eval_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation: {str(e)}")
    
    def _visualize_clustering_metrics(self, metrics: Dict[str, float],
                                    df: pd.DataFrame,
                                    vectors: np.ndarray,
                                    labels: np.ndarray) -> None:
        """
        Create visualizations for clustering metrics.
        
        Args:
            metrics: Dictionary with clustering metrics
            df: DataFrame with clustered articles
            vectors: Vector representations
            labels: Cluster labels
        """
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Cluster size distribution
            plt.figure(figsize=(10, 6))
            cluster_sizes = df['cluster_id'].value_counts().sort_index()
            sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
            plt.title('Cluster Size Distribution')
            plt.xlabel('Cluster ID')
            plt.ylabel('Number of Articles')
            plt.xticks(rotation=45)
            
            # Save plot
            cluster_size_path = os.path.join(self.results_dir, f"cluster_size_distribution_{timestamp}.png")
            plt.savefig(cluster_size_path)
            plt.close()
            
            # 2. Cluster visualization using dimensionality reduction
            if len(vectors) > 1:
                from sklearn.decomposition import TruncatedSVD
                
                # Reduce to 2 dimensions for visualization
                svd = TruncatedSVD(n_components=2, random_state=42)
                reduced_vectors = svd.fit_transform(vectors)
                
                plt.figure(figsize=(12, 8))
                
                # Plot points with cluster colors
                unique_labels = np.unique(labels)
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    plt.scatter(
                        reduced_vectors[mask, 0],
                        reduced_vectors[mask, 1],
                        c=[colors[i]],
                        label=f'Cluster {label}',
                        alpha=0.7
                    )
                
                plt.title('Article Clusters Visualization')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.legend(loc='best')
                
                # Save plot
                cluster_viz_path = os.path.join(self.results_dir, f"cluster_visualization_{timestamp}.png")
                plt.savefig(cluster_viz_path)
                plt.close()
            
            # 3. Metrics summary
            plt.figure(figsize=(10, 6))
            metric_names = []
            metric_values = []
            
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and name not in ['num_clusters', 'avg_cluster_size', 'max_cluster_size', 'min_cluster_size']:
                    metric_names.append(name)
                    metric_values.append(value)
            
            if metric_names:
                sns.barplot(x=metric_names, y=metric_values)
                plt.title('Clustering Quality Metrics')
                plt.xlabel('Metric')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                
                # Save plot
                metrics_path = os.path.join(self.results_dir, f"clustering_metrics_{timestamp}.png")
                plt.savefig(metrics_path)
                plt.close()
            
            logger.info("Created clustering visualizations")
        except Exception as e:
            logger.error(f"Error creating clustering visualizations: {str(e)}")
    
    def _visualize_impact_metrics(self, metrics: Dict[str, float],
                                df: pd.DataFrame,
                                actual_col: str,
                                predicted_col: str) -> None:
        """
        Create visualizations for impact prediction metrics.
        
        Args:
            metrics: Dictionary with impact prediction metrics
            df: DataFrame with actual and predicted impact scores
            actual_col: Column name for actual impact scores
            predicted_col: Column name for predicted impact scores
        """
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Actual vs Predicted scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(df[actual_col], df[predicted_col], alpha=0.7)
            
            # Add perfect prediction line
            min_val = min(df[actual_col].min(), df[predicted_col].min())
            max_val = max(df[actual_col].max(), df[predicted_col].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title('Actual vs Predicted Impact Scores')
            plt.xlabel('Actual Impact Score')
            plt.ylabel('Predicted Impact Score')
            
            # Save plot
            scatter_path = os.path.join(self.results_dir, f"impact_scatter_{timestamp}.png")
            plt.savefig(scatter_path)
            plt.close()
            
            # 2. Error distribution
            plt.figure(figsize=(10, 6))
            errors = df[predicted_col] - df[actual_col]
            sns.histplot(errors, kde=True)
            plt.title('Prediction Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            
            # Save plot
            error_path = os.path.join(self.results_dir, f"impact_error_distribution_{timestamp}.png")
            plt.savefig(error_path)
            plt.close()
            
            # 3. Metrics summary
            plt.figure(figsize=(10, 6))
            metric_names = []
            metric_values = []
            
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_names.append(name)
                    metric_values.append(value)
            
            if metric_names:
                sns.barplot(x=metric_names, y=metric_values)
                plt.title('Impact Prediction Metrics')
                plt.xlabel('Metric')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                
                # Save plot
                metrics_path = os.path.join(self.results_dir, f"impact_metrics_{timestamp}.png")
                plt.savefig(metrics_path)
                plt.close()
            
            logger.info("Created impact prediction visualizations")
        except Exception as e:
            logger.error(f"Error creating impact visualizations: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Sample data for clustering evaluation
    sample_vectors = np.random.rand(100, 50)
    sample_clusters = np.random.randint(0, 5, 100)
    
    df = pd.DataFrame({
        'vector': list(sample_vectors),
        'cluster_id': sample_clusters,
        'impact_score': np.random.uniform(-5, 5, 100),
        'predicted_impact': np.random.uniform(-5, 5, 100)
    })
    
    # Create evaluator
    evaluator = CLEAREvaluator()
    
    # Evaluate clustering
    clustering_metrics = evaluator.evaluate_clustering(df)
    print("Clustering Metrics:")
    print(clustering_metrics)
    
    # Evaluate impact prediction
    impact_metrics = evaluator.evaluate_impact_prediction(df)
    print("\nImpact Prediction Metrics:")
    print(impact_metrics)
    
    # Sample recommendations
    recommendations = {
        'top_articles': df.head(10),
        'top_clusters': {0: df[df['cluster_id'] == 0].head(3)},
        'trending_topics': [{'topic': 'Topic 1', 'score': 0.9}]
    }
    
    # Evaluate recommendations
    recommendation_metrics = evaluator.evaluate_recommendations(recommendations)
    print("\nRecommendation Metrics:")
    print(recommendation_metrics)
