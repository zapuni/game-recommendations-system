"""
Model Evaluation CLI Script
===========================
Run this script to evaluate all recommendation models and save results.

Usage:
    python run_evaluation.py [--samples 50] [--k 10] [--output results/evaluation_results.json]

Example:
    python run_evaluation.py --samples 100 --k 10
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from data_processor import DataProcessor
from recommender import RecommenderSystem
from evaluator import ModelEvaluator
from sklearn.feature_extraction.text import TfidfVectorizer


def get_content_features(df):
    """Extract content features using TF-IDF vectorization."""
    df_copy = df.copy()
    df_copy['combined_features'] = (
        df_copy['genres'].fillna('') + ' ' +
        df_copy['categories'].fillna('') + ' ' +
        df_copy['primary_genre'].fillna('')
    )
    
    tfidf = TfidfVectorizer(
        max_features=50,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    feature_matrix = tfidf.fit_transform(df_copy['combined_features']).toarray()
    return feature_matrix, df['appid'].tolist(), df['name'].tolist()


def load_data():
    """Load and process game data."""
    print("[INFO] Loading data...")
    processor = DataProcessor(data_path="data/steam-store-games")
    
    df = processor.load_data()
    print(f"[INFO] Loaded {len(df)} games")
    
    df = processor.clean_data()
    df = processor.engineer_features()
    df_filtered = processor.get_filtered_dataset(min_ratings=10)
    
    print(f"[INFO] Filtered to {len(df_filtered)} games with min_ratings >= 10")
    return processor, df, df_filtered


def initialize_recommender(df_filtered):
    """Initialize the recommender system."""
    print("[INFO] Initializing recommender system...")
    recommender = RecommenderSystem(df_filtered)
    feature_matrix, appids, names = get_content_features(df_filtered)
    recommender.set_content_features(feature_matrix)
    print("[OK] Recommender system initialized")
    return recommender


def run_evaluation(df_filtered, recommender, n_samples: int = 50, k: int = 10):
    """
    Run comprehensive model evaluation.
    
    Args:
        df_filtered: Filtered DataFrame
        recommender: RecommenderSystem instance
        n_samples: Number of test samples
        k: Top-K value for ranking metrics
    
    Returns:
        Dictionary with all evaluation results
    """
    print(f"\n[INFO] Running evaluation with {n_samples} samples, K={k}...")
    print("-" * 50)
    
    # Sample test games
    test_games = df_filtered.sample(min(n_samples, len(df_filtered)))['appid'].tolist()
    
    evaluator = ModelEvaluator(df_filtered)
    
    # Evaluate each model
    print("[INFO] Evaluating Content-Based model...")
    content_eval = evaluator.evaluate_content_based(recommender, test_games, k)
    print(f"       Precision@{k}: {content_eval['avg_precision_at_k']:.4f}")
    print(f"       Recall@{k}: {content_eval['avg_recall_at_k']:.4f}")
    print(f"       RMSE: {content_eval['rmse']:.4f}")
    print(f"       MAE: {content_eval['mae']:.4f}")
    
    print("\n[INFO] Evaluating Hybrid model...")
    hybrid_eval = evaluator.evaluate_hybrid(recommender, test_games, k)
    print(f"       Precision@{k}: {hybrid_eval['avg_precision_at_k']:.4f}")
    print(f"       Recall@{k}: {hybrid_eval['avg_recall_at_k']:.4f}")
    print(f"       RMSE: {hybrid_eval['rmse']:.4f}")
    print(f"       MAE: {hybrid_eval['mae']:.4f}")
    
    print("\n[INFO] Evaluating Popularity Baseline...")
    popularity_eval = evaluator.evaluate_popularity_baseline(test_games, k)
    print(f"       Precision@{k}: {popularity_eval['avg_precision_at_k']:.4f}")
    print(f"       Recall@{k}: {popularity_eval['avg_recall_at_k']:.4f}")
    print(f"       RMSE: {popularity_eval['rmse']:.4f}")
    print(f"       MAE: {popularity_eval['mae']:.4f}")
    
    # Compile results
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_samples': n_samples,
            'k_value': k,
            'total_games': len(df_filtered)
        },
        'ranking_metrics': [
            content_eval,
            hybrid_eval,
            popularity_eval
        ]
    }
    
    return results


def save_results(results: dict, output_path: str):
    """Save evaluation results to JSON file."""
    # Ensure directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {output_path}")


def print_summary(results: dict):
    """Print evaluation summary table."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    k = results['metadata']['k_value']
    
    # Header
    print(f"{'Model':<30} {'Precision@'+str(k):<12} {'Recall@'+str(k):<12} {'RMSE':<10} {'MAE':<10}")
    print("-" * 70)
    
    # Rows
    for r in results['ranking_metrics']:
        model_name = r['model'][:28]
        print(f"{model_name:<30} {r['avg_precision_at_k']:<12.4f} {r['avg_recall_at_k']:<12.4f} {r['rmse']:<10.4f} {r['mae']:<10.4f}")
    
    print("-" * 70)
    
    # Best models
    best_precision = max(results['ranking_metrics'], key=lambda x: x['avg_precision_at_k'])
    best_rmse = min(results['ranking_metrics'], key=lambda x: x['rmse'])
    
    print(f"\nBest Precision@{k}: {best_precision['model']}")
    print(f"Best RMSE: {best_rmse['model']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Run model evaluation and save results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evaluation.py
    python run_evaluation.py --samples 100 --k 10
    python run_evaluation.py --output results/my_eval.json
        """
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=50,
        help='Number of test samples (default: 50)'
    )
    
    parser.add_argument(
        '--k', '-k',
        type=int,
        default=10,
        help='Top-K value for ranking metrics (default: 10)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/evaluation_results.json',
        help='Output file path (default: results/evaluation_results.json)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL EVALUATION SCRIPT")
    print("=" * 70)
    print(f"Samples: {args.samples}")
    print(f"K value: {args.k}")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    try:
        # Load data
        processor, df, df_filtered = load_data()
        
        # Initialize recommender
        recommender = initialize_recommender(df_filtered)
        
        # Run evaluation
        results = run_evaluation(df_filtered, recommender, args.samples, args.k)
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        print_summary(results)
        
        print("\n[OK] Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

