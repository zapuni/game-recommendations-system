"""
Model Evaluation Module
=======================
Comprehensive evaluation metrics for recommendation systems.
Includes ranking metrics (Precision@K, Recall@K, NDCG) and 
rating prediction metrics (RMSE, MAE).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


class EvaluationMetrics:
    """
    Calculate recommendation system evaluation metrics.
    
    Metrics included:
    - RMSE: Root Mean Squared Error (rating prediction)
    - MAE: Mean Absolute Error (rating prediction)
    - Precision@K: Proportion of relevant items in top-K
    - Recall@K: Proportion of relevant items found
    - NDCG@K: Normalized Discounted Cumulative Gain
    - Coverage: Catalog coverage
    - Diversity: Recommendation diversity
    """
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error for rating prediction.
        
        Args:
            y_true: Actual ratings
            y_pred: Predicted ratings
        
        Returns:
            RMSE score (lower is better)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error for rating prediction.
        
        Args:
            y_true: Actual ratings
            y_pred: Predicted ratings
        
        Returns:
            MAE score (lower is better)
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def precision_at_k(relevant_items: set, recommended_items: List, k: int = 10) -> float:
        """
        Precision@K: Proportion of recommended items that are relevant
        
        Args:
            relevant_items: Set of relevant item IDs
            recommended_items: List of recommended item IDs (ordered)
            k: Cutoff for top-k
        
        Returns:
            Precision@K score (0-1)
        """
        recommended_at_k = set(recommended_items[:k])
        if len(recommended_at_k) == 0:
            return 0.0
        
        hits = len(relevant_items.intersection(recommended_at_k))
        return hits / k
    
    @staticmethod
    def recall_at_k(relevant_items: set, recommended_items: List, k: int = 10) -> float:
        """
        Recall@K: Proportion of relevant items that appear in top-k recommendations
        
        Args:
            relevant_items: Set of relevant item IDs
            recommended_items: List of recommended item IDs (ordered)
            k: Cutoff for top-k
        
        Returns:
            Recall@K score (0-1)
        """
        recommended_at_k = set(recommended_items[:k])
        if len(relevant_items) == 0:
            return 0.0
        
        hits = len(relevant_items.intersection(recommended_at_k))
        return hits / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(relevant_items: set, recommended_items: List, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K
        
        Args:
            relevant_items: Set of relevant item IDs
            recommended_items: List of recommended item IDs (ordered by score)
            k: Cutoff for top-k
        
        Returns:
            NDCG@K score (0-1)
        """
        recommended_at_k = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_at_k, 1):
            relevance = 1.0 if item in relevant_items else 0.0
            dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG with all relevant items ranked first)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def coverage(recommended_items: List, total_items: int) -> float:
        """
        Coverage: Proportion of unique items that can be recommended
        
        Args:
            recommended_items: List of all recommended items (from multiple queries)
            total_items: Total number of items in catalog
        
        Returns:
            Coverage score (0-1)
        """
        unique_items = len(set(recommended_items))
        return unique_items / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def diversity(recommended_items: List, item_features: Dict[int, List]) -> float:
        """
        Diversity: Average dissimilarity between recommended items
        
        Args:
            recommended_items: List of recommended item IDs
            item_features: Dict mapping item ID to feature vector
        
        Returns:
            Diversity score (0-1)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if len(recommended_items) < 2:
            return 1.0
        
        # Get feature vectors for recommended items
        features = []
        for item_id in recommended_items:
            if item_id in item_features:
                features.append(item_features[item_id])
        
        if len(features) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        features = np.array(features)
        similarity_matrix = cosine_similarity(features)
        
        # Diversity is 1 - average similarity
        mask = np.triu_indices_from(similarity_matrix, k=1)
        avg_similarity = similarity_matrix[mask].mean()
        
        return 1.0 - avg_similarity


class ModelEvaluator:
    """
    Comprehensive model evaluator for recommendation systems.
    
    Supports evaluation of:
    - Content-based filtering
    - Hybrid recommendations
    - Popularity baseline
    - Rating prediction (RMSE/MAE)
    """
    
    def __init__(self, games_df: pd.DataFrame):
        """
        Initialize evaluator.
        
        Args:
            games_df: DataFrame with game information
        """
        self.games_df = games_df
        self.metrics = EvaluationMetrics()
    
    def evaluate_rating_prediction(
        self,
        recommender,
        test_games: List[int],
        n_samples: int = 50
    ) -> Dict:
        """
        Evaluate rating prediction using RMSE and MAE.
        
        Uses quality_score as proxy for "predicted rating" and
        positive ratio as ground truth.
        
        Args:
            recommender: Recommender object
            test_games: List of game appids to test
            n_samples: Number of samples to evaluate
        
        Returns:
            Dictionary with RMSE and MAE scores
        """
        y_true = []
        y_pred = []
        
        sample_games = test_games[:n_samples]
        
        for game_appid in sample_games:
            if game_appid not in recommender.appid_to_idx:
                continue
            
            # Get actual rating
            game_data = self.games_df[self.games_df['appid'] == game_appid]
            if game_data.empty:
                continue
            
            actual_rating = self._get_positive_ratio(game_data.iloc[0])
            
            # Get recommendations and their predicted scores
            try:
                recs = recommender.hybrid_recommend(game_appid, n_recommendations=10)
                for rec in recs:
                    rec_game = self.games_df[self.games_df['appid'] == rec['appid']]
                    if not rec_game.empty:
                        # Use similarity as prediction proxy
                        predicted = rec.get('similarity', 0.5)
                        actual = self._get_positive_ratio(rec_game.iloc[0])
                        
                        y_true.append(actual)
                        y_pred.append(predicted)
            except Exception:
                continue
        
        if len(y_true) == 0:
            return {
                'rmse': 0.0,
                'mae': 0.0,
                'samples': 0
            }
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return {
            'rmse': self.metrics.rmse(y_true, y_pred),
            'mae': self.metrics.mae(y_true, y_pred),
            'samples': len(y_true)
        }
    
    def evaluate_content_based(
        self,
        recommender,
        test_games: List[int],
        k: int = 10
    ) -> Dict:
        """
        Evaluate content-based recommender
        
        Args:
            recommender: Recommender object with content_based_recommend method
            test_games: List of game appids to test on
            k: Number of recommendations
        
        Returns:
            Dictionary of evaluation metrics
        """
        precisions = []
        recalls = []
        ndcgs = []
        
        # For content-based, we use genre similarity as ground truth
        for game_appid in test_games:
            if game_appid not in recommender.appid_to_idx:
                continue
            
            game = self.games_df[self.games_df['appid'] == game_appid].iloc[0]
            game_genre = set(str(game['genres']).split(','))
            
            recommendations = recommender.content_based_recommend(game_appid, n_recommendations=k)
            recommended_ids = [r['appid'] for r in recommendations]
            
            # Get genres of recommended games
            relevant_games = set()
            for rec_id in recommended_ids:
                rec_game = self.games_df[self.games_df['appid'] == rec_id]
                if not rec_game.empty:
                    rec_genres = set(str(rec_game.iloc[0]['genres']).split(','))
                    if len(rec_genres.intersection(game_genre)) > 0:
                        relevant_games.add(rec_id)
            
            p_at_k = self.metrics.precision_at_k(relevant_games, recommended_ids, k)
            r_at_k = self.metrics.recall_at_k(relevant_games, recommended_ids, k)
            ndcg = self.metrics.ndcg_at_k(relevant_games, recommended_ids, k)
            
            precisions.append(p_at_k)
            recalls.append(r_at_k)
            ndcgs.append(ndcg)
        
        return {
            'model': 'Content-Based (Cosine Similarity)',
            'avg_precision_at_k': np.mean(precisions),
            'avg_recall_at_k': np.mean(recalls),
            'avg_ndcg_at_k': np.mean(ndcgs),
            'samples_tested': len(precisions),
            'rmse': self._calculate_model_rmse(recommender, test_games, 'content'),
            'mae': self._calculate_model_mae(recommender, test_games, 'content')
        }
    
    def evaluate_hybrid(
        self,
        recommender,
        test_games: List[int],
        k: int = 10,
        content_weight: float = 0.7,
        quality_weight: float = 0.3
    ) -> Dict:
        """
        Evaluate hybrid recommender
        
        Args:
            recommender: Recommender object with hybrid_recommend method
            test_games: List of game appids to test on
            k: Number of recommendations
            content_weight: Weight for content similarity
            quality_weight: Weight for quality score
        
        Returns:
            Dictionary of evaluation metrics
        """
        precisions = []
        recalls = []
        ndcgs = []
        
        for game_appid in test_games:
            if game_appid not in recommender.appid_to_idx:
                continue
            
            game = self.games_df[self.games_df['appid'] == game_appid].iloc[0]
            game_genre = set(str(game['genres']).split(','))
            
            recommendations = recommender.hybrid_recommend(
                game_appid,
                n_recommendations=k,
                content_weight=content_weight,
                quality_weight=quality_weight
            )
            recommended_ids = [r['appid'] for r in recommendations]
            
            relevant_games = set()
            for rec_id in recommended_ids:
                rec_game = self.games_df[self.games_df['appid'] == rec_id]
                if not rec_game.empty:
                    rec_genres = set(str(rec_game.iloc[0]['genres']).split(','))
                    if len(rec_genres.intersection(game_genre)) > 0:
                        relevant_games.add(rec_id)
            
            p_at_k = self.metrics.precision_at_k(relevant_games, recommended_ids, k)
            r_at_k = self.metrics.recall_at_k(relevant_games, recommended_ids, k)
            ndcg = self.metrics.ndcg_at_k(relevant_games, recommended_ids, k)
            
            precisions.append(p_at_k)
            recalls.append(r_at_k)
            ndcgs.append(ndcg)
        
        return {
            'model': 'Hybrid (Content + Quality)',
            'avg_precision_at_k': np.mean(precisions),
            'avg_recall_at_k': np.mean(recalls),
            'avg_ndcg_at_k': np.mean(ndcgs),
            'samples_tested': len(precisions),
            'rmse': self._calculate_model_rmse(recommender, test_games, 'hybrid'),
            'mae': self._calculate_model_mae(recommender, test_games, 'hybrid')
        }
    
    def evaluate_popularity_baseline(
        self,
        test_games: List[int],
        k: int = 10
    ) -> Dict:
        """
        Evaluate popularity-based baseline
        
        Args:
            test_games: List of game appids to test on
            k: Number of recommendations
        
        Returns:
            Dictionary of evaluation metrics
        """
        precisions = []
        recalls = []
        ndcgs = []
        
        # Get top-k games by quality score (or positive_ratings as fallback)
        if 'quality_score' in self.games_df.columns:
            top_games = self.games_df.nlargest(k, 'quality_score')['appid'].tolist()
        elif 'positive_ratings' in self.games_df.columns:
            top_games = self.games_df.nlargest(k, 'positive_ratings')['appid'].tolist()
        else:
            top_games = self.games_df.head(k)['appid'].tolist()
        
        for game_appid in test_games:
            game = self.games_df[self.games_df['appid'] == game_appid]
            if game.empty:
                continue
            
            game_genre = set(str(game.iloc[0]['genres']).split(','))
            
            relevant_games = set()
            for rec_id in top_games:
                rec_game = self.games_df[self.games_df['appid'] == rec_id]
                if not rec_game.empty:
                    rec_genres = set(str(rec_game.iloc[0]['genres']).split(','))
                    if len(rec_genres.intersection(game_genre)) > 0:
                        relevant_games.add(rec_id)
            
            p_at_k = self.metrics.precision_at_k(relevant_games, top_games, k)
            r_at_k = self.metrics.recall_at_k(relevant_games, top_games, k)
            ndcg = self.metrics.ndcg_at_k(relevant_games, top_games, k)
            
            precisions.append(p_at_k)
            recalls.append(r_at_k)
            ndcgs.append(ndcg)
        
        return {
            'model': 'Popularity Baseline',
            'avg_precision_at_k': np.mean(precisions) if precisions else 0,
            'avg_recall_at_k': np.mean(recalls) if recalls else 0,
            'avg_ndcg_at_k': np.mean(ndcgs) if ndcgs else 0,
            'samples_tested': len(precisions),
            'rmse': self._calculate_popularity_rmse(test_games),
            'mae': self._calculate_popularity_mae(test_games)
        }
    
    def evaluate_context_aware(
        self,
        recommender,
        test_games: List[int],
        k: int = 10
    ) -> Dict:
        """
        Evaluate context-aware recommender.
        
        Args:
            recommender: Recommender object with context_aware_recommend method
            test_games: List of game appids to test on
            k: Number of recommendations
        
        Returns:
            Dictionary of evaluation metrics
        """
        precisions = []
        recalls = []
        ndcgs = []
        
        # Sample context for evaluation
        sample_context = {
            'time_of_day': 'evening',
            'day_of_week': 'weekend',
            'session_length': 'medium',
            'preferred_genres': ['Action', 'Adventure', 'RPG'],
            'recently_viewed': [],
            'mood': 'relaxed'
        }
        
        for game_appid in test_games:
            if game_appid not in recommender.appid_to_idx:
                continue
            
            game = self.games_df[self.games_df['appid'] == game_appid].iloc[0]
            game_genre = set(str(game['genres']).split(','))
            
            # Update context with recently viewed
            sample_context['recently_viewed'] = [game_appid]
            
            try:
                recommendations = recommender.context_aware_recommend(
                    game_appid,
                    context=sample_context,
                    n_recommendations=k
                )
                recommended_ids = [r['appid'] for r in recommendations]
            except Exception:
                continue
            
            relevant_games = set()
            for rec_id in recommended_ids:
                rec_game = self.games_df[self.games_df['appid'] == rec_id]
                if not rec_game.empty:
                    rec_genres = set(str(rec_game.iloc[0]['genres']).split(','))
                    if len(rec_genres.intersection(game_genre)) > 0:
                        relevant_games.add(rec_id)
            
            p_at_k = self.metrics.precision_at_k(relevant_games, recommended_ids, k)
            r_at_k = self.metrics.recall_at_k(relevant_games, recommended_ids, k)
            ndcg = self.metrics.ndcg_at_k(relevant_games, recommended_ids, k)
            
            precisions.append(p_at_k)
            recalls.append(r_at_k)
            ndcgs.append(ndcg)
        
        return {
            'model': 'Context-Aware',
            'avg_precision_at_k': np.mean(precisions) if precisions else 0,
            'avg_recall_at_k': np.mean(recalls) if recalls else 0,
            'avg_ndcg_at_k': np.mean(ndcgs) if ndcgs else 0,
            'samples_tested': len(precisions),
            'rmse': self._calculate_model_rmse(recommender, test_games, 'context'),
            'mae': self._calculate_model_mae(recommender, test_games, 'context')
        }
    
    def _get_positive_ratio(self, game_row) -> float:
        """
        Calculate positive ratio from positive and negative ratings.
        
        Args:
            game_row: DataFrame row for a game
        
        Returns:
            Positive ratio as float (0-1)
        """
        pos = game_row.get('positive_ratings', 0)
        neg = game_row.get('negative_ratings', 0)
        total = pos + neg
        if total > 0:
            return pos / total
        return 0.5  # Default to neutral if no ratings
    
    def _calculate_model_rmse(
        self,
        recommender,
        test_games: List[int],
        model_type: str
    ) -> float:
        """
        Calculate RMSE for a specific model type.
        
        Args:
            recommender: Recommender object
            test_games: List of game appids to test
            model_type: 'content', 'hybrid', or 'context'
        
        Returns:
            RMSE score
        """
        squared_errors = []
        
        for game_appid in test_games:
            if game_appid not in recommender.appid_to_idx:
                continue
            
            game = self.games_df[self.games_df['appid'] == game_appid]
            if game.empty:
                continue
            
            actual_rating = self._get_positive_ratio(game.iloc[0])
            
            try:
                if model_type == 'content':
                    recs = recommender.content_based_recommend(game_appid, n_recommendations=5)
                elif model_type == 'hybrid':
                    recs = recommender.hybrid_recommend(game_appid, n_recommendations=5)
                elif model_type == 'context':
                    context = {'time_of_day': 'evening', 'preferred_genres': []}
                    recs = recommender.context_aware_recommend(game_appid, context, n_recommendations=5)
                else:
                    continue
                
                if recs:
                    # Use similarity score as predicted rating
                    predicted_rating = recs[0].get('similarity', 0.5)
                    squared_errors.append((actual_rating - predicted_rating) ** 2)
            except Exception:
                continue
        
        if squared_errors:
            return np.sqrt(np.mean(squared_errors))
        return 0.0
    
    def _calculate_model_mae(
        self,
        recommender,
        test_games: List[int],
        model_type: str
    ) -> float:
        """
        Calculate MAE for a specific model type.
        
        Args:
            recommender: Recommender object
            test_games: List of game appids to test
            model_type: 'content', 'hybrid', or 'context'
        
        Returns:
            MAE score
        """
        absolute_errors = []
        
        for game_appid in test_games:
            if game_appid not in recommender.appid_to_idx:
                continue
            
            game = self.games_df[self.games_df['appid'] == game_appid]
            if game.empty:
                continue
            
            actual_rating = self._get_positive_ratio(game.iloc[0])
            
            try:
                if model_type == 'content':
                    recs = recommender.content_based_recommend(game_appid, n_recommendations=5)
                elif model_type == 'hybrid':
                    recs = recommender.hybrid_recommend(game_appid, n_recommendations=5)
                elif model_type == 'context':
                    context = {'time_of_day': 'evening', 'preferred_genres': []}
                    recs = recommender.context_aware_recommend(game_appid, context, n_recommendations=5)
                else:
                    continue
                
                if recs:
                    # Use similarity score as predicted rating
                    predicted_rating = recs[0].get('similarity', 0.5)
                    absolute_errors.append(abs(actual_rating - predicted_rating))
            except Exception:
                continue
        
        if absolute_errors:
            return float(np.mean(absolute_errors))
        return 0.0
    
    def _calculate_popularity_rmse(self, test_games: List[int]) -> float:
        """
        Calculate RMSE for popularity baseline.
        
        Args:
            test_games: List of game appids to test
        
        Returns:
            RMSE score
        """
        squared_errors = []
        
        # Calculate average positive ratio across all games
        all_ratios = []
        for _, row in self.games_df.iterrows():
            all_ratios.append(self._get_positive_ratio(row))
        avg_rating = np.mean(all_ratios) if all_ratios else 0.5
        
        for game_appid in test_games:
            game = self.games_df[self.games_df['appid'] == game_appid]
            if game.empty:
                continue
            
            actual_rating = self._get_positive_ratio(game.iloc[0])
            squared_errors.append((actual_rating - avg_rating) ** 2)
        
        if squared_errors:
            return np.sqrt(np.mean(squared_errors))
        return 0.0
    
    def _calculate_popularity_mae(self, test_games: List[int]) -> float:
        """
        Calculate MAE for popularity baseline.
        
        Args:
            test_games: List of game appids to test
        
        Returns:
            MAE score
        """
        absolute_errors = []
        
        # Calculate average positive ratio across all games
        all_ratios = []
        for _, row in self.games_df.iterrows():
            all_ratios.append(self._get_positive_ratio(row))
        avg_rating = np.mean(all_ratios) if all_ratios else 0.5
        
        for game_appid in test_games:
            game = self.games_df[self.games_df['appid'] == game_appid]
            if game.empty:
                continue
            
            actual_rating = self._get_positive_ratio(game.iloc[0])
            absolute_errors.append(abs(actual_rating - avg_rating))
        
        if absolute_errors:
            return float(np.mean(absolute_errors))
        return 0.0
    
    def evaluate_all_models(
        self,
        recommender,
        test_games: List[int],
        k: int = 10
    ) -> Dict:
        """
        Run comprehensive evaluation on all models.
        
        Args:
            recommender: Recommender object
            test_games: List of game appids to test
            k: Number of recommendations
        
        Returns:
            Dictionary with all evaluation results including RMSE/MAE
        """
        results = {
            'ranking_metrics': [],
            'rating_metrics': None
        }
        
        # Evaluate ranking metrics for each model
        content_eval = self.evaluate_content_based(recommender, test_games, k)
        hybrid_eval = self.evaluate_hybrid(recommender, test_games, k)
        popularity_eval = self.evaluate_popularity_baseline(test_games, k)
        
        # Add Context-Aware evaluation if method exists
        try:
            context_eval = self.evaluate_context_aware(recommender, test_games, k)
            results['ranking_metrics'] = [content_eval, hybrid_eval, context_eval, popularity_eval]
        except Exception:
            results['ranking_metrics'] = [content_eval, hybrid_eval, popularity_eval]
        
        # Evaluate rating prediction metrics (overall)
        rating_eval = self.evaluate_rating_prediction(recommender, test_games)
        results['rating_metrics'] = rating_eval
        
        return results


print("EvaluationMetrics and ModelEvaluator classes ready!")
