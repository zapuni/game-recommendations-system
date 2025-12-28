"""
Recommender System Module
=========================
Multi-algorithm recommendation system with support for:
- Content-based filtering (TF-IDF, Sentence Transformers)
- Hybrid recommendations (content + quality)
- Popularity-based recommendations
- Context-aware recommendations
- Genre-based filtering
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import hashlib
import warnings

warnings.filterwarnings('ignore')

# Try to import sentence-transformers for advanced embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Note: sentence-transformers not installed. Using TF-IDF embeddings.")

# Cache directory for embeddings
CACHE_DIR = Path("data/embeddings_cache")


class EmbeddingManager:
    """
    Manages different embedding strategies for game content.
    
    Supports:
    - TF-IDF: Fast, lightweight text vectorization
    - Sentence Transformers (BERT): Semantic understanding
    - Caching: Save/load embeddings to avoid recomputation
    """
    
    def __init__(self, embedding_type: str = "tfidf"):
        """
        Initialize embedding manager.
        
        Args:
            embedding_type: 'tfidf' or 'bert'
        """
        self.embedding_type = embedding_type
        self.model = None
        self.embeddings = None
        
        if embedding_type == "bert" and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use a lightweight multilingual model
            self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        elif embedding_type == "bert":
            print("Warning: BERT requested but not available. Falling back to TF-IDF.")
            self.embedding_type = "tfidf"
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            Numpy array of embeddings
        """
        if self.embedding_type == "bert" and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Truncate texts to avoid memory issues
            truncated_texts = [t[:500] if t else "" for t in texts]
            self.embeddings = self.model.encode(
                truncated_texts, 
                show_progress_bar=True,
                batch_size=32
            )
        else:
            # TF-IDF fallback
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.embeddings = tfidf.fit_transform(texts).toarray()
        
        return self.embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if self.embeddings is not None:
            return self.embeddings.shape[1]
        return 0


class EmbeddingCache:
    """
    Manages caching of embedding matrices to disk.
    
    Saves embeddings as .npy files to avoid recomputation on app restart.
    Uses hash of game appids to detect data changes.
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_data_hash(self, appids: List[int]) -> str:
        """
        Generate hash from appids to detect data changes.
        
        Args:
            appids: List of game appids
        
        Returns:
            MD5 hash string (first 8 chars)
        """
        appids_str = ",".join(map(str, sorted(appids)))
        return hashlib.md5(appids_str.encode()).hexdigest()[:8]
    
    def _get_cache_path(self, embedding_type: str, data_hash: str) -> Path:
        """
        Get cache file path for given embedding type and data hash.
        
        Args:
            embedding_type: 'tfidf' or 'bert'
            data_hash: Hash of the data
        
        Returns:
            Path to cache file
        """
        filename = f"embeddings_{embedding_type}_{data_hash}.npy"
        return self.cache_dir / filename
    
    def _get_similarity_cache_path(self, embedding_type: str, data_hash: str) -> Path:
        """
        Get similarity matrix cache file path.
        
        Args:
            embedding_type: 'tfidf' or 'bert'
            data_hash: Hash of the data
        
        Returns:
            Path to similarity cache file
        """
        filename = f"similarity_{embedding_type}_{data_hash}.npy"
        return self.cache_dir / filename
    
    def exists(self, embedding_type: str, appids: List[int]) -> bool:
        """
        Check if cache exists for given embedding type and data.
        
        Args:
            embedding_type: 'tfidf' or 'bert'
            appids: List of game appids
        
        Returns:
            True if cache exists and is valid
        """
        data_hash = self._get_data_hash(appids)
        cache_path = self._get_cache_path(embedding_type, data_hash)
        return cache_path.exists()
    
    def load(self, embedding_type: str, appids: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load embeddings and similarity matrix from cache.
        
        Args:
            embedding_type: 'tfidf' or 'bert'
            appids: List of game appids
        
        Returns:
            Tuple of (embeddings, similarity_matrix) or (None, None) if not found
        """
        data_hash = self._get_data_hash(appids)
        embeddings_path = self._get_cache_path(embedding_type, data_hash)
        similarity_path = self._get_similarity_cache_path(embedding_type, data_hash)
        
        try:
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
                
                # Validate size matches
                if embeddings.shape[0] != len(appids):
                    print(f"[CACHE] Size mismatch: cache has {embeddings.shape[0]}, need {len(appids)}")
                    return None, None
                
                # Load similarity matrix if exists
                similarity_matrix = None
                if similarity_path.exists():
                    similarity_matrix = np.load(similarity_path)
                
                print(f"[CACHE] Loaded {embedding_type.upper()} embeddings from cache ({embeddings.shape})")
                return embeddings, similarity_matrix
        except Exception as e:
            print(f"[CACHE] Error loading cache: {e}")
        
        return None, None
    
    def save(self, embedding_type: str, appids: List[int], 
             embeddings: np.ndarray, similarity_matrix: Optional[np.ndarray] = None) -> bool:
        """
        Save embeddings and optionally similarity matrix to cache.
        
        Args:
            embedding_type: 'tfidf' or 'bert'
            appids: List of game appids
            embeddings: Embedding matrix to save
            similarity_matrix: Optional similarity matrix to save
        
        Returns:
            True if saved successfully
        """
        data_hash = self._get_data_hash(appids)
        embeddings_path = self._get_cache_path(embedding_type, data_hash)
        
        try:
            np.save(embeddings_path, embeddings)
            print(f"[CACHE] Saved {embedding_type.upper()} embeddings to {embeddings_path}")
            
            # Save similarity matrix too
            if similarity_matrix is not None:
                similarity_path = self._get_similarity_cache_path(embedding_type, data_hash)
                np.save(similarity_path, similarity_matrix)
                print(f"[CACHE] Saved similarity matrix to {similarity_path}")
            
            return True
        except Exception as e:
            print(f"[CACHE] Error saving cache: {e}")
            return False
    
    def clear(self, embedding_type: Optional[str] = None) -> int:
        """
        Clear cache files.
        
        Args:
            embedding_type: Optional - clear only this type, or all if None
        
        Returns:
            Number of files deleted
        """
        count = 0
        pattern = f"*_{embedding_type}_*.npy" if embedding_type else "*.npy"
        
        for file in self.cache_dir.glob(pattern):
            file.unlink()
            count += 1
        
        print(f"[CACHE] Cleared {count} cache files")
        return count
    
    def get_cache_info(self) -> Dict:
        """
        Get information about cached files.
        
        Returns:
            Dictionary with cache statistics
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'files': [],
            'total_size_mb': 0
        }
        
        for file in self.cache_dir.glob("*.npy"):
            size_mb = file.stat().st_size / (1024 * 1024)
            info['files'].append({
                'name': file.name,
                'size_mb': round(size_mb, 2)
            })
            info['total_size_mb'] += size_mb
        
        info['total_size_mb'] = round(info['total_size_mb'], 2)
        return info


class RecommenderSystem:
    """
    Unified recommender system with multiple algorithms.
    
    Algorithms:
    - Content-based: Cosine similarity on game features
    - Hybrid: Weighted combination of content + quality
    - Popularity: Top-rated games by quality score
    - Context-aware: Personalized based on user context
    - Genre-based: Filter by genre with quality ranking
    """
    
    def __init__(self, games_df: pd.DataFrame, embedding_type: str = "bert"):
        """
        Initialize recommender system.
        
        Args:
            games_df: DataFrame with game information
            embedding_type: 'tfidf' or 'bert' for embeddings
        """
        self.games_df = games_df.copy()
        self.appid_to_idx = {appid: idx for idx, appid in enumerate(games_df['appid'])}
        self.idx_to_appid = {idx: appid for appid, idx in self.appid_to_idx.items()}
        
        self.content_matrix = None
        self.similarity_matrix = None
        self.svd_matrix = None
        self.embedding_manager = EmbeddingManager(embedding_type)
        self.embedding_type = embedding_type
        
    def set_content_features(self, feature_matrix: np.ndarray):
        """Set content-based feature matrix"""
        self.content_matrix = feature_matrix
        self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self):
        """Compute cosine similarity matrix"""
        if self.content_matrix is not None:
            self.similarity_matrix = cosine_similarity(self.content_matrix)
    
    def content_based_recommend(
        self,
        game_appid: int,
        n_recommendations: int = 10,
        exclude_game: bool = True
    ) -> List[Dict]:
        """
        Content-based recommendations using cosine similarity
        
        Args:
            game_appid: AppID of the reference game
            n_recommendations: Number of games to recommend
            exclude_game: Whether to exclude the input game from results
        
        Returns:
            List of recommended games with scores
        """
        if self.similarity_matrix is None:
            raise ValueError("Content features not set. Call set_content_features first.")
        
        if game_appid not in self.appid_to_idx:
            return []
        
        idx = self.appid_to_idx[game_appid]
        similarities = self.similarity_matrix[idx]
        
        # Get top similar games
        if exclude_game:
            similarities = similarities.copy()
            similarities[idx] = -1
        
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for rank, top_idx in enumerate(top_indices, 1):
            appid = self.idx_to_appid[top_idx]
            game_info = self.games_df[self.games_df['appid'] == appid].iloc[0]
            
            recommendations.append({
                'rank': rank,
                'appid': appid,
                'name': game_info['name'],
                'similarity_score': float(similarities[top_idx]),
                'price': game_info['price'],
                'rating': game_info['rating_score'],
                'genres': game_info['genres'],
                'header_image': game_info.get('header_image', '')
            })
        
        return recommendations
    
    def hybrid_recommend(
        self,
        game_appid: int,
        n_recommendations: int = 10,
        content_weight: float = 0.7,
        quality_weight: float = 0.3
    ) -> List[Dict]:
        """
        Hybrid recommendations combining content similarity with quality metrics
        
        Args:
            game_appid: AppID of the reference game
            n_recommendations: Number of games to recommend
            content_weight: Weight for content similarity (0-1)
            quality_weight: Weight for quality score (0-1)
        
        Returns:
            List of recommended games with hybrid scores
        """
        if self.similarity_matrix is None:
            raise ValueError("Content features not set.")
        
        if game_appid not in self.appid_to_idx:
            return []
        
        idx = self.appid_to_idx[game_appid]
        content_scores = self.similarity_matrix[idx].copy()
        content_scores[idx] = -1  # Exclude input game
        
        # Normalize content scores
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-10)
        
        # Get quality scores
        quality_scores = self.games_df['quality_score'].values
        quality_scores = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-10)
        
        # Hybrid score
        hybrid_scores = (
            content_weight * content_scores +
            quality_weight * quality_scores
        )
        
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for rank, top_idx in enumerate(top_indices, 1):
            appid = self.idx_to_appid[top_idx]
            game_info = self.games_df[self.games_df['appid'] == appid].iloc[0]
            
            recommendations.append({
                'rank': rank,
                'appid': appid,
                'name': game_info['name'],
                'hybrid_score': float(hybrid_scores[top_idx]),
                'content_score': float(content_scores[top_idx]),
                'quality_score': float(game_info['quality_score']),
                'price': game_info['price'],
                'rating': game_info['rating_score'],
                'genres': game_info['genres'],
                'header_image': game_info.get('header_image', '')
            })
        
        return recommendations
    
    def popularity_based_recommend(
        self,
        n_recommendations: int = 10,
        exclude_free: bool = False
    ) -> List[Dict]:
        """
        Top games by popularity and quality
        
        Args:
            n_recommendations: Number of games to recommend
            exclude_free: Exclude free games
        
        Returns:
            List of top games
        """
        df = self.games_df.copy()
        
        if exclude_free:
            df = df[df['price'] > 0]
        
        # Sort by quality score (rating * popularity)
        df = df.sort_values('quality_score', ascending=False).head(n_recommendations)
        
        recommendations = []
        for rank, (_, game) in enumerate(df.iterrows(), 1):
            recommendations.append({
                'rank': rank,
                'appid': int(game['appid']),
                'name': game['name'],
                'quality_score': float(game['quality_score']),
                'rating': float(game['rating_score']),
                'popularity_score': float(game['popularity_score']),
                'price': float(game['price']),
                'genres': game['genres'],
                'header_image': game.get('header_image', '')
            })
        
        return recommendations
    
    def genre_based_recommend(
        self,
        genre: str,
        n_recommendations: int = 10
    ) -> List[Dict]:
        """
        Recommend top games in a specific genre
        
        Args:
            genre: Genre name
            n_recommendations: Number of games to recommend
        
        Returns:
            List of top games in genre
        """
        df = self.games_df.copy()
        
        # Filter by genre
        df = df[df['genres'].str.contains(genre, case=False, na=False)]
        
        if len(df) == 0:
            return []
        
        # Sort by quality score
        df = df.sort_values('quality_score', ascending=False).head(n_recommendations)
        
        recommendations = []
        for rank, (_, game) in enumerate(df.iterrows(), 1):
            recommendations.append({
                'rank': rank,
                'appid': int(game['appid']),
                'name': game['name'],
                'quality_score': float(game['quality_score']),
                'rating': float(game['rating_score']),
                'price': float(game['price']),
                'genres': game['genres'],
                'header_image': game.get('header_image', '')
            })
        
        return recommendations
    
    def search_games(
        self,
        query: str,
        n_results: int = 20
    ) -> List[Dict]:
        """
        Search for games by name
        
        Args:
            query: Search query
            n_results: Number of results
        
        Returns:
            List of matching games
        """
        df = self.games_df.copy()
        
        # Case-insensitive search
        mask = df['name'].str.contains(query, case=False, na=False)
        results = df[mask].head(n_results)
        
        games = []
        for _, game in results.iterrows():
            games.append({
                'appid': int(game['appid']),
                'name': game['name'],
                'rating': float(game['rating_score']),
                'price': float(game['price']),
                'genres': game['genres'],
                'header_image': game.get('header_image', '')
            })
        
        return games
    
    def get_game_details(self, appid: int) -> Dict:
        """Get detailed information about a game"""
        game = self.games_df[self.games_df['appid'] == appid]
        
        if game.empty:
            return None
        
        game = game.iloc[0]
        
        return {
            'appid': int(game['appid']),
            'name': game['name'],
            'price': float(game['price']),
            'rating': float(game['rating_score']),
            'positive_ratings': int(game['positive_ratings']),
            'negative_ratings': int(game['negative_ratings']),
            'total_ratings': int(game['total_ratings']),
            'average_playtime': int(game['average_playtime']),
            'genres': game['genres'],
            'categories': game['categories'],
            'release_date': str(game['release_date'])[:10],
            'developer': game['developer'],
            'publisher': game['publisher'],
            'description': game['short_description'],
            'header_image': game.get('header_image', ''),
            'quality_score': float(game['quality_score']),
            'popularity_score': float(game['popularity_score'])
        }
    
    # ==================== CONTEXT-AWARE RECOMMENDATIONS ====================
    
    def context_aware_recommend(
        self,
        game_appid: Optional[int] = None,
        user_context: Optional[Dict] = None,
        user_preferences: Optional[Dict] = None,
        viewed_games: Optional[List[int]] = None,
        n_recommendations: int = 10
    ) -> List[Dict]:
        """
        Context-aware recommendations based on user context and history.
        
        Considers:
        - Time of day (casual games in evening, etc.)
        - User's price preferences
        - Previously viewed games (for diversity)
        - Favorite genres from history
        - Playtime preferences
        
        Args:
            game_appid: Optional reference game for similarity
            user_context: Dict with time_of_day, is_weekend, etc.
            user_preferences: Dict with price_range, favorite_genres, etc.
            viewed_games: List of previously viewed appids to exclude/diversify
            n_recommendations: Number of recommendations
        
        Returns:
            List of context-aware recommendations
        """
        df = self.games_df.copy()
        
        # Default context and preferences
        if user_context is None:
            user_context = {}
        if user_preferences is None:
            user_preferences = {}
        if viewed_games is None:
            viewed_games = []
        
        # ---- Apply context filters ----
        
        # Price filter
        price_range = user_preferences.get('price_range', {})
        min_price = price_range.get('min', 0)
        max_price = price_range.get('max', 1000)
        df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
        
        # Playtime preference
        playtime_pref = user_preferences.get('preferred_playtime', 'any')
        if playtime_pref == 'short':
            df = df[df['average_playtime'] <= 60]
        elif playtime_pref == 'medium':
            df = df[(df['average_playtime'] > 60) & (df['average_playtime'] <= 300)]
        elif playtime_pref == 'long':
            df = df[df['average_playtime'] > 300]
        
        # Time-based adjustments
        time_of_day = user_context.get('time_of_day', 'any')
        is_weekend = user_context.get('is_weekend', False)
        
        # Boost certain genres based on time
        time_genre_boost = {
            'morning': ['Puzzle', 'Casual', 'Strategy'],
            'afternoon': ['Action', 'Adventure', 'RPG'],
            'evening': ['RPG', 'Adventure', 'Simulation'],
            'night': ['Horror', 'Adventure', 'Indie']
        }
        
        # Calculate scores
        scores = np.zeros(len(df))
        df_reset = df.reset_index(drop=True)
        
        # Base: quality score
        quality_scores = df_reset['quality_score'].values
        quality_norm = (quality_scores - quality_scores.min()) / (quality_scores.max() - quality_scores.min() + 1e-10)
        scores += 0.4 * quality_norm
        
        # Content similarity if reference game provided
        if game_appid is not None and self.similarity_matrix is not None:
            if game_appid in self.appid_to_idx:
                ref_idx = self.appid_to_idx[game_appid]
                for i, row in df_reset.iterrows():
                    if row['appid'] in self.appid_to_idx:
                        game_idx = self.appid_to_idx[row['appid']]
                        scores[i] += 0.3 * self.similarity_matrix[ref_idx][game_idx]
        
        # Genre preference boost
        favorite_genres = user_preferences.get('favorite_genres', [])
        for i, row in df_reset.iterrows():
            game_genres = str(row['genres']).lower()
            for fav_genre in favorite_genres:
                if fav_genre.lower() in game_genres:
                    scores[i] += 0.1
        
        # Time-based genre boost
        if time_of_day in time_genre_boost:
            boost_genres = time_genre_boost[time_of_day]
            for i, row in df_reset.iterrows():
                game_genres = str(row['genres'])
                for boost_genre in boost_genres:
                    if boost_genre in game_genres:
                        scores[i] += 0.05
        
        # Weekend boost for longer games
        if is_weekend:
            for i, row in df_reset.iterrows():
                if row['average_playtime'] > 120:
                    scores[i] += 0.05
        
        # Diversity: penalize already viewed games
        for i, row in df_reset.iterrows():
            if row['appid'] in viewed_games:
                scores[i] -= 0.5
        
        # Get top recommendations
        df_reset['context_score'] = scores
        df_sorted = df_reset.sort_values('context_score', ascending=False).head(n_recommendations)
        
        recommendations = []
        for rank, (_, game) in enumerate(df_sorted.iterrows(), 1):
            recommendations.append({
                'rank': rank,
                'appid': int(game['appid']),
                'name': game['name'],
                'context_score': float(game['context_score']),
                'quality_score': float(game['quality_score']),
                'price': float(game['price']),
                'rating': float(game['rating_score']),
                'genres': game['genres'],
                'average_playtime': int(game['average_playtime']),
                'header_image': game.get('header_image', '')
            })
        
        return recommendations
    
    def get_personalized_recommendations(
        self,
        user_history,
        n_recommendations: int = 10
    ) -> List[Dict]:
        """
        Get personalized recommendations based on user history object.
        
        Args:
            user_history: UserHistory object with browsing data
            n_recommendations: Number of recommendations
        
        Returns:
            List of personalized recommendations
        """
        # Get user data from history
        preferences = user_history.get_preferences()
        context = user_history.get_context()
        viewed = user_history.get_viewed_appids()
        
        # Get last viewed game for similarity
        recent_views = user_history.get_viewed_games(limit=5)
        reference_game = recent_views[0]['appid'] if recent_views else None
        
        return self.context_aware_recommend(
            game_appid=reference_game,
            user_context=context,
            user_preferences=preferences,
            viewed_games=viewed,
            n_recommendations=n_recommendations
        )
    
    def create_advanced_embeddings(self, use_bert: bool = False, use_cache: bool = True) -> np.ndarray:
        """
        Create advanced embeddings for all games with caching support.
        
        Args:
            use_bert: Whether to use BERT-based embeddings
            use_cache: Whether to use cached embeddings if available
        
        Returns:
            Embedding matrix
        """
        embedding_type = "bert" if use_bert else "tfidf"
        appids = self.games_df['appid'].tolist()
        
        # Initialize cache
        cache = EmbeddingCache()
        
        # Try to load from cache first
        if use_cache:
            cached_embeddings, cached_similarity = cache.load(embedding_type, appids)
            
            if cached_embeddings is not None:
                self.content_matrix = cached_embeddings
                self.embedding_manager = EmbeddingManager(embedding_type)
                self.embedding_manager.embeddings = cached_embeddings
                
                if cached_similarity is not None:
                    self.similarity_matrix = cached_similarity
                    print(f"[CACHE] Loaded similarity matrix from cache")
                else:
                    self._compute_similarity_matrix()
                    # Save the computed similarity matrix
                    cache.save(embedding_type, appids, cached_embeddings, self.similarity_matrix)
                
                return cached_embeddings
        
        # Cache miss - compute embeddings
        print(f"[INFO] Computing {embedding_type.upper()} embeddings...")
        
        # Combine text features
        texts = []
        for _, row in self.games_df.iterrows():
            text = f"{row.get('genres', '')} {row.get('categories', '')} {row.get('short_description', '')}"
            texts.append(text[:500])  # Truncate to avoid memory issues
        
        self.embedding_manager = EmbeddingManager(embedding_type)
        embeddings = self.embedding_manager.fit_transform(texts)
        
        self.content_matrix = embeddings
        self._compute_similarity_matrix()
        
        # Save to cache
        if use_cache:
            cache.save(embedding_type, appids, embeddings, self.similarity_matrix)
        
        return embeddings
    
    def clear_embedding_cache(self, embedding_type: Optional[str] = None) -> int:
        """
        Clear cached embeddings.
        
        Args:
            embedding_type: 'tfidf', 'bert', or None for all
        
        Returns:
            Number of files deleted
        """
        cache = EmbeddingCache()
        return cache.clear(embedding_type)
    
    def get_cache_info(self) -> Dict:
        """
        Get information about cached embeddings.
        
        Returns:
            Dictionary with cache statistics
        """
        cache = EmbeddingCache()
        return cache.get_cache_info()


print("RecommenderSystem class ready!")
