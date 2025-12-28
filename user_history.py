"""
User History Management Module
==============================
Handles user browsing history, preferences, and context-aware features.
Supports MongoDB storage with JSON fallback.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib

# Try to import pymongo
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False


class UserHistory:
    """
    Manages user interaction history and preferences.
    
    Features:
    - Track viewed games with timestamps
    - Store search history
    - Save user preferences (genres, price range, etc.)
    - Context-aware data (time of day, session info)
    - MongoDB support with JSON fallback
    - Game history for personalized recommendations
    """
    
    STORAGE_DIR = Path("data/user_data")
    
    def __init__(
        self, 
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        mongodb_uri: Optional[str] = None,
        db_name: str = "steam_recommender"
    ):
        """
        Initialize user history manager.
        
        Args:
            user_id: Unique user identifier (legacy)
            username: Username from auth system
            mongodb_uri: MongoDB connection string
            db_name: Database name
        """
        self.username = username
        self.user_id = user_id or self._generate_session_id()
        self.use_mongodb = False
        self.db = None
        self.history_collection = None
        
        # Try MongoDB connection
        if mongodb_uri and MONGODB_AVAILABLE:
            try:
                self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=3000)
                self.client.admin.command('ping')
                self.db = self.client[db_name]
                self.history_collection = self.db["user_history"]
                self.use_mongodb = True
            except Exception:
                self.use_mongodb = False
        
        # JSON storage path
        identifier = self.username or self.user_id
        self.storage_path = self.STORAGE_DIR / f"{identifier}.json"
        
        # Initialize data structure
        self.data = {
            "user_id": self.user_id,
            "username": self.username,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "viewed_games": [],
            "selected_games": [],  # Games user selected for recommendations
            "search_history": [],
            "recommendations_clicked": [],
            "game_ratings": {},  # appid -> rating
            "preferences": {
                "favorite_genres": [],
                "price_range": {"min": 0, "max": 100},
                "preferred_playtime": "any",
                "exclude_tags": []
            },
            "device_config": {
                "device_type": "pc",
                "cpu": "",
                "ram_gb": 8,
                "storage_gb": 256,
                "gpu": "",
                "has_dedicated_gpu": False
            },
            "context": {
                "total_sessions": 0,
                "avg_session_duration": 0,
                "preferred_time": "any"
            }
        }
        
        # Ensure storage directory exists
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_data()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def _load_data(self) -> None:
        """Load user data from storage"""
        if self.use_mongodb and self.username:
            doc = self.history_collection.find_one({"username": self.username})
            if doc:
                doc.pop('_id', None)
                self._merge_data(doc)
        elif self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    self._merge_data(stored_data)
            except (json.JSONDecodeError, IOError):
                pass
    
    def _merge_data(self, stored_data: Dict) -> None:
        """Merge stored data with current data structure"""
        for key, value in stored_data.items():
            if key in self.data:
                if isinstance(value, dict) and isinstance(self.data[key], dict):
                    self.data[key].update(value)
                else:
                    self.data[key] = value
    
    def save(self) -> None:
        """Persist user data to storage"""
        self.data["last_active"] = datetime.now().isoformat()
        
        if self.use_mongodb and self.username:
            self.history_collection.update_one(
                {"username": self.username},
                {"$set": self.data},
                upsert=True
            )
        else:
            try:
                with open(self.storage_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
            except IOError as e:
                print(f"Warning: Could not save user data: {e}")
    
    # ==================== VIEW HISTORY ====================
    
    def add_viewed_game(self, game_info: Dict) -> None:
        """
        Record a viewed game.
        
        Args:
            game_info: Dictionary with game details (appid, name, genres, etc.)
        """
        view_record = {
            "appid": game_info.get("appid"),
            "name": game_info.get("name"),
            "genres": game_info.get("genres", ""),
            "price": game_info.get("price", 0),
            "timestamp": datetime.now().isoformat(),
            "context": self._get_current_context()
        }
        
        # Avoid duplicates in recent history
        self.data["viewed_games"] = [
            g for g in self.data["viewed_games"] 
            if g["appid"] != view_record["appid"]
        ][-49:] + [view_record]
        
        # Update genre preferences
        self._update_genre_preferences(game_info.get("genres", ""))
        
        self.save()
    
    def get_viewed_games(self, limit: int = 20) -> List[Dict]:
        """Get recently viewed games (newest first)"""
        return self.data["viewed_games"][-limit:][::-1]
    
    def get_viewed_appids(self) -> List[int]:
        """Get list of viewed game appids"""
        return [g["appid"] for g in self.data["viewed_games"]]
    
    # ==================== SELECTED GAMES (for recommendations) ====================
    
    def add_selected_game(self, game_info: Dict) -> None:
        """
        Add a game to selected games list (for recommendation basis).
        
        Args:
            game_info: Dictionary with game details
        """
        selected_record = {
            "appid": game_info.get("appid"),
            "name": game_info.get("name"),
            "genres": game_info.get("genres", ""),
            "price": game_info.get("price", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Avoid duplicates
        self.data["selected_games"] = [
            g for g in self.data["selected_games"]
            if g["appid"] != selected_record["appid"]
        ] + [selected_record]
        
        # Keep last 20 selected games
        self.data["selected_games"] = self.data["selected_games"][-20:]
        
        self.save()
    
    def remove_selected_game(self, appid: int) -> None:
        """Remove a game from selected list"""
        self.data["selected_games"] = [
            g for g in self.data["selected_games"]
            if g["appid"] != appid
        ]
        self.save()
    
    def get_selected_games(self) -> List[Dict]:
        """Get all selected games"""
        return self.data["selected_games"]
    
    def get_selected_appids(self) -> List[int]:
        """Get selected game appids"""
        return [g["appid"] for g in self.data["selected_games"]]
    
    def clear_selected_games(self) -> None:
        """Clear all selected games"""
        self.data["selected_games"] = []
        self.save()
    
    # ==================== GAME RATINGS ====================
    
    def rate_game(self, appid: int, rating: float) -> None:
        """
        Rate a game (1-5 stars).
        
        Args:
            appid: Game AppID
            rating: Rating from 1.0 to 5.0
        """
        self.data["game_ratings"][str(appid)] = {
            "rating": max(1.0, min(5.0, rating)),
            "timestamp": datetime.now().isoformat()
        }
        self.save()
    
    def get_game_rating(self, appid: int) -> Optional[float]:
        """Get rating for a specific game"""
        rating_data = self.data["game_ratings"].get(str(appid))
        return rating_data["rating"] if rating_data else None
    
    def get_all_ratings(self) -> Dict[int, float]:
        """Get all game ratings"""
        return {
            int(appid): data["rating"] 
            for appid, data in self.data["game_ratings"].items()
        }
    
    def get_highly_rated_games(self, min_rating: float = 4.0) -> List[int]:
        """Get appids of highly rated games"""
        return [
            int(appid) 
            for appid, data in self.data["game_ratings"].items()
            if data["rating"] >= min_rating
        ]
    
    # ==================== SEARCH HISTORY ====================
    
    def add_search(self, query: str) -> None:
        """Record a search query"""
        search_record = {
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
        
        if (not self.data["search_history"] or 
            self.data["search_history"][-1]["query"] != query):
            self.data["search_history"].append(search_record)
            self.data["search_history"] = self.data["search_history"][-50:]
        
        self.save()
    
    def get_search_history(self, limit: int = 10) -> List[str]:
        """Get recent search queries"""
        return [s["query"] for s in self.data["search_history"][-limit:][::-1]]
    
    # ==================== RECOMMENDATIONS TRACKING ====================
    
    def add_recommendation_click(self, source_appid: int, clicked_appid: int) -> None:
        """Track when user clicks a recommended game"""
        click_record = {
            "source_appid": source_appid,
            "clicked_appid": clicked_appid,
            "timestamp": datetime.now().isoformat()
        }
        
        self.data["recommendations_clicked"].append(click_record)
        self.data["recommendations_clicked"] = self.data["recommendations_clicked"][-100:]
        
        self.save()
    
    # ==================== PREFERENCES ====================
    
    def _update_genre_preferences(self, genres_str: str) -> None:
        """Update favorite genres based on viewing history"""
        if not genres_str:
            return
        
        genres = [g.strip() for g in str(genres_str).split(';')]
        favorite_genres = self.data["preferences"]["favorite_genres"]
        
        for genre in genres:
            if genre and genre not in favorite_genres:
                favorite_genres.append(genre)
        
        self.data["preferences"]["favorite_genres"] = favorite_genres[-10:]
    
    def set_price_range(self, min_price: float, max_price: float) -> None:
        """Set preferred price range"""
        self.data["preferences"]["price_range"] = {
            "min": min_price,
            "max": max_price
        }
        self.save()
    
    def get_price_range(self) -> Dict[str, float]:
        """Get preferred price range"""
        return self.data["preferences"]["price_range"]
    
    def set_playtime_preference(self, playtime: str) -> None:
        """Set preferred playtime category"""
        valid_options = ['short', 'medium', 'long', 'any']
        if playtime in valid_options:
            self.data["preferences"]["preferred_playtime"] = playtime
            self.save()
    
    def get_preferences(self) -> Dict:
        """Get all user preferences"""
        return self.data["preferences"]
    
    def get_favorite_genres(self) -> List[str]:
        """Get user's favorite genres"""
        return self.data["preferences"]["favorite_genres"]
    
    def set_favorite_genres(self, genres: List[str]) -> None:
        """Set favorite genres explicitly"""
        self.data["preferences"]["favorite_genres"] = genres[:10]
        self.save()
    
    # ==================== DEVICE CONFIG ====================
    
    def set_device_config(self, config: Dict) -> None:
        """
        Set device configuration.
        
        Args:
            config: Dict with device_type, cpu, ram_gb, storage_gb, gpu, has_dedicated_gpu
        """
        self.data["device_config"].update(config)
        self.save()
    
    def get_device_config(self) -> Dict:
        """Get device configuration"""
        return self.data["device_config"]
    
    # ==================== CONTEXT ====================
    
    def _get_current_context(self) -> Dict:
        """Get current context information"""
        now = datetime.now()
        hour = now.hour
        
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        return {
            "time_of_day": time_of_day,
            "is_weekend": now.weekday() >= 5,
            "hour": hour,
            "day_of_week": now.weekday()
        }
    
    def get_context(self) -> Dict:
        """Get current context for recommendations"""
        return self._get_current_context()
    
    def update_session_stats(self, session_duration_minutes: float) -> None:
        """Update session statistics"""
        ctx = self.data["context"]
        total_sessions = ctx["total_sessions"]
        avg_duration = ctx["avg_session_duration"]
        
        new_avg = (avg_duration * total_sessions + session_duration_minutes) / (total_sessions + 1)
        
        ctx["total_sessions"] = total_sessions + 1
        ctx["avg_session_duration"] = new_avg
        
        self.save()
    
    # ==================== ANALYTICS ====================
    
    def get_analytics(self) -> Dict:
        """Get user analytics summary"""
        viewed = self.data["viewed_games"]
        selected = self.data["selected_games"]
        ratings = self.data["game_ratings"]
        
        # Count genre frequencies
        genre_counts = {}
        for game in viewed:
            for genre in str(game.get("genres", "")).split(';'):
                genre = genre.strip()
                if genre:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate average rating
        rating_values = [r["rating"] for r in ratings.values()]
        avg_rating = sum(rating_values) / len(rating_values) if rating_values else 0
        
        # Price preferences
        prices = [g.get("price", 0) for g in viewed if g.get("price") is not None]
        avg_price = sum(prices) / len(prices) if prices else 0
        
        return {
            "total_games_viewed": len(viewed),
            "total_games_selected": len(selected),
            "total_searches": len(self.data["search_history"]),
            "recommendations_clicked": len(self.data["recommendations_clicked"]),
            "games_rated": len(ratings),
            "avg_rating_given": round(avg_rating, 2),
            "top_genres": top_genres,
            "avg_price_viewed": round(avg_price, 2),
            "sessions": self.data["context"]["total_sessions"],
            "favorite_genres": self.data["preferences"]["favorite_genres"],
            "device_config": self.data["device_config"]
        }
    
    def clear_history(self) -> None:
        """Clear all user history (keep preferences)"""
        self.data["viewed_games"] = []
        self.data["search_history"] = []
        self.data["recommendations_clicked"] = []
        self.save()
    
    def export_data(self) -> Dict:
        """Export all user data"""
        return self.data.copy()


class SessionManager:
    """
    Manages user sessions for Streamlit.
    """
    
    @staticmethod
    def get_or_create_history(
        session_state,
        username: Optional[str] = None,
        mongodb_uri: Optional[str] = None
    ) -> UserHistory:
        """
        Get existing UserHistory or create new one.
        
        Args:
            session_state: Streamlit session state
            username: Username from auth system
            mongodb_uri: MongoDB connection string
        
        Returns:
            UserHistory instance
        """
        # If username changed, create new history
        current_username = session_state.get('username')
        if 'user_history' in session_state:
            if username and session_state['user_history'].username != username:
                # Username changed, create new history
                session_state['user_history'] = UserHistory(
                    username=username,
                    mongodb_uri=mongodb_uri
                )
        else:
            # Create new history
            user_id = session_state.get('user_id')
            session_state['user_history'] = UserHistory(
                user_id=user_id,
                username=username,
                mongodb_uri=mongodb_uri
            )
            session_state['user_id'] = session_state['user_history'].user_id
            session_state['session_start'] = datetime.now()
        
        return session_state['user_history']
    
    @staticmethod
    def end_session(session_state) -> None:
        """Record session end and update statistics"""
        if 'user_history' in session_state and 'session_start' in session_state:
            duration = (datetime.now() - session_state['session_start']).total_seconds() / 60
            session_state['user_history'].update_session_stats(duration)


print("[OK] UserHistory and SessionManager classes ready!")
