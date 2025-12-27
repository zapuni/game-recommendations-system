"""
Context-Aware Recommendation Module
====================================
Advanced recommendation system that considers:
- User's device configuration and compatibility
- Time of day preferences
- Location context
- Weather conditions
- User mood
- Previously selected games for recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum

from device_config import (
    DeviceConfig, 
    DeviceCompatibilityChecker,
    GameRequirements
)


class TimeContext(Enum):
    """Time of day context"""
    MORNING = "morning"      # 6-12
    AFTERNOON = "afternoon"  # 12-18
    EVENING = "evening"      # 18-24
    NIGHT = "night"          # 0-6


class DeviceType(Enum):
    """Device types"""
    PC = "pc"
    LAPTOP = "laptop"
    GAMING_LAPTOP = "gaming_laptop"
    MOBILE = "mobile"
    TABLET = "tablet"
    CONSOLE = "console"
    STEAM_DECK = "steam_deck"
    VR = "vr"


class ContextAwareRecommender:
    """
    Context-aware recommendation system that considers multiple factors:
    - Device configuration and game compatibility
    - Time of day
    - Location
    - Weather
    - User mood
    - Previously selected games
    """
    
    def __init__(self, games_df: pd.DataFrame, recommender):
        """
        Initialize Context-Aware Recommender.
        
        Args:
            games_df: DataFrame with game information
            recommender: RecommenderSystem instance
        """
        self.games_df = games_df
        self.recommender = recommender
        
        # Context weights
        self.context_weights = {
            'content_score': 0.35,
            'device_compatibility': 0.25,
            'time_of_day': 0.10,
            'location': 0.10,
            'weather': 0.05,
            'user_mood': 0.10,
            'history_bonus': 0.05
        }
    
    def get_time_context(self, hour: int = None) -> TimeContext:
        """Get time context from hour"""
        if hour is None:
            hour = datetime.now().hour
        
        if 6 <= hour < 12:
            return TimeContext.MORNING
        elif 12 <= hour < 18:
            return TimeContext.AFTERNOON
        elif 18 <= hour < 24:
            return TimeContext.EVENING
        else:
            return TimeContext.NIGHT
    
    def get_time_genre_preferences(self, context: TimeContext) -> Dict[str, float]:
        """
        Get genre preferences based on time of day.
        
        Returns:
            Dictionary of genre -> bonus multiplier
        """
        preferences = {
            TimeContext.MORNING: {
                'Puzzle': 1.3, 'Strategy': 1.2, 'Casual': 1.2,
                'Simulation': 1.1, 'Educational': 1.2,
                'Action': 0.8, 'Horror': 0.6
            },
            TimeContext.AFTERNOON: {
                'Action': 1.1, 'Adventure': 1.1, 'RPG': 1.1,
                'Sports': 1.2, 'Racing': 1.2, 'Shooter': 1.1
            },
            TimeContext.EVENING: {
                'RPG': 1.3, 'Adventure': 1.2, 'Action': 1.1,
                'Story': 1.2, 'Simulation': 1.1, 'Indie': 1.1
            },
            TimeContext.NIGHT: {
                'Horror': 1.4, 'Indie': 1.2, 'Casual': 1.2,
                'Relaxing': 1.3, 'Visual Novel': 1.2,
                'Action': 0.8, 'Sports': 0.7
            }
        }
        return preferences.get(context, {})
    
    def adjust_by_location(self, score: float, location: str, game_genre: str) -> float:
        """
        Adjust score based on location.
        
        Args:
            score: Base score
            location: Location (home, work, public, travel)
            game_genre: Game genre
        
        Returns:
            Adjusted score
        """
        location_adjustments = {
            'home': {'multiplier': 1.0, 'blocked': []},
            'work': {
                'multiplier': 0.6, 
                'blocked': ['Horror', 'Action', 'Shooter', 'Violence'],
                'boosted': ['Puzzle', 'Casual', 'Strategy']
            },
            'public': {
                'multiplier': 0.8,
                'blocked': ['Horror', 'Adult'],
                'boosted': ['Casual', 'Puzzle', 'Card']
            },
            'travel': {
                'multiplier': 0.9,
                'blocked': ['VR'],
                'boosted': ['Casual', 'Puzzle', 'Mobile-friendly', 'Indie']
            }
        }
        
        loc_config = location_adjustments.get(location, {'multiplier': 1.0, 'blocked': []})
        
        # Check if genre is blocked
        for blocked in loc_config.get('blocked', []):
            if blocked.lower() in game_genre.lower():
                return score * 0.3
        
        # Check if genre is boosted
        for boosted in loc_config.get('boosted', []):
            if boosted.lower() in game_genre.lower():
                return score * 1.2
        
        return score * loc_config['multiplier']
    
    def adjust_by_weather(self, score: float, weather: str, game_genre: str) -> float:
        """
        Adjust score based on weather.
        
        Args:
            score: Base score
            weather: Weather condition (sunny, rainy, snowy, cloudy, stormy)
            game_genre: Game genre
        
        Returns:
            Adjusted score
        """
        weather_preferences = {
            'sunny': {
                'boosted': ['Sports', 'Racing', 'Adventure', 'Casual'],
                'reduced': ['Horror', 'Dark']
            },
            'rainy': {
                'boosted': ['Puzzle', 'Strategy', 'RPG', 'Simulation', 'Story'],
                'reduced': ['Sports', 'Racing']
            },
            'snowy': {
                'boosted': ['Adventure', 'RPG', 'Cozy', 'Casual', 'Simulation'],
                'reduced': []
            },
            'cloudy': {
                'boosted': ['Indie', 'Strategy', 'Simulation'],
                'reduced': []
            },
            'stormy': {
                'boosted': ['Horror', 'Action', 'Adventure', 'Dramatic'],
                'reduced': ['Casual', 'Sports']
            }
        }
        
        weather_config = weather_preferences.get(weather, {})
        
        for boosted in weather_config.get('boosted', []):
            if boosted.lower() in game_genre.lower():
                return score * 1.15
        
        for reduced in weather_config.get('reduced', []):
            if reduced.lower() in game_genre.lower():
                return score * 0.9
        
        return score
    
    def adjust_by_mood(self, score: float, mood: str, game_genre: str) -> float:
        """
        Adjust score based on user mood.
        
        Args:
            score: Base score
            mood: User mood (happy, sad, angry, tired, excited, relaxed, stressed)
            game_genre: Game genre
        
        Returns:
            Adjusted score
        """
        mood_preferences = {
            'happy': {
                'boosted': ['Casual', 'Party', 'Comedy', 'Colorful', 'Adventure'],
                'reduced': ['Horror', 'Dark', 'Difficult']
            },
            'sad': {
                'boosted': ['Story', 'Adventure', 'Emotional', 'Music', 'Relaxing'],
                'reduced': ['Horror', 'Action', 'Competitive']
            },
            'angry': {
                'boosted': ['Action', 'Fighting', 'Shooter', 'Competitive'],
                'reduced': ['Puzzle', 'Casual', 'Slow']
            },
            'tired': {
                'boosted': ['Casual', 'Relaxing', 'Puzzle', 'Simple', 'Idle'],
                'reduced': ['Action', 'Horror', 'Competitive', 'Complex']
            },
            'excited': {
                'boosted': ['Action', 'Adventure', 'Racing', 'Sports', 'Multiplayer'],
                'reduced': ['Slow', 'Puzzle']
            },
            'relaxed': {
                'boosted': ['Simulation', 'Casual', 'Sandbox', 'Building', 'Farming'],
                'reduced': ['Horror', 'Stressful', 'Competitive']
            },
            'stressed': {
                'boosted': ['Casual', 'Relaxing', 'Simple', 'Music', 'Zen'],
                'reduced': ['Horror', 'Difficult', 'Competitive', 'Complex']
            }
        }
        
        mood_config = mood_preferences.get(mood, {})
        
        for boosted in mood_config.get('boosted', []):
            if boosted.lower() in game_genre.lower():
                return score * 1.25
        
        for reduced in mood_config.get('reduced', []):
            if reduced.lower() in game_genre.lower():
                return score * 0.7
        
        return score
    
    def context_aware_recommend(
        self,
        game_appid: Optional[int] = None,
        selected_games: Optional[List[int]] = None,
        n_recommendations: int = 10,
        device_config: Optional[Dict] = None,
        time_of_day: str = None,
        location: str = 'home',
        weather: str = 'sunny',
        mood: str = 'relaxed'
    ) -> List[Dict]:
        """
        Generate context-aware recommendations.
        
        Args:
            game_appid: Optional reference game AppID
            selected_games: List of previously selected game AppIDs
            n_recommendations: Number of recommendations
            device_config: User's device configuration dict
            time_of_day: Time context (morning, afternoon, evening, night)
            location: Location (home, work, public, travel)
            weather: Weather (sunny, rainy, snowy, cloudy, stormy)
            mood: User mood (happy, sad, angry, tired, excited, relaxed, stressed)
        
        Returns:
            List of recommended games with context-adjusted scores
        """
        # Initialize device compatibility checker
        if device_config:
            config = DeviceConfig.from_dict(device_config)
            compatibility_checker = DeviceCompatibilityChecker(config)
        else:
            compatibility_checker = None
        
        # Get base recommendations
        if game_appid:
            base_recs = self.recommender.hybrid_recommend(
                game_appid,
                n_recommendations=n_recommendations * 3
            )
        elif selected_games and len(selected_games) > 0:
            # Aggregate recommendations from selected games
            all_recs = {}
            for appid in selected_games[-5:]:  # Use last 5 selected games
                try:
                    recs = self.recommender.hybrid_recommend(
                        appid,
                        n_recommendations=n_recommendations * 2
                    )
                    for rec in recs:
                        if rec['appid'] not in selected_games:  # Don't recommend selected games
                            if rec['appid'] not in all_recs:
                                all_recs[rec['appid']] = rec
                                all_recs[rec['appid']]['recommendation_count'] = 1
                            else:
                                # Boost score for games recommended multiple times
                                all_recs[rec['appid']]['hybrid_score'] += rec['hybrid_score']
                                all_recs[rec['appid']]['recommendation_count'] += 1
                except Exception:
                    continue
            
            # Normalize scores
            base_recs = list(all_recs.values())
            for rec in base_recs:
                rec['hybrid_score'] /= rec.get('recommendation_count', 1)
            
            base_recs.sort(key=lambda x: x['hybrid_score'], reverse=True)
        else:
            # Fallback to popularity-based
            base_recs = self.recommender.popularity_based_recommend(
                n_recommendations=n_recommendations * 3
            )
        
        # Get time context
        if time_of_day is None:
            time_context = self.get_time_context()
        else:
            try:
                time_context = TimeContext(time_of_day)
            except ValueError:
                time_context = TimeContext.AFTERNOON
        
        time_genre_prefs = self.get_time_genre_preferences(time_context)
        
        # Apply context adjustments
        adjusted_recs = []
        
        for rec in base_recs:
            appid = rec['appid']
            game_info = self.games_df[self.games_df['appid'] == appid]
            
            if game_info.empty:
                continue
            
            game = game_info.iloc[0]
            genres = str(game['genres'])
            primary_genre = genres.split(',')[0].strip()
            
            # Base score
            base_score = rec.get('hybrid_score', rec.get('quality_score', 0.5))
            
            # Initialize context score
            context_score = base_score * self.context_weights['content_score'] / 0.35
            
            # Device compatibility adjustment
            device_compatibility = 1.0
            compatibility_message = ""
            if compatibility_checker:
                is_compatible, comp_score, message = compatibility_checker.check_game_compatibility(
                    genres, rec['name']
                )
                device_compatibility = comp_score / 100.0
                context_score *= (0.5 + 0.5 * device_compatibility)  # Scale 0.5-1.0
                compatibility_message = message
            
            # Time of day adjustment
            time_bonus = 1.0
            for genre_key, bonus in time_genre_prefs.items():
                if genre_key.lower() in genres.lower():
                    time_bonus = max(time_bonus, bonus)
            context_score *= time_bonus * self.context_weights['time_of_day'] * 10
            
            # Location adjustment
            context_score = self.adjust_by_location(context_score, location, genres)
            
            # Weather adjustment
            context_score = self.adjust_by_weather(context_score, weather, genres)
            
            # Mood adjustment
            context_score = self.adjust_by_mood(context_score, mood, genres)
            
            # History bonus (games similar to selected games get bonus)
            if selected_games:
                if rec.get('recommendation_count', 1) > 1:
                    context_score *= (1 + 0.1 * rec['recommendation_count'])
            
            # Build recommendation dict
            adjusted_rec = {
                'rank': 0,
                'appid': appid,
                'name': rec['name'],
                'context_score': float(context_score),
                'base_score': float(base_score),
                'device_compatibility': float(device_compatibility * 100),
                'compatibility_message': compatibility_message,
                'price': rec['price'],
                'rating': rec.get('rating', game['rating_score']),
                'quality_score': rec.get('quality_score', game['quality_score']),
                'genres': genres,
                'header_image': rec.get('header_image', game.get('header_image', '')),
                'context_info': {
                    'time_of_day': time_context.value,
                    'location': location,
                    'weather': weather,
                    'mood': mood
                }
            }
            adjusted_recs.append(adjusted_rec)
        
        # Sort by context score
        adjusted_recs.sort(key=lambda x: x['context_score'], reverse=True)
        
        # Assign ranks
        for i, rec in enumerate(adjusted_recs[:n_recommendations], 1):
            rec['rank'] = i
        
        return adjusted_recs[:n_recommendations]
    
    def get_recommendation_explanation(self, rec: Dict) -> str:
        """
        Generate explanation for why a game was recommended.
        
        Args:
            rec: Recommendation dictionary with context info
        
        Returns:
            Human-readable explanation string
        """
        explanations = []
        context = rec.get('context_info', {})
        
        # Device compatibility
        compatibility = rec.get('device_compatibility', 100)
        if compatibility >= 80:
            explanations.append("[OK] Compatible with your device")
        elif compatibility >= 60:
            explanations.append("[!] May run on your device")
        else:
            explanations.append("[X] May not be compatible with your device")
        
        # Time-based
        time = context.get('time_of_day', '')
        if time == 'night':
            if 'Horror' in rec.get('genres', ''):
                explanations.append("Perfect for evening - horror experience!")
            else:
                explanations.append("Great for late night gaming")
        elif time == 'morning':
            explanations.append("Good to start your day")
        
        # Mood-based
        mood = context.get('mood', '')
        mood_messages = {
            'tired': "Light gameplay, no intense focus needed",
            'excited': "Action-packed for you!",
            'relaxed': "Relax and enjoy",
            'stressed': "Helps you de-stress"
        }
        if mood in mood_messages:
            explanations.append(mood_messages[mood])
        
        # Quality
        quality = rec.get('quality_score', 0)
        if quality >= 80:
            explanations.append(f"Highly rated ({quality:.0f}%)")
        
        return " | ".join(explanations) if explanations else "Recommended based on your preferences"
    
    def set_context_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom context weights.
        
        Args:
            weights: Dictionary of weight names to values
        """
        self.context_weights.update(weights)
        
        # Normalize
        total = sum(self.context_weights.values())
        if total > 0:
            self.context_weights = {k: v/total for k, v in self.context_weights.items()}
    
    def recommend_from_history(
        self,
        user_history,
        n_recommendations: int = 10,
        device_config: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate recommendations based on user's complete history.
        
        Args:
            user_history: UserHistory object
            n_recommendations: Number of recommendations
            device_config: Device configuration dict
        
        Returns:
            List of personalized recommendations
        """
        # Get user data
        selected_games = user_history.get_selected_appids()
        viewed_games = user_history.get_viewed_appids()
        favorite_genres = user_history.get_favorite_genres()
        context = user_history.get_context()
        
        # Use device config from history if not provided
        if device_config is None:
            device_config = user_history.get_device_config()
        
        # Combine selected and highly viewed games
        reference_games = selected_games if selected_games else viewed_games[:5]
        
        # Generate recommendations
        recommendations = self.context_aware_recommend(
            selected_games=reference_games,
            n_recommendations=n_recommendations,
            device_config=device_config,
            time_of_day=context.get('time_of_day'),
            mood='relaxed'
        )
        
        # Filter out already viewed games
        viewed_set = set(viewed_games)
        recommendations = [
            rec for rec in recommendations
            if rec['appid'] not in viewed_set
        ]
        
        # Boost games matching favorite genres
        for rec in recommendations:
            for fav_genre in favorite_genres:
                if fav_genre.lower() in rec['genres'].lower():
                    rec['context_score'] *= 1.1
        
        # Re-sort and re-rank
        recommendations.sort(key=lambda x: x['context_score'], reverse=True)
        for i, rec in enumerate(recommendations[:n_recommendations], 1):
            rec['rank'] = i
        
        return recommendations[:n_recommendations]


print("[OK] ContextAwareRecommender class ready!")
