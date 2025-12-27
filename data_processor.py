"""
Data Processor Module
=====================
Handles all data loading, cleaning, preprocessing, and feature engineering
for the Steam game recommendation system.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, List, Dict

warnings.filterwarnings('ignore')


class DataProcessor:
    """Handle all data loading, cleaning, and preprocessing tasks."""
    
    def __init__(self, data_path: str = "data/steam-store-games"):
        """
        Initialize DataProcessor.
        
        Args:
            data_path: Path to extracted Steam dataset directory
        """
        self.data_path = Path(data_path).expanduser()
        self.steam_df = None
        self.description_df = None
        self.media_df = None
        self.tags_df = None
        self.master_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load all Steam CSV files and merge them."""
        try:
            print("Loading data files...")
            
            # Load main steam data
            self.steam_df = pd.read_csv(self.data_path / "steam.csv", on_bad_lines='skip')
            print(f"  steam.csv: {len(self.steam_df)} records")
            
            # Load additional data
            self.description_df = pd.read_csv(
                self.data_path / "steam_description_data.csv", 
                on_bad_lines='skip'
            )
            print(f"  steam_description_data.csv: {len(self.description_df)} records")
            
            self.media_df = pd.read_csv(
                self.data_path / "steam_media_data.csv",
                on_bad_lines='skip'
            )
            print(f"  steam_media_data.csv: {len(self.media_df)} records")
            
            # Load tags data
            self.tags_df = pd.read_csv(
                self.data_path / "steamspy_tag_data.csv",
                on_bad_lines='skip'
            )
            print(f"  steamspy_tag_data.csv: {len(self.tags_df)} records")
            
            return self._merge_data()
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Dataset files not found at {self.data_path}: {e}")
    
    def _merge_data(self) -> pd.DataFrame:
        """Merge all data into a single DataFrame."""
        print("\nMerging data...")
        
        # Merge steam with description
        df = self.steam_df.merge(
            self.description_df,
            left_on='appid',
            right_on='steam_appid',
            how='left'
        )
        
        # Merge with media
        df = df.merge(
            self.media_df,
            left_on='appid',
            right_on='steam_appid',
            how='left'
        )
        
        # Merge with tags
        df = df.merge(
            self.tags_df,
            left_on='appid',
            right_on='appid',
            how='left'
        )
        
        self.master_df = df
        print(f"  Merged dataset: {len(df)} records, {len(df.columns)} features")
        return df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and prepare data."""
        df = self.master_df.copy()
        
        print("\nCleaning data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['appid'])
        print(f"  Removed {initial_count - len(df)} duplicates")
        
        # Remove rows with missing critical columns
        critical_cols = ['appid', 'name', 'genres', 'positive_ratings', 'negative_ratings']
        df = df.dropna(subset=critical_cols)
        print(f"  After removing missing critical values: {len(df)} records")
        
        # Handle missing values in other columns
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['average_playtime'] = pd.to_numeric(df['average_playtime'], errors='coerce').fillna(0)
        df['median_playtime'] = pd.to_numeric(df['median_playtime'], errors='coerce').fillna(0)
        
        # Fill text columns
        df['short_description'] = df['short_description'].fillna('')
        df['about_the_game'] = df['about_the_game'].fillna('')
        df['detailed_description'] = df['detailed_description'].fillna('')
        
        # Calculate ratings
        df['positive_ratings'] = pd.to_numeric(df['positive_ratings'], errors='coerce').fillna(0)
        df['negative_ratings'] = pd.to_numeric(df['negative_ratings'], errors='coerce').fillna(0)
        
        df['total_ratings'] = df['positive_ratings'] + df['negative_ratings']
        df['rating_score'] = np.where(
            df['total_ratings'] > 0,
            df['positive_ratings'] / df['total_ratings'] * 100,
            0
        )
        
        # Remove outliers in rating (games with extreme ratings)
        df = df[(df['total_ratings'] > 10) | (df['total_ratings'] == 0)]
        
        print(f"  After outlier removal: {len(df)} records")
        
        self.master_df = df
        return df
    
    def engineer_features(self) -> pd.DataFrame:
        """Create engineered features for recommendations."""
        df = self.master_df.copy()
        
        print("\nEngineering features...")
        
        # Popularity score (0-100)
        max_ratings = df['total_ratings'].max()
        df['popularity_score'] = (df['total_ratings'] / max_ratings * 100).fillna(0)
        
        # Quality score (combination of rating and popularity)
        df['quality_score'] = (df['rating_score'] * 0.6 + df['popularity_score'] * 0.4)
        
        # Game age (days since release, or 0 if missing)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['game_age_days'] = (pd.Timestamp.now() - df['release_date']).dt.days.fillna(0)
        
        # Extract primary genre (first genre listed)
        df['primary_genre'] = df['genres'].str.split(',').str[0].str.strip()
        
        # Price category
        df['price_category'] = pd.cut(
            df['price'],
            bins=[-1, 0, 5, 15, 30, 100000],
            labels=['Free', 'Budget', 'Standard', 'Premium', 'Expensive']
        )
        
        print("  Features engineered: popularity_score, quality_score, game_age_days, primary_genre, price_category")
        
        self.master_df = df
        return df
    
    def get_filtered_dataset(self, min_ratings: int = 10) -> pd.DataFrame:
        """Get dataset filtered for quality games"""
        df = self.master_df.copy()
        
        # Filter games with minimum ratings for better recommendations
        df_filtered = df[df['total_ratings'] >= min_ratings].copy()
        
        print(f"\n  Filtered dataset: {len(df_filtered)} games (min {min_ratings} ratings)")
        
        return df_filtered
    
    def get_feature_matrix_for_content(self) -> Tuple[np.ndarray, List[int], List[str]]:
        """Create feature vectors for content-based filtering"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        df = self.master_df.copy()
        
        # Combine text features
        df['combined_features'] = (
            df['genres'].fillna('') + ' ' +
            df['categories'].fillna('') +
            ' ' +
            df['short_description'].fillna('').str[:200]
        )
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        feature_matrix = tfidf.fit_transform(df['combined_features']).toarray()
        
        return feature_matrix, df['appid'].tolist(), df['name'].tolist()
    
    def get_rating_matrix(self) -> Tuple[pd.DataFrame, Dict]:
        """Create user-item rating matrix for collaborative filtering"""
        # For Steam data, we'll create a synthetic user-item matrix based on:
        # - Player count (estimate number of users who played)
        # - Rating score (implicit feedback)
        
        df = self.master_df.copy()
        
        # Extract owners (approximate user count)
        df['owners'] = pd.to_numeric(
            df['owners'].str.replace(r'[^\d\-]', '', regex=True),
            errors='coerce'
        ).fillna(0)
        
        # Create a sparse user-item matrix
        # For each game, approximate user ratings based on positive/total ratio
        rating_matrix = pd.DataFrame({
            'appid': df['appid'],
            'name': df['name'],
            'estimated_users': np.log1p(df['owners']) / np.log1p(df['owners'].max()) * 1000,
            'rating_score': df['rating_score'],
            'quality_score': df['quality_score']
        })
        
        metadata = {
            'total_games': len(df),
            'avg_rating': df['rating_score'].mean(),
            'avg_playtime': df['average_playtime'].mean()
        }
        
        return rating_matrix, metadata
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the dataset"""
        df = self.master_df.copy()
        
        return {
            'total_games': len(df),
            'avg_rating': df['rating_score'].mean(),
            'avg_price': df['price'].mean(),
            'avg_playtime': df['average_playtime'].mean(),
            'total_genres': df['primary_genre'].nunique(),
            'date_range': f"{df['release_date'].min().date()} to {df['release_date'].max().date()}",
            'free_games': len(df[df['price'] == 0]),
            'avg_age_days': df['game_age_days'].mean()
        }

# Test import
print("DataProcessor class ready!")
