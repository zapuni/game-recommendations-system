"""
Steam Game Recommender System - Main Application
================================================
A comprehensive game recommendation system with:
- User authentication (MongoDB/JSON)
- Content-based filtering with advanced embeddings
- Hybrid recommendations
- Context-aware personalization with device config
- User history tracking
- Model evaluation (RMSE, MAE, Precision@K, Recall@K)

Tech Stack:
- Streamlit (Web UI)
- Pandas/NumPy (Data processing)
- Scikit-learn (ML/Vectorization)
- Plotly (Visualization)
- MongoDB (optional - user data)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from recommender import RecommenderSystem
from evaluator import ModelEvaluator, EvaluationMetrics
from user_history import UserHistory, SessionManager
from auth import AuthManager, SessionAuthManager
from context_aware import ContextAwareRecommender
from device_config import (
    DeviceConfig, DeviceCompatibilityChecker,
    get_device_types, get_gpu_tiers, get_cpu_tiers,
    get_storage_types, get_resolutions
)

from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM STYLES

def load_custom_css():
    """Load custom CSS styles for the application."""
    st.markdown("""
    <style>
        [data-testid="stMetricLabel"] {
            font-size: 14px;
            font-weight: 600;
        }
        .metric-card {
            background: #667eea;
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        .game-card {
            background: var(--background-color, rgba(100, 100, 100, 0.1));
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            color: inherit;
        }
        .game-card-with-image {
            background: var(--background-color, rgba(100, 100, 100, 0.1));
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
            border: 1px solid rgba(100, 100, 100, 0.2);
        }
        .game-card-image {
            width: 100%;
            height: 120px;
            object-fit: cover;
        }
        .game-card-content {
            padding: 10px;
        }
        .game-card-title {
            font-weight: 700;
            font-size: 0.95em;
            margin-bottom: 5px;
            color: inherit;
        }
        .game-card-info {
            font-size: 0.8em;
            opacity: 0.8;
            margin-bottom: 2px;
        }
        .game-card b {
            color: inherit;
        }
        .game-card small {
            color: inherit;
            opacity: 0.8;
        }
        .header-title {
            font-size: 2.5em;
            font-weight: 700;
            background: #667eea;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .auth-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
        }
        .selected-game {
            background: #11998e;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            margin: 5px;
            display: inline-block;
        }
        .compatibility-good {
            color: #38ef7d;
            font-weight: bold;
        }
        .compatibility-warning {
            color: #f39c12;
            font-weight: bold;
        }
        .compatibility-bad {
            color: #e74c3c;
            font-weight: bold;
        }
        .detail-header-image {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        .screenshot-img {
            width: 100%;
            border-radius: 6px;
            cursor: pointer;
            margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)



# DATA LOADING & INITIALIZATION


@st.cache_resource
def load_data():
    """Load and process game data."""
    processor = DataProcessor(data_path="data/steam-store-games")
    
    df = processor.load_data()
    df = processor.clean_data()
    df = processor.engineer_features()
    df_filtered = processor.get_filtered_dataset(min_ratings=10)
    
    return processor, df, df_filtered


@st.cache_resource
def initialize_recommender(_df_filtered):
    """Initialize the recommender system."""
    recommender = RecommenderSystem(_df_filtered)
    feature_matrix, appids, names = get_content_features(_df_filtered)
    recommender.set_content_features(feature_matrix)
    return recommender


def get_content_features(df):
    """Extract content features using TF-IDF vectorization."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
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


def get_auth_manager():
    """Get or create auth manager."""
    if 'auth_manager' not in st.session_state:
        st.session_state['auth_manager'] = AuthManager(mongodb_uri=MONGODB_URI)
    return st.session_state['auth_manager']


def get_user_history(username=None):
    """Get or create user history from session state."""
    return SessionManager.get_or_create_history(
        st.session_state,
        username=username,
        mongodb_uri=MONGODB_URI
    )



# AUTHENTICATION UI


def render_auth_page():
    """Render login/register page."""
    st.markdown('<div class="header-title">Steam Game Recommender</div>', unsafe_allow_html=True)
    st.markdown("*Login to get personalized game recommendations*")
    st.divider()
    
    auth_manager = get_auth_manager()
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                success, message, user_data = auth_manager.login(username, password)
                if success:
                    SessionAuthManager.login_user(st.session_state, username, user_data)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        st.divider()
        if st.button("Continue without login", use_container_width=True):
            st.session_state['guest_mode'] = True
            st.rerun()
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Register Account")
            new_username = st.text_input("Username", key="reg_username")
            new_email = st.text_input("Email (optional)", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            
            submit = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                else:
                    success, message = auth_manager.register(
                        new_username, new_password, new_email
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)



# UI COMPONENTS


def render_header():
    """Render the application header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(
            '<div class="header-title">Steam Game Recommender</div>',
            unsafe_allow_html=True
        )
        if SessionAuthManager.is_logged_in(st.session_state):
            username = SessionAuthManager.get_username(st.session_state)
            st.markdown(f"*Hello, **{username}**! Discover games tailored for you.*")
        else:
            st.markdown("*Discover your next favorite game with AI-powered recommendations*")
    with col3:
        if SessionAuthManager.is_logged_in(st.session_state):
            if st.button("Logout"):
                SessionAuthManager.logout_user(st.session_state)
                st.rerun()
    st.divider()


def render_game_card(game: dict, show_score: bool = True, show_compatibility: bool = False):
    """Render a simple game card with details (no image)."""
    # Build optional parts
    optional_parts = []
    if show_score:
        if 'context_score' in game:
            optional_parts.append(f"<br>Score: {game['context_score']:.2f}")
        elif 'hybrid_score' in game:
            optional_parts.append(f"<br>Score: {game['hybrid_score']:.2f}")
    
    if show_compatibility and 'device_compatibility' in game:
        comp = game['device_compatibility']
        if comp >= 80:
            cls = "compatibility-good"
            icon = "[OK]"
        elif comp >= 60:
            cls = "compatibility-warning"
            icon = "[!]"
        else:
            cls = "compatibility-bad"
            icon = "[X]"
        optional_parts.append(f'<br><span class="{cls}">{icon} Compatibility: {comp:.0f}%</span>')
    
    optional_html = ''.join(optional_parts)
    
    # Escape text to prevent HTML issues
    game_name = str(game.get('name', ''))[:40].replace('<', '&lt;').replace('>', '&gt;')
    game_genres = str(game.get('genres', ''))[:50].replace('<', '&lt;').replace('>', '&gt;')
    
    # Build HTML without extra whitespace
    card_html = (
        '<div class="game-card">'
        f'<b>{game_name}</b>'
        f'<br>Rating: {game.get("quality_score", game.get("rating", 0)):.1f}%'
        f'<br>Price: ${game["price"]:.2f}'
        f'<br><small>{game_genres}</small>'
        f'{optional_html}'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_game_card_with_image(
    game: dict, 
    card_key: str,
    user_history: UserHistory = None,
    show_score: bool = True, 
    show_compatibility: bool = False,
    show_buttons: bool = True
):
    """
    Render a game card with header image and action buttons.
    
    Args:
        game: Game data dictionary
        card_key: Unique key for this card (for buttons)
        user_history: UserHistory object for adding to selected games
        show_score: Show recommendation score
        show_compatibility: Show device compatibility
        show_buttons: Show action buttons (+, Select, View Details)
    """
    # Get header image
    header_image = game.get('header_image', '')
    if not header_image or pd.isna(header_image):
        header_image = "https://placeholder.pics/svg/460x215/DEDEDE/555555/No%20image%20available"
    
    # Score text
    score_text = ""
    if show_score:
        if 'context_score' in game:
            score_text = f"Score: {game['context_score']:.2f}"
        elif 'hybrid_score' in game:
            score_text = f"Score: {game['hybrid_score']:.2f}"
    
    # Compatibility text
    compatibility_info = ""
    if show_compatibility and 'device_compatibility' in game:
        comp = game['device_compatibility']
        if comp >= 80:
            compatibility_info = f"[OK] {comp:.0f}%"
        elif comp >= 60:
            compatibility_info = f"[!] {comp:.0f}%"
        else:
            compatibility_info = f"[X] {comp:.0f}%"
    
    # Build optional HTML parts as list
    optional_parts = []
    if score_text:
        optional_parts.append(f'<div class="game-card-info">{score_text}</div>')
    if compatibility_info:
        optional_parts.append(f'<div class="game-card-info">Compatibility: {compatibility_info}</div>')
    optional_html = ''.join(optional_parts)
    
    # Escape game name and genres to prevent HTML injection
    game_name = str(game.get('name', ''))[:35].replace('<', '&lt;').replace('>', '&gt;')
    game_genres = str(game.get('genres', ''))[:40].replace('<', '&lt;').replace('>', '&gt;')
    
    # Render card with image - no extra whitespace
    card_html = (
        '<div class="game-card-with-image">'
        f'<img src="{header_image}" class="game-card-image" onerror="this.src=\'https://placeholder.pics/svg/460x215/DEDEDE/555555/No%20image%20available\'"/>'
        '<div class="game-card-content">'
        f'<div class="game-card-title">{game_name}</div>'
        f'<div class="game-card-info">Rating: {game.get("quality_score", game.get("rating", 0)):.1f}% | ${game["price"]:.2f}</div>'
        f'<div class="game-card-info">{game_genres}</div>'
        f'{optional_html}'
        '</div>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Action buttons
    if show_buttons:
        btn_cols = st.columns(3)
        
        with btn_cols[0]:
            if st.button("+Add", key=f"add_{card_key}", help="Add to selected games"):
                if user_history:
                    user_history.add_selected_game(game)
                    st.success(f"Added!")
                    st.rerun()
        
        with btn_cols[1]:
            if st.button("Select", key=f"sel_{card_key}", help="Select for recommendations"):
                st.session_state['selected_game'] = game
                if user_history:
                    user_history.add_viewed_game(game)
                st.rerun()
        
        with btn_cols[2]:
            if st.button("Details", key=f"det_{card_key}", help="View game details"):
                st.session_state['view_game_details'] = game
                st.rerun()


def show_game_details_dialog(game: dict, df_filtered: pd.DataFrame):
    """Show game details in a dialog/modal with media."""
    import ast
    
    # Get full game data from dataframe
    game_data = df_filtered[df_filtered['appid'] == game['appid']]
    if game_data.empty:
        st.error("Game not found")
        return
    
    game_row = game_data.iloc[0]
    
    # Create a container for the details view
    with st.container():
        # Close button at top
        col_title, col_close = st.columns([4, 1])
        with col_title:
            st.markdown(f"## {game_row['name']}")
        with col_close:
            if st.button("X Close", key="close_details_top"):
                if 'view_game_details' in st.session_state:
                    del st.session_state['view_game_details']
                st.rerun()
        
        st.divider()
        
        # Header image
        header_image = game_row.get('header_image', '')
        if header_image and not pd.isna(header_image):
            st.image(header_image, use_column_width=True)
        
        # Game info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rating", f"{game_row.get('rating_score', 0):.1f}%")
        with col2:
            st.metric("Price", f"${game_row.get('price', 0):.2f}")
        with col3:
            st.metric("Quality Score", f"{game_row.get('quality_score', 0):.1f}")
        
        st.markdown(f"**Genres:** {game_row.get('genres', 'N/A')}")
        st.markdown(f"**Categories:** {game_row.get('categories', 'N/A')}")
        
        # Description
        description = game_row.get('short_description', '')
        if description and not pd.isna(description):
            st.markdown("### Description")
            st.write(description)
        
        # Screenshots
        screenshots_str = game_row.get('screenshots', '')
        if screenshots_str and not pd.isna(screenshots_str):
            st.markdown("### Screenshots")
            try:
                # Parse screenshots (stored as string representation of list)
                if isinstance(screenshots_str, str) and screenshots_str.startswith('['):
                    screenshots = ast.literal_eval(screenshots_str)
                    
                    # Display screenshots in grid
                    if screenshots:
                        num_cols = 3
                        cols = st.columns(num_cols)
                        for i, screenshot in enumerate(screenshots[:9]):  # Limit to 9 screenshots
                            with cols[i % num_cols]:
                                img_url = screenshot.get('path_thumbnail', screenshot.get('path_full', ''))
                                if img_url:
                                    st.image(img_url, use_column_width=True)
            except Exception:
                st.caption("Could not load screenshots")
        
        # Background image
        background = game_row.get('background', '')
        if background and not pd.isna(background):
            with st.expander("Background Image"):
                st.image(background, use_column_width=True)
        
        # Movies/Videos
        movies_str = game_row.get('movies', '')
        if movies_str and not pd.isna(movies_str) and str(movies_str).strip():
            st.markdown("### Videos")
            try:
                if isinstance(movies_str, str) and movies_str.startswith('['):
                    movies = ast.literal_eval(movies_str)
                    for movie in movies[:2]:  # Limit to 2 videos
                        movie_url = movie.get('webm', {}).get('480', '') or movie.get('mp4', {}).get('480', '')
                        if movie_url:
                            st.video(movie_url)
            except Exception:
                st.caption("Could not load videos")
        
        st.divider()
        
        # Close button at bottom
        if st.button("Close Details", use_container_width=True, key="close_details_bottom"):
            if 'view_game_details' in st.session_state:
                del st.session_state['view_game_details']
            st.rerun()


def render_selected_games(user_history: UserHistory):
    """Render selected games for recommendations."""
    selected = user_history.get_selected_games()
    
    if selected:
        st.markdown("**Selected games for recommendations:**")
        cols = st.columns(min(5, len(selected)))
        for i, game in enumerate(selected):
            with cols[i % 5]:
                # Escape game name
                game_name = str(game.get('name', ''))[:20].replace('<', '&lt;').replace('>', '&gt;')
                card_html = (
                    '<div class="selected-game">'
                    f'{game_name}'
                    f'<br><small>${game.get("price", 0):.2f}</small>'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
        
        if st.button("Clear all selected games"):
            user_history.clear_selected_games()
            st.rerun()


def render_history_sidebar(user_history: UserHistory):
    """Render user history in sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown("### Your History")
    
    # Selected games count
    selected = user_history.get_selected_games()
    st.sidebar.markdown(f"**Games selected:** {len(selected)}")
    
    # Recent views
    recent_games = user_history.get_viewed_games(limit=5)
    if recent_games:
        st.sidebar.markdown("**Recently viewed:**")
        for game in recent_games[:5]:
            st.sidebar.markdown(f"- {game['name'][:25]}")
    
    # Analytics summary
    analytics = user_history.get_analytics()
    if analytics['total_games_viewed'] > 0:
        st.sidebar.markdown("**Statistics:**")
        st.sidebar.markdown(f"- Games viewed: {analytics['total_games_viewed']}")
        st.sidebar.markdown(f"- Searches: {analytics['total_searches']}")
    
    # Clear history button
    if st.sidebar.button("Clear History"):
        user_history.clear_history()
        st.rerun()



# DEVICE CONFIGURATION FORM


def render_device_config_form(user_history: UserHistory):
    """Render device configuration form for context-aware recommendations."""
    st.subheader("Your Device Configuration")
    st.markdown("*Enter your device specs to get compatible game recommendations*")
    
    # Load existing config
    existing_config = user_history.get_device_config()
    
    # Initialize session state for has_dedicated_gpu (outside form for dynamic updates)
    if 'has_dedicated_gpu' not in st.session_state:
        st.session_state['has_dedicated_gpu'] = existing_config.get('has_dedicated_gpu', True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Device Info**")
        
        device_type = st.selectbox(
            "Device Type",
            get_device_types(),
            index=get_device_types().index(existing_config.get('device_type', 'PC Desktop'))
            if existing_config.get('device_type', 'PC Desktop') in get_device_types() else 0,
            key="device_type_select"
        )
        
        cpu = st.text_input(
            "CPU (e.g., Intel Core i5-12400, AMD Ryzen 5 5600)",
            value=existing_config.get('cpu', ''),
            key="cpu_input"
        )
        
        cpu_tier = st.selectbox(
            "CPU Tier",
            get_cpu_tiers(),
            index=get_cpu_tiers().index(existing_config.get('cpu_tier', 'Mid Range (i5/Ryzen 5)'))
            if existing_config.get('cpu_tier', 'Mid Range (i5/Ryzen 5)') in get_cpu_tiers() else 1,
            key="cpu_tier_select"
        )
        
        ram_gb = st.slider(
            "RAM (GB)",
            min_value=2,
            max_value=128,
            value=existing_config.get('ram_gb', 8),
            step=2,
            key="ram_slider"
        )
    
    with col2:
        st.markdown("**GPU & Storage**")
        
        # Checkbox outside form for dynamic enable/disable
        has_dedicated_gpu = st.checkbox(
            "Has Dedicated GPU",
            value=st.session_state['has_dedicated_gpu'],
            key="has_dedicated_gpu_checkbox",
            on_change=lambda: st.session_state.update({'has_dedicated_gpu': not st.session_state['has_dedicated_gpu']})
        )
        
        # Use session state value to control disabled state
        gpu_disabled = not st.session_state['has_dedicated_gpu']
        
        gpu = st.text_input(
            "GPU (e.g., NVIDIA RTX 3060, AMD RX 6700 XT)",
            value=existing_config.get('gpu', '') if not gpu_disabled else '',
            disabled=gpu_disabled,
            key="gpu_input"
        )
        
        gpu_tier = st.selectbox(
            "GPU Tier",
            get_gpu_tiers(),
            index=get_gpu_tiers().index(existing_config.get('gpu_tier', 'Mid Range (GTX 1050-1660, RTX 2060)'))
            if existing_config.get('gpu_tier', 'Mid Range (GTX 1050-1660, RTX 2060)') in get_gpu_tiers() else 2,
            disabled=gpu_disabled,
            key="gpu_tier_select"
        )
        
        storage_col1, storage_col2 = st.columns(2)
        with storage_col1:
            storage_gb = st.number_input(
                "Storage (GB)",
                min_value=64,
                max_value=8000,
                value=existing_config.get('storage_gb', 512),
                step=64,
                key="storage_gb_input"
            )
        with storage_col2:
            storage_type = st.selectbox(
                "Storage Type",
                get_storage_types(),
                index=get_storage_types().index(existing_config.get('storage_type', 'SSD'))
                if existing_config.get('storage_type', 'SSD') in get_storage_types() else 0,
                key="storage_type_select"
            )
    
    st.markdown("**Display & Other**")
    col3, col4 = st.columns(2)
    
    with col3:
        screen_resolution = st.selectbox(
            "Screen Resolution",
            get_resolutions(),
            index=0,
            key="screen_resolution_select"
        )
    
    with col4:
        vr_capable = st.checkbox(
            "VR Capable",
            value=existing_config.get('vr_capable', False),
            key="vr_capable_checkbox"
        )
    
    # Save button
    if st.button("Save Configuration", use_container_width=True, key="save_config_btn"):
        config = {
            'device_type': device_type,
            'cpu': cpu,
            'cpu_tier': cpu_tier,
            'ram_gb': ram_gb,
            'storage_gb': storage_gb,
            'storage_type': storage_type,
            'gpu': gpu if has_dedicated_gpu else '',
            'gpu_tier': gpu_tier if has_dedicated_gpu else 'Integrated (Intel HD/UHD)',
            'has_dedicated_gpu': has_dedicated_gpu,
            'screen_resolution': screen_resolution,
            'vr_capable': vr_capable
        }
        user_history.set_device_config(config)
        st.success("Device configuration saved!")
        st.rerun()
    
    # Show current performance score
    if existing_config:
        config_obj = DeviceConfig.from_dict(existing_config)
        score = config_obj.get_performance_score()
        
        st.divider()
        st.markdown("**Device Performance Score:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Score", f"{score:.0f}/100")
        with col2:
            checker = DeviceCompatibilityChecker(config_obj)
            settings = checker.get_recommended_settings()
            st.metric("Recommended Settings", settings['quality'])
        with col3:
            st.metric("Target FPS", settings['fps_target'])
        
        st.info(settings['description'])



# PAGE: RECOMMENDER


def page_recommender(recommender, df_filtered, user_history):
    """Render the main recommender page."""
    st.header("Game Recommendations")
    
    # Check if we need to show game details view
    if 'view_game_details' in st.session_state and st.session_state['view_game_details']:
        show_game_details_dialog(st.session_state['view_game_details'], df_filtered)
        return  # Don't show the rest of the page when viewing details
    
    # Show selected games
    render_selected_games(user_history)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Find Similar",
        "Device-Based",
        "Browse by Genre",
        "Top Rated",
        "Personalized"
    ])
    
    # --- Tab 1: Similar Games ---
    with tab1:
        render_similar_games_tab(recommender, df_filtered, user_history)
    
    # --- Tab 2: Context-Aware with Device Config ---
    with tab2:
        render_context_aware_tab(recommender, df_filtered, user_history)
    
    # --- Tab 3: Browse by Genre ---
    with tab3:
        render_genre_tab(recommender, df_filtered, user_history)
    
    # --- Tab 4: Top Rated ---
    with tab4:
        render_top_rated_tab(recommender, df_filtered, user_history)
    
    # --- Tab 5: Personalized ---
    with tab5:
        render_personalized_tab(recommender, df_filtered, user_history)


def render_similar_games_tab(recommender, df_filtered, user_history):
    """Render the similar games search tab."""
    st.subheader("Find Similar Games")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input(
            "Search for a game:",
            placeholder="e.g., The Witcher, Elden Ring, Cyberpunk..."
        )
    with col2:
        n_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    if search_query:
        user_history.add_search(search_query)
        search_results = recommender.search_games(search_query, n_results=10)
        
        if search_results:
            st.write(f"Found {len(search_results)} games:")
            
            # Display search results as cards with images
            cols = st.columns(3)
            for idx, game in enumerate(search_results):
                with cols[idx % 3]:
                    # Get header_image from df_filtered
                    game_data = df_filtered[df_filtered['appid'] == game['appid']]
                    if not game_data.empty:
                        game['header_image'] = game_data.iloc[0].get('header_image', '')
                    
                    render_game_card_with_image(
                        game,
                        card_key=f"search_{game['appid']}",
                        user_history=user_history,
                        show_score=False
                    )
            
            # Show recommendations for selected game
            if 'selected_game' in st.session_state:
                st.divider()
                selected = st.session_state['selected_game']
                st.success(f"**Selected:** {selected['name']}")
                
                with st.spinner(f"Finding games similar to {selected['name']}..."):
                    hybrid_recs = recommender.hybrid_recommend(
                        selected['appid'],
                        n_recommendations=n_recs,
                        content_weight=0.6,
                        quality_weight=0.4
                    )
                
                st.subheader("Similar Games")
                rec_cols = st.columns(3)
                for idx, rec in enumerate(hybrid_recs[:9]):
                    with rec_cols[idx % 3]:
                        # Get header_image from df_filtered
                        rec_data = df_filtered[df_filtered['appid'] == rec['appid']]
                        if not rec_data.empty:
                            rec['header_image'] = rec_data.iloc[0].get('header_image', '')
                        
                        render_game_card_with_image(
                            rec,
                            card_key=f"rec_{rec['appid']}",
                            user_history=user_history,
                            show_score=True
                        )
                
                if hybrid_recs:
                    render_recommendation_chart(hybrid_recs)
        else:
            st.warning(f"No games found for '{search_query}'. Try different keywords!")


def render_context_aware_tab(recommender, df_filtered, user_history):
    """Render context-aware recommendations with device config."""
    st.subheader("Device-Based Recommendations")
    
    # Device config form
    with st.expander("Device Configuration", expanded=True):
        render_device_config_form(user_history)
    
    st.divider()
    
    # Context options
    st.markdown("**Current Context:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_of_day = st.selectbox(
            "Time of Day",
            ['morning', 'afternoon', 'evening', 'night'],
            index=2
        )
    with col2:
        location = st.selectbox(
            "Location",
            ['home', 'work', 'public', 'travel'],
            index=0
        )
    with col3:
        weather = st.selectbox(
            "Weather",
            ['sunny', 'rainy', 'snowy', 'cloudy', 'stormy'],
            index=0
        )
    with col4:
        mood = st.selectbox(
            "Mood",
            ['happy', 'relaxed', 'excited', 'tired', 'stressed', 'sad', 'angry'],
            index=1
        )
    
    n_recs = st.slider("Number of recommendations:", 5, 20, 10, key="context_n_recs")
    
    if st.button("Get Context-Aware Recommendations", use_container_width=True):
        device_config = user_history.get_device_config()
        selected_games = user_history.get_selected_appids()
        
        context_recommender = ContextAwareRecommender(df_filtered, recommender)
        
        with st.spinner("Analyzing and generating recommendations..."):
            recommendations = context_recommender.context_aware_recommend(
                selected_games=selected_games if selected_games else None,
                n_recommendations=n_recs,
                device_config=device_config,
                time_of_day=time_of_day,
                location=location,
                weather=weather,
                mood=mood
            )
        
        if recommendations:
            st.success(f"Found {len(recommendations)} compatible games!")
            
            # Display recommendations with compatibility
            cols = st.columns(3)
            for idx, rec in enumerate(recommendations):
                with cols[idx % 3]:
                    render_game_card_with_image(
                        rec,
                        card_key=f"ctx_{rec['appid']}",
                        user_history=user_history,
                        show_score=True,
                        show_compatibility=True
                    )
                    # Show explanation below card
                    explanation = context_recommender.get_recommendation_explanation(rec)
                    st.caption(explanation)
        else:
            st.warning("No recommendations found. Try selecting some games first!")


def render_genre_tab(recommender, df_filtered, user_history):
    """Render the genre browsing tab."""
    st.subheader("Top Games by Genre")
    
    available_genres = sorted(df_filtered['primary_genre'].dropna().unique())
    selected_genre = st.selectbox("Select genre:", available_genres)
    
    if selected_genre:
        genre_recs = recommender.genre_based_recommend(selected_genre, n_recommendations=12)
        
        if genre_recs:
            # Stats column
            genre_df = df_filtered[
                df_filtered['genres'].str.contains(selected_genre, case=False, na=False)
            ]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Games in Genre", len(genre_df))
            with col2:
                st.metric("Average Rating", f"{genre_df['rating_score'].mean():.1f}%")
            with col3:
                st.metric("Average Price", f"${genre_df['price'].mean():.2f}")
            
            st.divider()
            st.subheader(f"Top Games - {selected_genre}")
            
            # Display games as cards with images
            cols = st.columns(3)
            for idx, rec in enumerate(genre_recs[:9]):
                with cols[idx % 3]:
                    # Get header_image from df_filtered
                    rec_data = df_filtered[df_filtered['appid'] == rec['appid']]
                    if not rec_data.empty:
                        rec['header_image'] = rec_data.iloc[0].get('header_image', '')
                    
                    render_game_card_with_image(
                        rec,
                        card_key=f"genre_{rec['appid']}",
                        user_history=user_history,
                        show_score=False
                    )
            
            # Show chart
            st.divider()
            fig = px.histogram(
                genre_df,
                x='rating_score',
                nbins=20,
                title=f"Rating Distribution - {selected_genre}",
                labels={'rating_score': 'Rating %'},
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)


def render_top_rated_tab(recommender, df_filtered, user_history):
    """Render the top rated games tab."""
    st.subheader("Top Rated Games")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write("Discover the most popular games")
    with col2:
        show_free = st.checkbox("Show free games on top", value=False)
    with col3:
        view_mode = st.selectbox("View", ["Cards", "Table"], key="top_view_mode")
    
    top_games = recommender.popularity_based_recommend(
        n_recommendations=12,
        exclude_free=not show_free
    )
    
    if view_mode == "Table":
        top_df = pd.DataFrame({
            'Rank': [g['rank'] for g in top_games],
            'Game': [g['name'] for g in top_games],
            'Rating': [f"{g['rating']:.1f}%" for g in top_games],
            'Quality': [f"{g['quality_score']:.1f}" for g in top_games],
            'Price': [f"${g['price']:.2f}" for g in top_games],
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True)
    else:
        # Display as cards with images
        cols = st.columns(3)
        for idx, game in enumerate(top_games):
            with cols[idx % 3]:
                # Get header_image from df_filtered
                game_data = df_filtered[df_filtered['appid'] == game['appid']]
                if not game_data.empty:
                    game['header_image'] = game_data.iloc[0].get('header_image', '')
                
                render_game_card_with_image(
                    game,
                    card_key=f"top_{game['appid']}",
                    user_history=user_history,
                    show_score=False
                )


def render_personalized_tab(recommender, df_filtered, user_history):
    """Render personalized recommendations based on user history."""
    st.subheader("Personalized Recommendations")
    
    analytics = user_history.get_analytics()
    selected_games = user_history.get_selected_games()
    
    if len(selected_games) < 1 and analytics['total_games_viewed'] < 3:
        st.info(
            "Please select some games or browse more to get personalized recommendations! "
            "We need to learn your preferences."
        )
        return
    
    # Get recommendations
    context_recommender = ContextAwareRecommender(df_filtered, recommender)
    
    with st.spinner("Generating recommendations just for you..."):
        personalized_recs = context_recommender.recommend_from_history(
            user_history,
            n_recommendations=12
        )
    
    if personalized_recs:
        # Context info
        context = user_history.get_context()
        st.caption(
            f"Based on {len(selected_games)} selected games and browsing history "
            f"({context['time_of_day']}, {'Weekend' if context['is_weekend'] else 'Weekday'})"
        )
        
        # Display in grid with images
        cols = st.columns(3)
        for idx, rec in enumerate(personalized_recs):
            with cols[idx % 3]:
                render_game_card_with_image(
                    rec,
                    card_key=f"pers_{rec['appid']}",
                    user_history=user_history,
                    show_score=True,
                    show_compatibility=True
                )
    else:
        st.warning("Cannot generate recommendations yet. Please select more games!")


def render_recommendation_chart(recommendations):
    """Render a bar chart comparing recommendation scores."""
    st.subheader("Recommendation Score Comparison")
    
    rec_df = pd.DataFrame({
        'Game': [r['name'][:20] for r in recommendations[:10]],
        'Hybrid Score': [r['hybrid_score'] for r in recommendations[:10]],
        'Quality Score': [r['quality_score'] for r in recommendations[:10]],
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rec_df['Game'],
        y=rec_df['Hybrid Score'],
        name='Hybrid Score',
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        x=rec_df['Game'],
        y=rec_df['Quality Score'],
        name='Quality Score',
        marker_color='#764ba2'
    ))
    
    fig.update_layout(
        barmode='group',
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)



# PAGE: DATA ANALYTICS


def page_data_analytics(processor, df_filtered):
    """Render the data analytics page."""
    st.header("Dataset Analytics")
    
    stats = processor.get_summary_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Games", f"{stats['total_games']:,}")
    col2.metric("Avg Rating", f"{stats['avg_rating']:.1f}%")
    col3.metric("Avg Price", f"${stats['avg_price']:.2f}")
    col4.metric("Genres", stats['total_genres'])
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs([
        "Rating Distribution",
        "Price Analysis",
        "Genre Popularity"
    ])
    
    with tab1:
        fig = px.histogram(
            df_filtered,
            x='rating_score',
            nbins=30,
            title="Game Rating Distribution",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        price_df = df_filtered[df_filtered['price'] <= df_filtered['price'].quantile(0.95)]
        fig = px.histogram(
            price_df,
            x='price',
            nbins=50,
            title="Game Price Distribution",
            color_discrete_sequence=['#764ba2']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        genre_counts = {}
        for genres_str in df_filtered['genres'].dropna():
            for genre in str(genres_str).split(','):
                genre = genre.strip()
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        genre_names, genre_counts_vals = zip(*top_genres)
        
        fig = px.bar(
            x=list(genre_counts_vals),
            y=list(genre_names),
            orientation='h',
            title="Top 15 Popular Genres",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)



# PAGE: MODEL EVALUATION

EVALUATION_RESULTS_PATH = "results/evaluation_results.json"


def load_evaluation_results():
    """Load saved evaluation results from JSON file."""
    import json
    
    if not os.path.exists(EVALUATION_RESULTS_PATH):
        return None
    
    try:
        with open(EVALUATION_RESULTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def page_model_evaluation(df_filtered, recommender):
    """Render the model evaluation page with RMSE, MAE, Precision@K, Recall@K."""
    st.header("Model Evaluation")
    
    # Load saved results
    saved_results = load_evaluation_results()
    
    if saved_results is None:
        st.warning("No evaluation results found!")
        st.info(
            "Please run the evaluation script first:\n\n"
            "```bash\n"
            "python run_evaluation.py --samples 50 --k 10\n"
            "```\n\n"
            "This will evaluate all models and save results to `results/evaluation_results.json`"
        )
        
        st.markdown("""
        **Available options:**
        - `--samples` or `-s`: Number of test samples (default: 50)
        - `--k` or `-k`: Top-K value for ranking metrics (default: 10)
        - `--output` or `-o`: Output file path
        
        **Examples:**
        ```bash
        python run_evaluation.py
        python run_evaluation.py --samples 100 --k 10
        python run_evaluation.py -s 200 -k 5
        ```
        """)
        return
    
    # Display metadata
    metadata = saved_results.get('metadata', {})
    k_value = metadata.get('k_value', 10)
    
    st.success(f"Loaded evaluation results from: `{EVALUATION_RESULTS_PATH}`")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Samples", metadata.get('n_samples', 'N/A'))
    with col2:
        st.metric("K Value", k_value)
    with col3:
        st.metric("Total Games", metadata.get('total_games', 'N/A'))
    with col4:
        timestamp = metadata.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            # Format timestamp
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        st.metric("Evaluated At", timestamp)
    
    st.divider()
    
    # Main metrics table
    st.subheader("Evaluation Results Table")
    
    ranking_results = saved_results.get('ranking_metrics', [])
    
    if not ranking_results:
        st.error("No ranking metrics found in results file.")
        return
    
    # Create dataframe with required metrics: RMSE, MAE, Precision@K, Recall@K
    eval_data = []
    for r in ranking_results:
        eval_data.append({
            'Model': r['model'],
            f'Precision@{k_value}': f"{r['avg_precision_at_k']:.4f}",
            f'Recall@{k_value}': f"{r['avg_recall_at_k']:.4f}",
            'RMSE': f"{r.get('rmse', 0):.4f}",
            'MAE': f"{r.get('mae', 0):.4f}",
            f'NDCG@{k_value}': f"{r['avg_ndcg_at_k']:.4f}",
            'Samples': r['samples_tested']
        })
    
    eval_df = pd.DataFrame(eval_data)
    
    # Style the dataframe
    st.dataframe(
        eval_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Model': st.column_config.TextColumn('Model', width='large'),
            f'Precision@{k_value}': st.column_config.TextColumn(f'Precision@{k_value}', width='medium'),
            f'Recall@{k_value}': st.column_config.TextColumn(f'Recall@{k_value}', width='medium'),
            'RMSE': st.column_config.TextColumn('RMSE', width='small'),
            'MAE': st.column_config.TextColumn('MAE', width='small'),
        }
    )
    
    # Explanation
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Metrics Explanation:**
        
        | Metric | Meaning | Better |
        |--------|---------|---------|
        | **RMSE** | Root Mean Squared Error | Lower |
        | **MAE** | Mean Absolute Error | Lower |
        | **Precision@K** | % correct in top-K | Higher |
        | **Recall@K** | % relevant items found | Higher |
        | **NDCG@K** | Ranking quality | Higher |
        """)
    
    with col2:
        st.markdown("""
        **How to Read Results:**
        
        - **Low RMSE/MAE** = More accurate rating prediction
        - **High Precision** = More relevant recommendations
        - **High Recall** = Found more good items
        - **High NDCG** = Good items ranked at top
        """)
    
    st.divider()
    
    # Visualization: Precision/Recall chart
    st.subheader("Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Precision/Recall/NDCG chart
        fig = go.Figure()
        metrics = ['avg_precision_at_k', 'avg_recall_at_k', 'avg_ndcg_at_k']
        metric_names = [f'Precision@{k_value}', f'Recall@{k_value}', f'NDCG@{k_value}']
        colors = ['#667eea', '#38ef7d', '#764ba2']
        
        for metric, name, color in zip(metrics, metric_names, colors):
            fig.add_trace(go.Bar(
                x=[r['model'].split('(')[0].strip() for r in ranking_results],
                y=[r[metric] for r in ranking_results],
                name=name,
                marker_color=color
            ))
        
        fig.update_layout(
            barmode='group',
            title="Ranking Metrics (Higher = Better)",
            xaxis_title="Model",
            yaxis_title="Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # RMSE/MAE chart
        fig_error = go.Figure()
        
        fig_error.add_trace(go.Bar(
            x=[r['model'].split('(')[0].strip() for r in ranking_results],
            y=[r.get('rmse', 0) for r in ranking_results],
            name='RMSE',
            marker_color='#EF553B'
        ))
        
        fig_error.add_trace(go.Bar(
            x=[r['model'].split('(')[0].strip() for r in ranking_results],
            y=[r.get('mae', 0) for r in ranking_results],
            name='MAE',
            marker_color='#636EFA'
        ))
        
        fig_error.update_layout(
            barmode='group',
            title="Error Metrics (Lower = Better)",
            xaxis_title="Model",
            yaxis_title="Error Score",
            height=400
        )
        st.plotly_chart(fig_error, use_container_width=True)
    
    # Best model summary
    st.divider()
    st.subheader("Summary")
    
    # Find best model for each metric
    best_precision = max(ranking_results, key=lambda x: x['avg_precision_at_k'])
    best_rmse = min(ranking_results, key=lambda x: x.get('rmse', float('inf')))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"**Best Precision:** {best_precision['model'].split('(')[0]}")
    with col2:
        st.success(f"**Lowest RMSE:** {best_rmse['model'].split('(')[0]}")
    with col3:
        st.info("**Recommendation:** Hybrid Model for best balance")
    
    # Re-run evaluation hint
    st.divider()
    st.caption(
        "To re-run evaluation with different parameters, use: "
        "`python run_evaluation.py --samples N --k K`"
    )



# PAGE: ABOUT


def page_about(df_filtered):
    """Render the about page."""
    st.header("About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Steam Game Recommendation System
        
        A game recommendation system with advanced features.
        
        ### Dataset
        - **Source**: Steam Store Games (Kaggle)
        - **Games**: 2,000+ titles
        - **Features**: Rating, genres, price, playtime, descriptions...
        
        ### Algorithms
        
        #### 1. Content-Based Filtering
        - **TF-IDF Vectorization** for genres and categories
        - **Cosine Similarity** to find similar games
        
        #### 2. Hybrid Approach
        - **Content (60%) + Quality (40%)**
        - Balance between similarity and quality
        
        #### 3. Context-Aware
        - **Device compatibility** matching
        - **Time/Location/Weather/Mood** adjustments
        - User context personalization
        
        ### Evaluation Metrics
        - **RMSE/MAE**: Rating prediction evaluation
        - **Precision@K**: % correct recommendations
        - **Recall@K**: % good items found
        - **NDCG@K**: Ranking quality
        
        ### Tech Stack
        - **Backend**: Python, Pandas, Scikit-learn
        - **Frontend**: Streamlit
        - **Database**: MongoDB (optional) / JSON
        - **Visualization**: Plotly
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=150)
        
        st.markdown("""
        ### Statistics
        """)
        st.metric("Games Indexed", f"{len(df_filtered):,}")
        st.metric("Genres", f"{df_filtered['primary_genre'].nunique()}")
        st.metric("Algorithms", "4")
        st.metric("Metrics", "6+")



# MAIN APPLICATION


def main():
    """Main application entry point."""
    load_custom_css()
    
    # Check if logged in or guest mode
    if not SessionAuthManager.is_logged_in(st.session_state) and not st.session_state.get('guest_mode', False):
        render_auth_page()
        return
    
    # Render header
    render_header()
    
    # Load data
    with st.spinner("Loading Steam data..."):
        processor, df, df_filtered = load_data()
        recommender = initialize_recommender(df_filtered)
    
    # Get username if logged in
    username = None
    if SessionAuthManager.is_logged_in(st.session_state):
        username = SessionAuthManager.get_username(st.session_state)
    
    # Initialize user history
    user_history = get_user_history(username)
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select page:",
        ["Recommender", "Analytics", "Model Evaluation", "About"]
    )
    
    # Render user history in sidebar
    render_history_sidebar(user_history)
    
    # Route to appropriate page
    if page == "Recommender":
        page_recommender(recommender, df_filtered, user_history)
    elif page == "Analytics":
        page_data_analytics(processor, df_filtered)
    elif page == "Model Evaluation":
        page_model_evaluation(df_filtered, recommender)
    elif page == "About":
        page_about(df_filtered)



# RUN APPLICATION


if __name__ == "__main__":
    main()
