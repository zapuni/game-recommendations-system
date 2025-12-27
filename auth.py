"""
Authentication Module
=====================
Simple authentication system with MongoDB support.
Features:
- User registration/login
- Password hashing
- Session management
- Fallback to JSON if MongoDB unavailable
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path

# Try to import pymongo
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("Note: pymongo not installed. Using JSON storage.")


class AuthManager:
    """
    Manages user authentication with MongoDB or JSON fallback.
    
    Features:
    - User registration with password hashing
    - User login validation
    - Session management
    - User profile management
    """
    
    # JSON fallback storage
    STORAGE_DIR = Path("data/users")
    USERS_FILE = STORAGE_DIR / "users.json"
    
    def __init__(self, mongodb_uri: Optional[str] = None, db_name: str = "steam_recommender"):
        """
        Initialize authentication manager.
        
        Args:
            mongodb_uri: MongoDB connection URI (optional)
            db_name: Database name
        """
        self.use_mongodb = False
        self.db = None
        self.users_collection = None
        
        # Try MongoDB connection
        if mongodb_uri and MONGODB_AVAILABLE:
            try:
                self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=3000)
                # Test connection
                self.client.admin.command('ping')
                self.db = self.client[db_name]
                self.users_collection = self.db["users"]
                self.use_mongodb = True
                print("[OK] Connected to MongoDB for authentication")
            except Exception as e:
                print(f"[WARNING] MongoDB connection failed: {e}")
                print("  -> Using JSON storage instead")
                self.use_mongodb = False
        
        # Setup JSON fallback
        if not self.use_mongodb:
            self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            if not self.USERS_FILE.exists():
                self._save_users_json({})
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users_json(self) -> Dict:
        """Load users from JSON file"""
        try:
            with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_users_json(self, users: Dict) -> None:
        """Save users to JSON file"""
        with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
    
    def register(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Unique username
            password: User password
            email: Optional email address
            display_name: Optional display name
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Validate inputs
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if not password or len(password) < 4:
            return False, "Password must be at least 4 characters"
        
        # Check if user exists
        if self.user_exists(username):
            return False, "Username already exists"
        
        # Create user document
        user_doc = {
            "username": username.lower(),
            "password_hash": self._hash_password(password),
            "email": email,
            "display_name": display_name or username,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "profile": {
                "favorite_genres": [],
                "device_config": {},
                "preferences": {}
            },
            "stats": {
                "games_viewed": 0,
                "recommendations_clicked": 0,
                "total_sessions": 0
            }
        }
        
        if self.use_mongodb:
            self.users_collection.insert_one(user_doc)
        else:
            users = self._load_users_json()
            users[username.lower()] = user_doc
            self._save_users_json(users)
        
        return True, "Registration successful! Please login."
    
    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Authenticate user.
        
        Args:
            username: Username
            password: Password
        
        Returns:
            Tuple of (success: bool, message: str, user_data: Dict or None)
        """
        if not username or not password:
            return False, "Please enter all required fields", None
        
        password_hash = self._hash_password(password)
        username_lower = username.lower()
        
        if self.use_mongodb:
            user = self.users_collection.find_one({
                "username": username_lower,
                "password_hash": password_hash
            })
            
            if user:
                # Update last login
                self.users_collection.update_one(
                    {"username": username_lower},
                    {"$set": {"last_login": datetime.now().isoformat()}}
                )
                user.pop('_id', None)
                user.pop('password_hash', None)
                return True, "Login successful!", user
        else:
            users = self._load_users_json()
            if username_lower in users:
                user = users[username_lower]
                if user['password_hash'] == password_hash:
                    # Update last login
                    user['last_login'] = datetime.now().isoformat()
                    users[username_lower] = user
                    self._save_users_json(users)
                    
                    user_copy = user.copy()
                    user_copy.pop('password_hash', None)
                    return True, "Login successful!", user_copy
        
        return False, "Invalid username or password", None
    
    def user_exists(self, username: str) -> bool:
        """Check if username exists"""
        username_lower = username.lower()
        
        if self.use_mongodb:
            return self.users_collection.find_one({"username": username_lower}) is not None
        else:
            users = self._load_users_json()
            return username_lower in users
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user data by username"""
        username_lower = username.lower()
        
        if self.use_mongodb:
            user = self.users_collection.find_one({"username": username_lower})
            if user:
                user.pop('_id', None)
                user.pop('password_hash', None)
                return user
        else:
            users = self._load_users_json()
            if username_lower in users:
                user = users[username_lower].copy()
                user.pop('password_hash', None)
                return user
        
        return None
    
    def update_user_profile(self, username: str, profile_data: Dict) -> bool:
        """Update user profile data"""
        username_lower = username.lower()
        
        if self.use_mongodb:
            result = self.users_collection.update_one(
                {"username": username_lower},
                {"$set": {"profile": profile_data}}
            )
            return result.modified_count > 0
        else:
            users = self._load_users_json()
            if username_lower in users:
                users[username_lower]['profile'] = profile_data
                self._save_users_json(users)
                return True
        
        return False
    
    def update_device_config(self, username: str, device_config: Dict) -> bool:
        """Update user's device configuration"""
        username_lower = username.lower()
        
        if self.use_mongodb:
            result = self.users_collection.update_one(
                {"username": username_lower},
                {"$set": {"profile.device_config": device_config}}
            )
            return result.modified_count > 0
        else:
            users = self._load_users_json()
            if username_lower in users:
                if 'profile' not in users[username_lower]:
                    users[username_lower]['profile'] = {}
                users[username_lower]['profile']['device_config'] = device_config
                self._save_users_json(users)
                return True
        
        return False
    
    def get_device_config(self, username: str) -> Dict:
        """Get user's device configuration"""
        user = self.get_user(username)
        if user and 'profile' in user:
            return user['profile'].get('device_config', {})
        return {}
    
    def update_stats(self, username: str, stat_key: str, increment: int = 1) -> bool:
        """Increment a user statistic"""
        username_lower = username.lower()
        
        if self.use_mongodb:
            result = self.users_collection.update_one(
                {"username": username_lower},
                {"$inc": {f"stats.{stat_key}": increment}}
            )
            return result.modified_count > 0
        else:
            users = self._load_users_json()
            if username_lower in users:
                if 'stats' not in users[username_lower]:
                    users[username_lower]['stats'] = {}
                current = users[username_lower]['stats'].get(stat_key, 0)
                users[username_lower]['stats'][stat_key] = current + increment
                self._save_users_json(users)
                return True
        
        return False
    
    def logout(self, session_state) -> None:
        """Clear user session"""
        keys_to_remove = ['logged_in', 'username', 'user_data', 'user_history']
        for key in keys_to_remove:
            if key in session_state:
                del session_state[key]


class SessionAuthManager:
    """
    Manages authentication state in Streamlit session.
    """
    
    @staticmethod
    def init_session(session_state, auth_manager: AuthManager) -> None:
        """Initialize session state for auth"""
        if 'auth_manager' not in session_state:
            session_state['auth_manager'] = auth_manager
        if 'logged_in' not in session_state:
            session_state['logged_in'] = False
        if 'username' not in session_state:
            session_state['username'] = None
        if 'user_data' not in session_state:
            session_state['user_data'] = None
    
    @staticmethod
    def is_logged_in(session_state) -> bool:
        """Check if user is logged in"""
        return session_state.get('logged_in', False)
    
    @staticmethod
    def get_username(session_state) -> Optional[str]:
        """Get current username"""
        return session_state.get('username', None)
    
    @staticmethod
    def get_user_data(session_state) -> Optional[Dict]:
        """Get current user data"""
        return session_state.get('user_data', None)
    
    @staticmethod
    def login_user(session_state, username: str, user_data: Dict) -> None:
        """Set logged in state"""
        session_state['logged_in'] = True
        session_state['username'] = username
        session_state['user_data'] = user_data
    
    @staticmethod
    def logout_user(session_state) -> None:
        """Clear login state"""
        session_state['logged_in'] = False
        session_state['username'] = None
        session_state['user_data'] = None
        if 'user_history' in session_state:
            del session_state['user_history']


print("[OK] AuthManager and SessionAuthManager classes ready!")
