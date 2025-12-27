"""
Device Configuration Module
===========================
Handles device configuration for context-aware game recommendations.
Matches game requirements with user's device specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class DeviceType(Enum):
    """Device type enumeration"""
    PC = "PC Desktop"
    LAPTOP = "Laptop"
    GAMING_LAPTOP = "Gaming Laptop"
    PHONE = "Phone"
    TABLET = "Tablet"
    CONSOLE = "Console (PS/Xbox)"
    STEAM_DECK = "Steam Deck"
    VR = "VR Headset"


class GPUTier(Enum):
    """GPU performance tier"""
    INTEGRATED = "Integrated (Intel HD/UHD)"
    LOW = "Entry Level (GT 1030, GTX 750)"
    MEDIUM = "Mid Range (GTX 1050-1660, RTX 2060)"
    HIGH = "High End (RTX 2070-3070, RTX 4060)"
    ULTRA = "Ultra (RTX 3080+, RTX 4070+)"


class CPUTier(Enum):
    """CPU performance tier"""
    LOW = "Entry Level (i3/Ryzen 3, older)"
    MEDIUM = "Mid Range (i5/Ryzen 5)"
    HIGH = "High End (i7/Ryzen 7)"
    ULTRA = "Ultra (i9/Ryzen 9)"


@dataclass
class DeviceConfig:
    """
    User's device configuration.
    
    Attributes:
        device_type: Type of device
        cpu: CPU name/model
        cpu_tier: CPU performance tier
        ram_gb: RAM in GB
        storage_gb: Storage in GB
        storage_type: SSD or HDD
        gpu: GPU name/model
        gpu_tier: GPU performance tier
        has_dedicated_gpu: Whether has dedicated GPU
        screen_resolution: Screen resolution (e.g., "1920x1080")
        vr_capable: VR capability
    """
    device_type: str = "PC Desktop"
    cpu: str = ""
    cpu_tier: str = "Mid Range (i5/Ryzen 5)"
    ram_gb: int = 8
    storage_gb: int = 256
    storage_type: str = "SSD"
    gpu: str = ""
    gpu_tier: str = "Mid Range (GTX 1050-1660, RTX 2060)"
    has_dedicated_gpu: bool = True
    screen_resolution: str = "1920x1080"
    vr_capable: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "device_type": self.device_type,
            "cpu": self.cpu,
            "cpu_tier": self.cpu_tier,
            "ram_gb": self.ram_gb,
            "storage_gb": self.storage_gb,
            "storage_type": self.storage_type,
            "gpu": self.gpu,
            "gpu_tier": self.gpu_tier,
            "has_dedicated_gpu": self.has_dedicated_gpu,
            "screen_resolution": self.screen_resolution,
            "vr_capable": self.vr_capable
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DeviceConfig':
        """Create from dictionary"""
        return cls(
            device_type=data.get("device_type", "PC Desktop"),
            cpu=data.get("cpu", ""),
            cpu_tier=data.get("cpu_tier", "Mid Range (i5/Ryzen 5)"),
            ram_gb=data.get("ram_gb", 8),
            storage_gb=data.get("storage_gb", 256),
            storage_type=data.get("storage_type", "SSD"),
            gpu=data.get("gpu", ""),
            gpu_tier=data.get("gpu_tier", "Mid Range (GTX 1050-1660, RTX 2060)"),
            has_dedicated_gpu=data.get("has_dedicated_gpu", True),
            screen_resolution=data.get("screen_resolution", "1920x1080"),
            vr_capable=data.get("vr_capable", False)
        )
    
    def get_performance_score(self) -> float:
        """
        Calculate overall performance score (0-100).
        
        Returns:
            Performance score
        """
        score = 0.0
        
        # CPU score (25 points)
        cpu_scores = {
            "Entry Level (i3/Ryzen 3, older)": 10,
            "Mid Range (i5/Ryzen 5)": 18,
            "High End (i7/Ryzen 7)": 23,
            "Ultra (i9/Ryzen 9)": 25
        }
        score += cpu_scores.get(self.cpu_tier, 15)
        
        # GPU score (35 points)
        gpu_scores = {
            "Integrated (Intel HD/UHD)": 5,
            "Entry Level (GT 1030, GTX 750)": 12,
            "Mid Range (GTX 1050-1660, RTX 2060)": 22,
            "High End (RTX 2070-3070, RTX 4060)": 30,
            "Ultra (RTX 3080+, RTX 4070+)": 35
        }
        if self.has_dedicated_gpu:
            score += gpu_scores.get(self.gpu_tier, 15)
        else:
            score += 5  # Integrated graphics
        
        # RAM score (20 points)
        if self.ram_gb >= 32:
            score += 20
        elif self.ram_gb >= 16:
            score += 16
        elif self.ram_gb >= 8:
            score += 12
        elif self.ram_gb >= 4:
            score += 6
        else:
            score += 3
        
        # Storage score (10 points)
        if self.storage_type == "SSD":
            score += 8
            if self.storage_gb >= 512:
                score += 2
        else:
            score += 4
            if self.storage_gb >= 1000:
                score += 1
        
        # Device type bonus (10 points)
        device_scores = {
            "PC Desktop": 10,
            "Gaming Laptop": 9,
            "Laptop": 7,
            "Steam Deck": 7,
            "Console (PS/Xbox)": 8,
            "VR Headset": 8,
            "Tablet": 4,
            "Phone": 3
        }
        score += device_scores.get(self.device_type, 5)
        
        return min(100, score)


class GameRequirements:
    """
    Represents game system requirements.
    """
    
    # Genre-based typical requirements
    GENRE_REQUIREMENTS = {
        "Action": {"min_ram": 8, "gpu_tier": "Medium", "storage": 50},
        "Adventure": {"min_ram": 8, "gpu_tier": "Medium", "storage": 40},
        "RPG": {"min_ram": 8, "gpu_tier": "Medium", "storage": 60},
        "Strategy": {"min_ram": 8, "gpu_tier": "Low", "storage": 30},
        "Simulation": {"min_ram": 8, "gpu_tier": "Medium", "storage": 40},
        "Racing": {"min_ram": 8, "gpu_tier": "Medium", "storage": 50},
        "Sports": {"min_ram": 8, "gpu_tier": "Medium", "storage": 50},
        "Puzzle": {"min_ram": 4, "gpu_tier": "Integrated", "storage": 5},
        "Casual": {"min_ram": 4, "gpu_tier": "Integrated", "storage": 5},
        "Indie": {"min_ram": 4, "gpu_tier": "Low", "storage": 10},
        "VR": {"min_ram": 16, "gpu_tier": "High", "storage": 30},
        "MMO": {"min_ram": 8, "gpu_tier": "Medium", "storage": 60},
        "Horror": {"min_ram": 8, "gpu_tier": "Medium", "storage": 40},
        "Shooter": {"min_ram": 8, "gpu_tier": "Medium", "storage": 50},
        "Fighting": {"min_ram": 8, "gpu_tier": "Medium", "storage": 40},
    }
    
    # GPU tier order for comparison
    GPU_TIER_ORDER = [
        "Integrated (Intel HD/UHD)",
        "Entry Level (GT 1030, GTX 750)",
        "Mid Range (GTX 1050-1660, RTX 2060)",
        "High End (RTX 2070-3070, RTX 4060)",
        "Ultra (RTX 3080+, RTX 4070+)"
    ]
    
    @classmethod
    def get_gpu_tier_index(cls, tier: str) -> int:
        """Get numeric index for GPU tier comparison"""
        try:
            return cls.GPU_TIER_ORDER.index(tier)
        except ValueError:
            return 2  # Default to mid-range
    
    @classmethod
    def get_requirements_for_genre(cls, genre: str) -> Dict:
        """Get typical requirements for a genre"""
        # Check for exact match first
        if genre in cls.GENRE_REQUIREMENTS:
            return cls.GENRE_REQUIREMENTS[genre]
        
        # Check for partial match
        for key, reqs in cls.GENRE_REQUIREMENTS.items():
            if key.lower() in genre.lower() or genre.lower() in key.lower():
                return reqs
        
        # Default requirements
        return {"min_ram": 8, "gpu_tier": "Medium", "storage": 30}


class DeviceCompatibilityChecker:
    """
    Checks game compatibility with user's device.
    """
    
    def __init__(self, device_config: DeviceConfig):
        """
        Initialize compatibility checker.
        
        Args:
            device_config: User's device configuration
        """
        self.config = device_config
        self.performance_score = device_config.get_performance_score()
    
    def check_game_compatibility(
        self,
        game_genres: str,
        game_name: str = "",
        estimated_size_gb: float = 30.0
    ) -> Tuple[bool, float, str]:
        """
        Check if a game is compatible with user's device.
        
        Args:
            game_genres: Comma-separated genre string
            game_name: Game name for special checks
            estimated_size_gb: Estimated game size in GB
        
        Returns:
            Tuple of (is_compatible, compatibility_score, message)
        """
        genres = [g.strip() for g in game_genres.split(',')]
        primary_genre = genres[0] if genres else "Indie"
        
        # Get requirements for primary genre
        requirements = GameRequirements.get_requirements_for_genre(primary_genre)
        
        # Check RAM
        ram_ok = self.config.ram_gb >= requirements["min_ram"]
        
        # Check GPU
        required_gpu_idx = self._gpu_tier_to_index(requirements["gpu_tier"])
        user_gpu_idx = GameRequirements.get_gpu_tier_index(self.config.gpu_tier)
        gpu_ok = user_gpu_idx >= required_gpu_idx or not self.config.has_dedicated_gpu
        
        # Check storage
        storage_ok = self.config.storage_gb >= requirements["storage"] * 1.5  # Leave some buffer
        
        # Check VR
        vr_ok = True
        if "VR" in genres or "vr" in game_name.lower():
            vr_ok = self.config.vr_capable and self.config.has_dedicated_gpu
        
        # Check device type compatibility
        device_ok = self._check_device_type_compatibility(genres)
        
        # Calculate compatibility score
        score = 0.0
        messages = []
        
        if ram_ok:
            score += 25
        else:
            messages.append(f"Requires at least {requirements['min_ram']}GB RAM")
        
        if gpu_ok:
            score += 35
        else:
            messages.append(f"Requires better GPU for {primary_genre} games")
        
        if storage_ok:
            score += 20
        else:
            messages.append(f"Needs more storage (approx {requirements['storage']}GB)")
        
        if vr_ok:
            score += 10
        else:
            messages.append("VR games require VR-capable device")
        
        if device_ok:
            score += 10
        else:
            messages.append(f"Device {self.config.device_type} may not be optimal")
        
        is_compatible = score >= 60
        message = " | ".join(messages) if messages else "[OK] Compatible with your device"
        
        return is_compatible, score, message
    
    def _gpu_tier_to_index(self, tier_name: str) -> int:
        """Convert GPU tier name to index"""
        tier_map = {
            "Integrated": 0,
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Ultra": 4
        }
        return tier_map.get(tier_name, 2)
    
    def _check_device_type_compatibility(self, genres: List[str]) -> bool:
        """Check if device type is suitable for genres"""
        mobile_friendly = ["Casual", "Puzzle", "Card", "Board", "Indie"]
        heavy_genres = ["VR", "Racing", "Simulation", "RPG", "MMO"]
        
        if self.config.device_type in ["Phone", "Tablet"]:
            return any(g in mobile_friendly for g in genres)
        
        if self.config.device_type == "VR Headset":
            return "VR" in genres
        
        return True
    
    def get_recommended_settings(self) -> Dict[str, str]:
        """
        Get recommended graphics settings based on device.
        
        Returns:
            Dictionary with recommended settings
        """
        score = self.performance_score
        
        if score >= 85:
            return {
                "quality": "Ultra",
                "resolution": "4K (3840x2160)",
                "fps_target": "144+",
                "ray_tracing": "On",
                "description": "Your device can run most games at highest settings"
            }
        elif score >= 70:
            return {
                "quality": "High",
                "resolution": "1440p (2560x1440)",
                "fps_target": "60-144",
                "ray_tracing": "Medium",
                "description": "Can play most games at high settings"
            }
        elif score >= 50:
            return {
                "quality": "Medium",
                "resolution": "1080p (1920x1080)",
                "fps_target": "60",
                "ray_tracing": "Off",
                "description": "Suitable for games at medium settings"
            }
        elif score >= 30:
            return {
                "quality": "Low",
                "resolution": "720p (1280x720)",
                "fps_target": "30-60",
                "ray_tracing": "Off",
                "description": "Should choose lighter games or lower settings"
            }
        else:
            return {
                "quality": "Lowest",
                "resolution": "720p or lower",
                "fps_target": "30",
                "ray_tracing": "Not supported",
                "description": "Only suitable for casual, light indie games"
            }
    
    def filter_compatible_games(
        self,
        games: List[Dict],
        min_compatibility: float = 60.0
    ) -> List[Dict]:
        """
        Filter games by compatibility.
        
        Args:
            games: List of game dictionaries
            min_compatibility: Minimum compatibility score (0-100)
        
        Returns:
            Filtered list with compatibility info
        """
        compatible_games = []
        
        for game in games:
            genres = game.get('genres', '')
            name = game.get('name', '')
            
            is_compatible, score, message = self.check_game_compatibility(genres, name)
            
            if score >= min_compatibility:
                game_copy = game.copy()
                game_copy['compatibility_score'] = score
                game_copy['compatibility_message'] = message
                game_copy['is_fully_compatible'] = is_compatible
                compatible_games.append(game_copy)
        
        # Sort by compatibility score
        compatible_games.sort(key=lambda x: x['compatibility_score'], reverse=True)
        
        return compatible_games


# Helper functions for Streamlit forms

def get_device_types() -> List[str]:
    """Get list of device types for dropdown"""
    return [dt.value for dt in DeviceType]


def get_gpu_tiers() -> List[str]:
    """Get list of GPU tiers for dropdown"""
    return [gt.value for gt in GPUTier]


def get_cpu_tiers() -> List[str]:
    """Get list of CPU tiers for dropdown"""
    return [ct.value for ct in CPUTier]


def get_storage_types() -> List[str]:
    """Get storage type options"""
    return ["SSD", "HDD", "NVMe SSD"]


def get_resolutions() -> List[str]:
    """Get common screen resolutions"""
    return [
        "1280x720 (HD)",
        "1920x1080 (Full HD)",
        "2560x1440 (2K/QHD)",
        "3840x2160 (4K/UHD)",
        "Other"
    ]


print("[OK] DeviceConfig and DeviceCompatibilityChecker classes ready!")
