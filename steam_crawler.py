import requests
import csv
import time
import os
import json
import argparse
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dotenv import load_dotenv
load_dotenv()
STEAM_API_KEY = os.getenv("STEAM_API_KEY")


class SteamCrawler:
    def __init__(self):
        self.store_url = "https://store.steampowered.com/api/appdetails"
        self.spy_url = "https://steamspy.com/api.php"
        self.app_list_url = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
        self.output_dir = "data/steam-store-games"
        self.applist_file = "data/steam_all_app.csv"
        
        # Danh sách tất cả tags có thể có (dựa trên form)
        self.all_tags = [
            '1980s', '1990s', '2.5d', '2d', '2d_fighter', '360_video', '3d', '3d_platformer', '3d_vision', 
            '4_player_local', '4x', '6dof', 'atv', 'abstract', 'action', 'action_rpg', 'action_adventure', 
            'addictive', 'adventure', 'agriculture', 'aliens', 'alternate_history', 'america', 
            'animation_&_modeling', 'anime', 'arcade', 'arena_shooter', 'artificial_intelligence', 'assassin',
            'asynchronous_multiplayer', 'atmospheric', 'audio_production', 'bmx', 'base_building', 'baseball',
            'based_on_a_novel', 'basketball', 'batman', 'battle_royale', 'beat_em_up', 'beautiful', 'benchmark',
            'bikes', 'blood', 'board_game', 'bowling', 'building', 'bullet_hell', 'bullet_time', 'crpg',
            'capitalism', 'card_game', 'cartoon', 'cartoony', 'casual', 'cats', 'character_action_game',
            'character_customization', 'chess', 'choices_matter', 'choose_your_own_adventure', 'cinematic',
            'city_builder', 'class_based', 'classic', 'clicker', 'co_op', 'co_op_campaign', 'cold_war',
            'colorful', 'comedy', 'comic_book', 'competitive', 'conspiracy', 'controller', 'conversation',
            'crafting', 'crime', 'crowdfunded', 'cult_classic', 'cute', 'cyberpunk', 'cycling', 'dark',
            'dark_comedy', 'dark_fantasy', 'dark_humor', 'dating_sim', 'demons', 'design_&_illustration',
            'destruction', 'detective', 'difficult', 'dinosaurs', 'diplomacy', 'documentary', 'dog', 'dragons',
            'drama', 'driving', 'dungeon_crawler', 'dungeons_&_dragons', 'dynamic_narration', 'dystopian_',
            'early_access', 'economy', 'education', 'emotional', 'epic', 'episodic', 'experience', 'experimental',
            'exploration', 'fmv', 'fps', 'faith', 'family_friendly', 'fantasy', 'fast_paced', 'feature_film',
            'female_protagonist', 'fighting', 'first_person', 'fishing', 'flight', 'football', 'foreign',
            'free_to_play', 'funny', 'futuristic', 'gambling', 'game_development', 'gamemaker', 'games_workshop',
            'gaming', 'god_game', 'golf', 'gore', 'gothic', 'grand_strategy', 'great_soundtrack',
            'grid_based_movement', 'gun_customization', 'hack_and_slash', 'hacking', 'hand_drawn', 'hardware',
            'heist', 'hex_grid', 'hidden_object', 'historical', 'hockey', 'horror', 'horses', 'hunting',
            'illuminati', 'indie', 'intentionally_awkward_controls', 'interactive_fiction', 'inventory_management',
            'investigation', 'isometric', 'jrpg', 'jet', 'kickstarter', 'lego', 'lara_croft', 'lemmings',
            'level_editor', 'linear', 'local_co_op', 'local_multiplayer', 'logic', 'loot', 'lore_rich',
            'lovecraftian', 'mmorpg', 'moba', 'magic', 'management', 'mars', 'martial_arts', 'massively_multiplayer',
            'masterpiece', 'match_3', 'mature', 'mechs', 'medieval', 'memes', 'metroidvania', 'military',
            'mini_golf', 'minigames', 'minimalist', 'mining', 'mod', 'moddable', 'modern', 'motocross',
            'motorbike', 'mouse_only', 'movie', 'multiplayer', 'multiple_endings', 'music',
            'music_based_procedural_generation', 'mystery', 'mystery_dungeon', 'mythology', 'nsfw', 'narration',
            'naval', 'ninja', 'noir', 'nonlinear', 'nudity', 'offroad', 'old_school', 'on_rails_shooter',
            'online_co_op', 'open_world', 'otome', 'parkour', 'parody_', 'party_based_rpg', 'perma_death',
            'philisophical', 'photo_editing', 'physics', 'pinball', 'pirates', 'pixel_graphics', 'platformer',
            'point_&_click', 'political', 'politics', 'pool', 'post_apocalyptic', 'procedural_generation',
            'programming', 'psychedelic', 'psychological', 'psychological_horror', 'puzzle', 'puzzle_platformer',
            'pve', 'pvp', 'quick_time_events', 'rpg', 'rpgmaker', 'rts', 'racing', 'real_time_tactics',
            'real_time', 'real_time_with_pause', 'realistic', 'relaxing', 'remake', 'replay_value',
            'resource_management', 'retro', 'rhythm', 'robots', 'rogue_like', 'rogue_lite', 'romance', 'rome',
            'runner', 'sailing', 'sandbox', 'satire', 'sci_fi', 'science', 'score_attack', 'sequel',
            'sexual_content', 'shoot_em_up', 'shooter', 'short', 'side_scroller', 'silent_protagonist',
            'simulation', 'singleplayer', 'skateboarding', 'skating', 'skiing', 'sniper', 'snow', 'snowboarding',
            'soccer', 'software', 'software_training', 'sokoban', 'souls_like', 'soundtrack', 'space',
            'space_sim', 'spectacle_fighter', 'spelling', 'split_screen', 'sports', 'star_wars', 'stealth',
            'steam_machine', 'steampunk', 'story_rich', 'strategy', 'strategy_rpg', 'stylized', 'submarine',
            'superhero', 'supernatural', 'surreal', 'survival', 'survival_horror', 'swordplay', 'tactical',
            'tactical_rpg', 'tanks', 'team_based', 'tennis', 'text_based', 'third_person', 'third_person_shooter',
            'thriller', 'time_attack', 'time_management', 'time_manipulation', 'time_travel', 'top_down',
            'top_down_shooter', 'touch_friendly', 'tower_defense', 'trackir', 'trading', 'trading_card_game',
            'trains', 'transhumanism', 'turn_based', 'turn_based_combat', 'turn_based_strategy',
            'turn_based_tactics', 'tutorial', 'twin_stick_shooter', 'typing', 'underground', 'underwater',
            'unforgiving', 'utilities', 'vr', 'vr_only', 'vampire', 'video_production', 'villain_protagonist',
            'violent', 'visual_novel', 'voice_control', 'voxel', 'walking_simulator', 'war', 'wargame',
            'warhammer_40k', 'web_publishing', 'werewolves', 'western', 'word_game', 'world_war_i',
            'world_war_ii', 'wrestling', 'zombies', 'e_sports'
        ]
        
    def save_applist_to_csv(self, apps: List[Dict]):
        """Lưu danh sách AppList vào file CSV"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        with open(self.applist_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['appid', 'name'])
            for app in apps:
                writer.writerow([app['appid'], app['name']])
        
        print(f"✓ Đã lưu {len(apps)} games vào {self.applist_file}")
    
    def load_applist_from_csv(self, limit: Optional[int] = None) -> List[Dict]:
        """Đọc danh sách AppList từ file CSV"""
        if not os.path.exists(self.applist_file):
            print(f"✗ Không tìm thấy file {self.applist_file}")
            return []
        
        apps = []
        try:
            with open(self.applist_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        apps.append({'appid': int(row['appid']), 'name': row['name']})
                        if limit and len(apps) >= limit:
                            break
                    except (ValueError, KeyError) as e:
                        continue  # Skip invalid rows
        except Exception as e:
            print(f"✗ Lỗi đọc file {self.applist_file}: {e}")
            return []
        
        print(f"✓ Đã đọc {len(apps)} games từ {self.applist_file}")
        return apps
    
    def get_crawled_appids(self) -> Set[int]:
        """Lấy danh sách appid đã crawl từ file steam.csv"""
        crawled = set()
        steam_csv = f'{self.output_dir}/steam.csv'
        
        if not os.path.exists(steam_csv):
            return crawled
        
        try:
            with open(steam_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        appid = int(row['appid'])
                        crawled.add(appid)
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f"⚠ Lỗi đọc file đã crawl: {e}")
        
        return crawled
    
    def get_all_appids(self, limit: Optional[int] = None, max_retries: int = 3) -> List[Dict]:
        """Lấy danh sách appid từ Steam sử dụng IStoreService API v1 với pagination"""
        print("Đang lấy danh sách tất cả game từ Steam...")
        
        all_apps = []
        last_appid = 0
        max_results = 50000  # Số tối đa mỗi request
        
        for attempt in range(max_retries):
            try:
                # Sử dụng API mới với pagination
                params = {
                    'key': STEAM_API_KEY,
                    'max_results': max_results,
                    'include_games': 'true',
                    'include_dlc': 'false',
                    'include_software': 'false',
                    'include_videos': 'false',
                    'include_hardware': 'false',
                    'last_appid': last_appid
                }
                
                response = requests.get(self.app_list_url, params=params, timeout=30)
                response.raise_for_status()
                
                if response.text:
                    data = response.json()
                    
                    # API v1 trả về response.apps (array of {appid, name})
                    apps = data.get('response', {}).get('apps', [])
                    
                    if not apps:
                        # Không còn dữ liệu
                        break
                    
                    # Lọc chỉ lấy các apps có name
                    apps = [{'appid': app['appid'], 'name': app.get('name', '')} 
                            for app in apps if app.get('name')]
                    
                    all_apps.extend(apps)
                    print(f"  Đã tải {len(all_apps)} games...")
                    
                    # Nếu đã đủ limit, dừng lại
                    if limit and len(all_apps) >= limit:
                        all_apps = all_apps[:limit]
                        break
                    
                    # Nếu nhận đủ max_results, còn nhiều dữ liệu -> tiếp tục pagination
                    if len(apps) < max_results:
                        # Đã hết dữ liệu
                        break
                    
                    # Cập nhật last_appid để lấy batch tiếp theo
                    last_appid = apps[-1]['appid']
                    time.sleep(0.5)  # Delay nhẹ giữa các batch
                    
                else:
                    print(f"  Response rỗng, thử lại ({attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    
            except requests.exceptions.RequestException as e:
                print(f"  Lỗi kết nối: {e}, thử lại ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            except json.JSONDecodeError as e:
                print(f"  Lỗi parse JSON: {e}, thử lại ({attempt + 1}/{max_retries})...")
                time.sleep(2)
        
        if all_apps:
            print(f"Tìm thấy tổng cộng {len(all_apps)} games có tên")
            return all_apps
        
        # Nếu không lấy được từ API, sử dụng danh sách game phổ biến làm fallback
        print("Không thể lấy danh sách từ Steam API, sử dụng danh sách game phổ biến...")
        popular_games = [
            {"appid": 730, "name": "Counter-Strike 2"},
            {"appid": 570, "name": "Dota 2"},
            {"appid": 440, "name": "Team Fortress 2"},
            {"appid": 1172470, "name": "Apex Legends"},
            {"appid": 578080, "name": "PUBG: BATTLEGROUNDS"},
            {"appid": 1245620, "name": "ELDEN RING"},
            {"appid": 1091500, "name": "Cyberpunk 2077"},
            {"appid": 292030, "name": "The Witcher 3: Wild Hunt"},
            {"appid": 1174180, "name": "Red Dead Redemption 2"},
            {"appid": 413150, "name": "Stardew Valley"},
            {"appid": 105600, "name": "Terraria"},
            {"appid": 252490, "name": "Rust"},
            {"appid": 271590, "name": "Grand Theft Auto V"},
            {"appid": 381210, "name": "Dead by Daylight"},
            {"appid": 1085660, "name": "Destiny 2"},
            {"appid": 230410, "name": "Warframe"},
            {"appid": 1938090, "name": "Call of Duty"},
            {"appid": 1599340, "name": "Lost Ark"},
            {"appid": 394360, "name": "Hearts of Iron IV"},
            {"appid": 289070, "name": "Civilization VI"},
            {"appid": 550, "name": "Left 4 Dead 2"},
            {"appid": 620, "name": "Portal 2"},
            {"appid": 4000, "name": "Garry's Mod"},
            {"appid": 322330, "name": "Don't Starve Together"},
            {"appid": 346110, "name": "ARK: Survival Evolved"},
            {"appid": 359550, "name": "Tom Clancy's Rainbow Six Siege"},
            {"appid": 1238810, "name": "Battlefield 1"},
            {"appid": 1203220, "name": "NARAKA: BLADEPOINT"},
            {"appid": 892970, "name": "Valheim"},
            {"appid": 1517290, "name": "Battlefield 2042"},
            {"appid": 252950, "name": "Rocket League"},
            {"appid": 431960, "name": "Wallpaper Engine"},
            {"appid": 242760, "name": "The Forest"},
            {"appid": 1623730, "name": "Palworld"},
            {"appid": 1966720, "name": "Lethal Company"},
            {"appid": 367520, "name": "Hollow Knight"},
            {"appid": 814380, "name": "Sekiro: Shadows Die Twice"},
            {"appid": 601150, "name": "Devil May Cry 5"},
            {"appid": 1145360, "name": "Hades"},
            {"appid": 1817070, "name": "Monster Hunter Rise"},
            {"appid": 582010, "name": "Monster Hunter: World"},
            {"appid": 1238840, "name": "Battlefield V"},
            {"appid": 1063730, "name": "New World"},
            {"appid": 548430, "name": "Deep Rock Galactic"},
            {"appid": 1172380, "name": "Star Wars Jedi: Fallen Order"},
            {"appid": 1151640, "name": "Horizon Zero Dawn"},
            {"appid": 1222670, "name": "The Sims 4"},
            {"appid": 1286830, "name": "STAR WARS: Battlefront II"},
            {"appid": 1237970, "name": "Titanfall 2"},
            {"appid": 1284210, "name": "Guilty Gear -Strive-"},
        ]
        
        if limit:
            return popular_games[:limit]
        return popular_games

    def fetch_game_data(self, appid: int) -> Optional[Tuple[Dict, Dict]]:
        """Kết hợp dữ liệu từ Store API và SteamSpy"""
        try:
            # 1. Lấy dữ liệu từ Steam Store
            params = {"appids": appid, "l": "english"}
            store_res = requests.get(self.store_url, params=params, timeout=10).json()
            
            if not store_res or str(appid) not in store_res or not store_res[str(appid)]['success']:
                return None
            
            data = store_res[str(appid)]['data']
            
            # 2. Lấy dữ liệu từ SteamSpy (Tags, Ratings, Owners)
            spy_params = {"request": "appdetails", "appid": appid}
            spy_data = requests.get(self.spy_url, params=spy_params, timeout=10).json()
            
            return data, spy_data
        except Exception as e:
            print(f"  Lỗi khi crawl appid {appid}: {e}")
            return None

    def normalize_tag_name(self, tag: str) -> str:
        """Chuẩn hóa tên tag để khớp với danh sách all_tags"""
        # Chuyển sang lowercase, thay space bằng underscore
        normalized = tag.lower().strip().replace(' ', '_').replace('-', '_')
        # Xử lý các ký tự đặc biệt
        normalized = normalized.replace('&', '&')
        return normalized
    
    def clean_text(self, text: str) -> str:
        """Làm sạch text để tránh lỗi CSV (xử lý newline, special chars)"""
        if not text:
            return ""
        try:
            # Thay thế các ký tự xuống dòng bằng space
            text = str(text).replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
            # Loại bỏ multiple spaces
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        except Exception:
            return ""
    
    def safe_get(self, data: any, key: str, default: any = '') -> any:
        """An toàn lấy giá trị từ dict, xử lý trường hợp data không phải dict"""
        if isinstance(data, dict):
            return data.get(key, default)
        return default
    
    def safe_join(self, items: any, separator: str = ';') -> str:
        """An toàn join list thành string"""
        if not items:
            return ""
        if isinstance(items, list):
            return separator.join([str(item) for item in items if item])
        return str(items)

    def save_to_csv(self, game_list: List[Tuple[Dict, Dict]], append_mode: bool = False):
        """Lưu dữ liệu vào các file CSV"""
        if not game_list:
            return
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        mode = 'a' if append_mode else 'w'
        write_header = not append_mode
        
        try:
            # 1. steam.csv
            with open(f'{self.output_dir}/steam.csv', mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        'appid', 'name', 'release_date', 'english', 'developer', 'publisher', 
                        'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 
                        'achievements', 'positive_ratings', 'negative_ratings', 'average_playtime', 
                        'median_playtime', 'owners', 'price'
                    ])
                
                for store_data, spy_data in game_list:
                    try:
                        platforms_dict = self.safe_get(store_data, 'platforms', {})
                        if isinstance(platforms_dict, dict):
                            platforms = ";".join([k for k, v in platforms_dict.items() if v])
                        else:
                            platforms = ""
                        
                        categories_list = self.safe_get(store_data, 'categories', [])
                        if isinstance(categories_list, list):
                            categories = ";".join([self.safe_get(c, 'description', '') for c in categories_list])
                        else:
                            categories = ""
                        
                        genres_list = self.safe_get(store_data, 'genres', [])
                        if isinstance(genres_list, list):
                            genres = ";".join([self.safe_get(g, 'description', '') for g in genres_list])
                        else:
                            genres = ""
                        
                        # SteamSpy tags
                        tags_data = self.safe_get(spy_data, 'tags', {})
                        if isinstance(tags_data, dict):
                            spy_tags = ";".join([str(tag) for tag in tags_data.keys()])
                        else:
                            spy_tags = ""
                        
                        # Price
                        price_overview = self.safe_get(store_data, 'price_overview', {})
                        if isinstance(price_overview, dict):
                            price = self.safe_get(price_overview, 'final', 0) / 100
                        else:
                            price = 0
                        
                        # Release date
                        release_info = self.safe_get(store_data, 'release_date', {})
                        if isinstance(release_info, dict):
                            coming_soon = self.safe_get(release_info, 'coming_soon', False)
                            release_date = self.safe_get(release_info, 'date', '') if not coming_soon else ''
                        else:
                            release_date = ""
                        
                        # Supported languages
                        supported_langs = str(self.safe_get(store_data, 'supported_languages', '')).lower()
                        english = 1 if 'english' in supported_langs else 0
                        
                        # Achievements
                        achievements_data = self.safe_get(store_data, 'achievements', {})
                        if isinstance(achievements_data, dict):
                            achievements = self.safe_get(achievements_data, 'total', 0)
                        else:
                            achievements = 0
                        
                        # Developers/Publishers
                        developers = self.safe_get(store_data, 'developers', [])
                        publishers = self.safe_get(store_data, 'publishers', [])
                        
                        writer.writerow([
                            self.safe_get(store_data, 'steam_appid', ''),
                            self.safe_get(store_data, 'name', ''),
                            release_date,
                            english,
                            self.safe_join(developers),
                            self.safe_join(publishers),
                            platforms,
                            self.safe_get(store_data, 'required_age', 0),
                            categories,
                            genres,
                            spy_tags,
                            achievements,
                            self.safe_get(spy_data, 'positive', 0),
                            self.safe_get(spy_data, 'negative', 0),
                            self.safe_get(spy_data, 'average_forever', 0),
                            self.safe_get(spy_data, 'median_forever', 0),
                            self.safe_get(spy_data, 'owners', ''),
                            price
                        ])
                    except Exception as e:
                        print(f"  ⚠ Lỗi ghi steam.csv cho appid {self.safe_get(store_data, 'steam_appid', '?')}: {e}")
                        continue
            
            # 2. steam_description_data.csv
            with open(f'{self.output_dir}/steam_description_data.csv', mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['steam_appid', 'detailed_description', 'about_the_game', 'short_description'])
                
                for store_data, spy_data in game_list:
                    try:
                        writer.writerow([
                            self.safe_get(store_data, 'steam_appid', ''),
                            self.clean_text(self.safe_get(store_data, 'detailed_description', '')),
                            self.clean_text(self.safe_get(store_data, 'about_the_game', '')),
                            self.clean_text(self.safe_get(store_data, 'short_description', ''))
                        ])
                    except Exception as e:
                        continue
            
            # 3. steam_media_data.csv
            with open(f'{self.output_dir}/steam_media_data.csv', mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['steam_appid', 'header_image', 'screenshots', 'background', 'movies'])
                
                for store_data, spy_data in game_list:
                    try:
                        screenshots_list = self.safe_get(store_data, 'screenshots', [])
                        if isinstance(screenshots_list, list):
                            screenshots = ";".join([self.safe_get(s, 'path_thumbnail', '') for s in screenshots_list])
                        else:
                            screenshots = ""
                        
                        movies_list = self.safe_get(store_data, 'movies', [])
                        movies_urls = []
                        if isinstance(movies_list, list):
                            for m in movies_list:
                                mp4_data = self.safe_get(m, 'mp4', {})
                                if isinstance(mp4_data, dict):
                                    url = self.safe_get(mp4_data, '480', '')
                                    if url:
                                        movies_urls.append(url)
                        movies = ";".join(movies_urls)
                        
                        writer.writerow([
                            self.safe_get(store_data, 'steam_appid', ''),
                            self.safe_get(store_data, 'header_image', ''),
                            screenshots,
                            self.safe_get(store_data, 'background', ''),
                            movies
                        ])
                    except Exception as e:
                        continue
            
            # 4. steam_requirements_data.csv
            with open(f'{self.output_dir}/steam_requirements_data.csv', mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['steam_appid', 'pc_requirements', 'mac_requirements', 'linux_requirements', 'minimum', 'recommended'])
                
                for store_data, spy_data in game_list:
                    try:
                        pc_req = self.safe_get(store_data, 'pc_requirements', {})
                        mac_req = self.safe_get(store_data, 'mac_requirements', {})
                        linux_req = self.safe_get(store_data, 'linux_requirements', {})
                        
                        # Xử lý trường hợp requirements là list
                        if not isinstance(pc_req, dict):
                            pc_req = {}
                        if not isinstance(mac_req, dict):
                            mac_req = {}
                        if not isinstance(linux_req, dict):
                            linux_req = {}
                        
                        # Clean từng phần trước khi nối chuỗi
                        pc_min = self.clean_text(self.safe_get(pc_req, 'minimum', ''))
                        pc_rec = self.clean_text(self.safe_get(pc_req, 'recommended', ''))
                        mac_min = self.clean_text(self.safe_get(mac_req, 'minimum', ''))
                        mac_rec = self.clean_text(self.safe_get(mac_req, 'recommended', ''))
                        linux_min = self.clean_text(self.safe_get(linux_req, 'minimum', ''))
                        linux_rec = self.clean_text(self.safe_get(linux_req, 'recommended', ''))
                        
                        pc_req_str = f"Minimum: {pc_min}; Recommended: {pc_rec}" if pc_req else ""
                        mac_req_str = f"Minimum: {mac_min}; Recommended: {mac_rec}" if mac_req else ""
                        linux_req_str = f"Minimum: {linux_min}; Recommended: {linux_rec}" if linux_req else ""
                        
                        writer.writerow([
                            self.safe_get(store_data, 'steam_appid', ''),
                            pc_req_str,
                            mac_req_str,
                            linux_req_str,
                            pc_min,
                            pc_rec
                        ])
                    except Exception as e:
                        continue
            
            # 5. steam_support_info.csv
            with open(f'{self.output_dir}/steam_support_info.csv', mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['steam_appid', 'website', 'support_url', 'support_email'])
                
                for store_data, spy_data in game_list:
                    try:
                        support_info = self.safe_get(store_data, 'support_info', {})
                        if not isinstance(support_info, dict):
                            support_info = {}
                        
                        writer.writerow([
                            self.safe_get(store_data, 'steam_appid', ''),
                            self.safe_get(store_data, 'website', ''),
                            self.safe_get(support_info, 'url', ''),
                            self.safe_get(support_info, 'email', '')
                        ])
                    except Exception as e:
                        continue
            
            # 6. steamspy_tag_data.csv
            with open(f'{self.output_dir}/steamspy_tag_data.csv', mode, newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['appid'] + self.all_tags)
                
                for store_data, spy_data in game_list:
                    try:
                        appid = self.safe_get(store_data, 'steam_appid', '')
                        game_tags = self.safe_get(spy_data, 'tags', {})
                        
                        # Xử lý trường hợp tags không phải dict
                        if not isinstance(game_tags, dict):
                            game_tags = {}
                        
                        normalized_game_tags = {}
                        for tag, count in game_tags.items():
                            try:
                                normalized_game_tags[self.normalize_tag_name(str(tag))] = count
                            except:
                                continue
                        
                        row = [appid]
                        for tag in self.all_tags:
                            row.append(1 if tag in normalized_game_tags else 0)
                        
                        writer.writerow(row)
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"✗ Lỗi lưu CSV: {e}")

    def run(self, limit: Optional[int] = 5000, crawl_applist: bool = False, skip_detail: bool = False):
        """Chạy crawler"""
        print("=" * 60)
        print("STEAM GAME CRAWLER")
        print("=" * 60)
        
        # Lấy danh sách apps
        if crawl_applist:
            print("\n→ Crawling AppList từ Steam API...")
            apps = self.get_all_appids(limit=limit)
            if apps:
                self.save_applist_to_csv(apps)
        else:
            print("\n→ Đọc AppList từ file CSV...")
            apps = self.load_applist_from_csv(limit=limit)
            if not apps:
                print("✗ Không có dữ liệu AppList. Hãy chạy với --crawl-applist để crawl mới.")
                return
        
        print(f"Tìm thấy {len(apps)} games để crawl\n")
        
        # Nếu chỉ crawl applist thì dừng lại
        if skip_detail:
            print("✓ Chỉ crawl AppList, bỏ qua chi tiết game.")
            return
        
        # Lấy danh sách đã crawl để skip
        crawled_appids = self.get_crawled_appids()
        if crawled_appids:
            print(f"✓ Đã tìm thấy {len(crawled_appids)} games đã crawl trước đó, sẽ tiếp tục từ vị trí cuối.\n")
        
        # Kiểm tra xem có cần tạo file mới hay append
        files_exist = os.path.exists(f'{self.output_dir}/steam.csv')
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for idx, app in enumerate(apps, 1):
            appid = app['appid']
            
            # Skip nếu đã crawl
            if appid in crawled_appids:
                skipped_count += 1
                continue
            
            print(f"[{idx}/{len(apps)}] Đang crawl: {app['name']} (ID: {appid})...", end=" ")
            
            try:
                result = self.fetch_game_data(appid)
                if result:
                    # Xác định append mode: append nếu đã có file hoặc đã có dữ liệu trước đó
                    append_mode = files_exist or success_count > 0
                    
                    # Lưu ngay từng game vào CSV
                    self.save_to_csv([result], append_mode=append_mode)
                    success_count += 1
                    print("✓")
                else:
                    error_count += 1
                    print("✗ (không có dữ liệu)")
            except KeyboardInterrupt:
                print("\n\n⚠ Đã dừng bởi người dùng (Ctrl+C)")
                print(f"Đã lưu {success_count} games. Chạy lại script để tiếp tục.")
                break
            except Exception as e:
                error_count += 1
                print(f"✗ (lỗi: {e})")
            
            # Delay để tránh bị block
            time.sleep(1.5)
        
        print("\n" + "=" * 60)
        print(f"HOÀN THÀNH!")
        print(f"- Thành công: {success_count} games")
        print(f"- Thất bại: {error_count} games")
        print(f"- Đã skip (crawl trước đó): {skipped_count} games")
        print(f"- Tổng cộng: {len(apps)} games")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steam Game Crawler')
    parser.add_argument('--limit', type=int, default=100, 
                        help='Số lượng game tối đa để crawl (mặc định: 100, 0 = tất cả)')
    parser.add_argument('--crawl-applist', action='store_true',
                        help='Crawl danh sách AppList mới từ Steam API')
    parser.add_argument('--applist-only', action='store_true',
                        help='Chỉ crawl AppList, không crawl chi tiết game')
    
    args = parser.parse_args()
    
    # Khởi tạo crawler
    crawler = SteamCrawler()
    
    # Xác định limit
    limit = None if args.limit == 0 else args.limit
    
    # Chạy crawler
    crawler.run(
        limit=limit,
        crawl_applist=args.crawl_applist,
        skip_detail=args.applist_only
    )
    
    # Ví dụ sử dụng:
    # python steam_crawler.py                        # Crawl 100 game từ CSV có sẵn
    # python steam_crawler.py --crawl-applist        # Crawl AppList mới + 100 game
    # python steam_crawler.py --crawl-applist --applist-only  # Chỉ crawl AppList
    # python steam_crawler.py --limit 1000           # Crawl 1000 game từ CSV
    # python steam_crawler.py --limit 0              # Crawl tất cả game
