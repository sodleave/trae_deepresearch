import os
import json
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger('DeepResearch')

class CacheManager:
    """
    Manage local caching of web content to avoid redundant fetching.
    """
    def __init__(self, cache_file="web_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Cache file {self.cache_file} is corrupted. Starting with empty cache.")
                return {}
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, url):
        """Get content for a URL if cached."""
        if url in self.cache:
            entry = self.cache[url]
            logger.debug(f"Cache hit for URL: {url}")
            return entry.get("content")
        return None

    def set(self, url, content):
        """Cache content for a URL."""
        if not url or not content:
            return
            
        self.cache[url] = {
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self._save_cache()
        logger.debug(f"Cached content for URL: {url}")

# Singleton instance
cache_manager = CacheManager()
