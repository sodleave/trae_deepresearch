import os
import json
import logging
import threading
import atexit
from datetime import datetime

logger = logging.getLogger('DeepResearch')

class CacheManager:
    """
    Manage local caching of web content to avoid redundant fetching.
    """
    def __init__(self, cache_file="web_cache.json"):
        self.cache_file = cache_file
        self._lock = threading.RLock()
        self._dirty_count = 0
        self._flush_every = 5
        self.cache = self._load_cache()
        atexit.register(self.flush)

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
            tmp_path = f"{self.cache_file}.tmp"
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.cache_file)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, url):
        """Get content for a URL if cached."""
        with self._lock:
            entry = self.cache.get(url)
        if entry:
            logger.debug(f"Cache hit for URL: {url}")
            return entry.get("content")
        return None

    def set(self, url, content):
        """Cache content for a URL."""
        if not url or not content:
            return

        with self._lock:
            self.cache[url] = {
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            self._dirty_count += 1
            should_flush = self._dirty_count >= self._flush_every

        if should_flush:
            self.flush()
        logger.debug(f"Cached content for URL: {url}")

    def flush(self):
        with self._lock:
            if self._dirty_count == 0:
                return
            self._save_cache()
            self._dirty_count = 0

# Singleton instance
cache_manager = CacheManager()
