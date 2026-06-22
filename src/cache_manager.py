# cache_manager.py
import hashlib
import json
import os
from typing import Optional

import config


def get_cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def load_from_cache(query: str) -> Optional[str]:
    if not os.path.exists(config.CACHE_DIR):
        return None

    cache_key = get_cache_key(query)
    cache_file = os.path.join(config.CACHE_DIR, f"{cache_key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("answer")

    return None


def save_to_cache(query: str, answer: str):
    if not os.path.exists(config.CACHE_DIR):
        os.makedirs(config.CACHE_DIR)

    cache_key = get_cache_key(query)
    cache_file = os.path.join(config.CACHE_DIR, f"{cache_key}.json")

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"query": query, "answer": answer}, f, ensure_ascii=False, indent=2)
