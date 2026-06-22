import os

# Absolute paths relative to this config file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_BASE_URL = ""
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MD_DIR = os.path.join(BASE_DIR, "md")
USE_CACHE = True
