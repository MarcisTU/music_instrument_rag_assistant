from pathlib import Path


# the projects ./src dir
SRC_DIR = Path(__file__).resolve().parent.parent
BM25_CACHE_DIR = SRC_DIR / "data" / "bm25_cache" / "thomann_product_index_bm25"

PRODUCTS_JSONL_PATH = SRC_DIR / "data" / "products_scraper" / "thomann_products.jsonl"
