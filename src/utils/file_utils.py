import json
from typing import List, Dict

from loguru import logger


class FileUtils:
    @staticmethod
    def load_jsonl_file(file_path: str) -> List[Dict]:
        """Loads all valid JSON lines from a file into an in-memory list."""
        logger.info(f"Loading data from file: {file_path}")
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    try:
                        samples.append(json.loads(clean_line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON on line {line_num}: {e}")
        except FileNotFoundError:
            logger.critical(f"Data file not found at path: {file_path}")
            raise

        logger.info(f"Successfully loaded {len(samples)} samples from JSONL file.")
        return samples
