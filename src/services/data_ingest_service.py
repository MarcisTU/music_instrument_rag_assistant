import asyncio
import json
import os
import re
import time
import httpx
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import requests
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from src.models.prompts import STRUCTURED_TEXT_TEMPLATE, AiProductDescription, LLM_DESCRIPTION_TEMPLATE
from src.models.schemas import ProductEmbeddingCreate, ProductCreate, ReviewCreate
from src.db.service import ProductService


class DataIngestService:
    def __init__(self):
        try:
            self.embedding_url = os.environ["EMBEDDING_API_URL"]
            self.llm_url = None
            self.llm_model = os.environ["LLM_MODEL"]
            self.llm_description_len_threshold = 10  # in tokens

            self.llm_client = OpenAI(
                base_url=os.environ["LLM_API_URL"],
                api_key="EMPTY"  # vLLM does not require a real key
            )

            self._warmup_services()

            logger.info("DataIngestService initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize DataIngestService: {e}")
            raise

    def _warmup_services(self):
        logger.info("Running LLM warmup test")

        start_time = time.time()
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system",
                 "content": """Generate an informative description summary about given product information."""},
                {"role": "user", "content": "Solar Guitars 6-string with Fishmann Fluence pickups."}
            ],
            max_tokens=400,
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "description",
                    "schema": AiProductDescription.model_json_schema()
                },
            }
        )
        end_time = time.time()
        logger.info(f"Elapsed time: {end_time - start_time:4f} seconds")

        generated_output = response.choices[0].message.content
        logger.debug(generated_output)

        logger.info("Performing test request for embedding service")

        response = requests.post(
            self.embedding_url,
            json={"input": ["I am testing embedding request."]},
            timeout=10.0
        )
        response.raise_for_status()
        response_data = response.json()
        logger.debug(np.array(response_data["data"][0]["embedding"]).shape)
        logger.debug(self.embedding_url)

    def load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads all valid JSON lines from a file into an in-memory list."""
        logger.info(f"Loading data from file: {file_path}")
        products = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    try:
                        products.append(json.loads(clean_line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON on line {line_num}: {e}")
        except FileNotFoundError:
            logger.critical(f"Data file not found at path: {file_path}")
            raise

        logger.info(f"Successfully loaded {len(products)} products from JSONL file.")
        return products

    def llm_generate_description(
            self,
            product: Dict[str, Any],
            product_description: str,
            main_category: str
    ) -> str:

        user_prompt = LLM_DESCRIPTION_TEMPLATE.format(
            name=product["name"],
            category=main_category,
            description=product_description
        )

        try:
            start_time = time.time()
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": """Generate a concise informative description summary about given product information."""},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "description",
                        "schema": AiProductDescription.model_json_schema()
                    },
                }
            )
            end_time = time.time()
            logger.info(f"Elapsed time: {end_time - start_time:4f} seconds")

            generated_output = response.choices[0].message.content

            # robust json parsing (vLLM sometimes spits out incomplete json response)
            try:
                generated_output_json = json.loads(generated_output)

                ai_product_description = AiProductDescription(**generated_output_json)
                description_out = ai_product_description.description
            except json.JSONDecodeError:
                logger.warning("Standard JSON parsing failed due to truncation. Extracting description via regex...")

                # This regex captures everything inside the quotes following "description":
                # It stops at the end of the string if the closing quote is missing.
                match = re.search(r'"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', generated_output)

                if match:
                    description_out = match.group(1)
                    logger.info("Successfully extracted partial description from truncated JSON.")
                else:
                    logger.error("Failed to extract description from malformed output.")
                    raise

            description_out = re.sub(r" {2,}", " ", description_out.strip())

            logger.debug(description_out)

            return description_out

        except Exception as e:
            logger.error(f"LLM product description generation failed: {e}")
            raise

    def build_structured_text(self, product: Dict[str, Any]) -> str:
        """Converts raw product fields into a structured string suitable for embeddings."""
        product_name = product["name"]
        product_price = product["price"]
        availability = "In Stock" if product["in_stock"] else "Out of Stock"

        points = product["description_points"]
        bullet_points_str = "\n".join([f"- {point}" for point in points])

        structured_text_description = STRUCTURED_TEXT_TEMPLATE.format(
            name=product_name,
            price=product_price,
            availability=availability,
            bullet_points=bullet_points_str
        )
        return structured_text_description

    def get_product_category(self, category_values: List[str]):
        filtered = [c for c in category_values if c != "All Categories"]

        # Slice the list to remove the last item (the brand/specific item) This leaves us with ['Software', 'Virtual Instruments + Samplers']
        structural_categories = filtered[0]
        category_str = " ".join(structural_categories).strip()

        return category_str

    async def get_embedding_async(self, client: httpx.AsyncClient, text: str) -> np.ndarray:
        try:
            response = await client.post(
                self.embedding_url,
                json={"input": [text]},
                timeout=10.0
            )
            response.raise_for_status()
            response_data = response.json()
            return np.array(response_data["data"][0]["embedding"])
        except Exception as e:
            logger.error(f"Embedding API async request failed: {e}")
            raise

    async def ingest_and_index_async(self, file_path: str):
        logger.info("Starting async data ingestion process...")

        # Fetch existing URLs that have been processed. Can be important for updating/skipping
        logger.info("Fetching existing product URLs from database...")
        existing_urls = await ProductService.get_all_product_urls()
        logger.info(f"Found {len(existing_urls)} existing products in database.")

        products_list = self.load_jsonl_file(file_path)

        total_product_count = len(products_list)

        product_emb_to_insert = []
        products_to_insert = []
        product_reviews_to_insert = []
        async with httpx.AsyncClient() as client:
            for idx, raw_product in tqdm(enumerate(products_list)):
                try:
                    # Check if the product has already been scraped and inserted. TODO in live data ingestion pipeline might want to update the data here
                    product_url = raw_product["url"]
                    if product_url in existing_urls:
                        continue

                    raw_product["price"] = raw_product["price"].strip().replace(" ", "")
                    raw_product["price"] = float(raw_product["price"].replace("€", "").replace(",", "."))
                    description = "\n".join(raw_product["description_points"])
                    category_str = self.get_product_category(raw_product["breadcrumbs"])

                    structured_text = self.build_structured_text(raw_product)
                    llm_description_text = self.llm_generate_description(raw_product, description, category_str)

                    if len(llm_description_text.split(" ")) <= self.llm_description_len_threshold:
                        logger.warning(f"Skipping short llm_description_text: len={len(llm_description_text.split(" "))}")
                        continue

                    embedding_task = self.get_embedding_async(client, structured_text)
                    llm_embedding_task = self.get_embedding_async(client, llm_description_text)
                    embedding_vector, llm_embedding_vector = await asyncio.gather(
                        embedding_task,
                        llm_embedding_task
                    )

                    product_record = ProductCreate(
                        name=raw_product["name"],
                        url=product_url,
                        description=description,
                        category_name=category_str,
                        price=raw_product["price"],
                        store_name="thomann",  # ! currently only one store
                        in_stock=raw_product["in_stock"],
                    )
                    product_embedding_record = ProductEmbeddingCreate(
                        name=raw_product["name"],
                        structured_text=structured_text,
                        generated_description=llm_description_text,
                        structured_embedding=embedding_vector,
                        generated_description_embedding=llm_embedding_vector
                    )

                    reviews = []
                    for rev in raw_product["reviews"]:
                        if len(rev["title"].strip()) == 0 or len(rev["author"].strip()) == 0 or len(rev["text"].strip()) == 0:
                            continue

                        review_create = ReviewCreate(
                            author=rev["author"],
                            title=rev["title"],
                            text=rev["text"],
                            rating=rev["rating"],
                            votes_up=rev["votes_up"],
                            votes_down=rev["votes_down"]
                        )
                        reviews.append(review_create)

                    # Perform data tracking after all requests are made
                    products_to_insert.append(product_record)
                    product_emb_to_insert.append(product_embedding_record)
                    product_reviews_to_insert.append(reviews if len(reviews) > 0 else [])  # keep length mapping

                    # Perform saving every nth item batch
                    if idx % 100 == 0:
                        inserted_product_ids = await ProductService.insert_products_with_embeddings_and_reviews(
                            products_create=products_to_insert,
                            embeddings_create=product_emb_to_insert,
                            reviews_create=product_reviews_to_insert
                        )
                        logger.info(f"Inserted {len(inserted_product_ids)} products into the database. {idx}/{total_product_count}")

                        # Next batch init
                        product_emb_to_insert = []
                        products_to_insert = []
                        product_reviews_to_insert = []

                except Exception as ex:
                    logger.error(f"Skipping product index {idx} due to processing error: {ex}")
                    continue

            if len(products_to_insert) > 0:
                inserted_product_ids = await ProductService.insert_products_with_embeddings_and_reviews(
                    products_create=products_to_insert,
                    embeddings_create=product_emb_to_insert,
                    reviews_create=product_reviews_to_insert
                )
                logger.info(f"Inserted {len(inserted_product_ids)} last products into database.")


async def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    # cached product data path. Feature ->> Could also perform the scraping overnight in background to check product updates in future!!
    file_path = f"{str(SCRIPT_DIR.parent)}/data/products_scraper/thomann_products.jsonl"

    ingest_service = DataIngestService()
    await ingest_service.ingest_and_index_async(file_path=file_path)


if __name__ == "__main__":
    asyncio.run(main())
