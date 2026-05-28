import asyncio
import json
import os
import re
import time
import httpx
from typing import List

import numpy as np
from loguru import logger
from openai import AsyncOpenAI
from tqdm import tqdm
import bm25s

from src.models.enums import Store
from src.models.prompts import STRUCTURED_TEXT_TEMPLATE, AiProductDescription, LLM_DESCRIPTION_TEMPLATE, \
    LLM_SYS_PROMPT_DESCRIPTION
from src.models.schemas import ProductEmbeddingCreate, ProductCreate, ReviewCreate, ProductEmbeddingRead, ProductSource, \
    ProductReview
from src.db.service import ProductService
from src.modules.constants import BM25_CACHE_DIR, PRODUCTS_JSONL_PATH
from src.utils.file_utils import FileUtils


class DataIngestService:
    """
    Data Ingestion Service for processing and indexing embedded data into PGVector database

    Currently we process only Thomann store scraped products, but this could be extended in future to multiple stores.
    """

    def __init__(self):
        try:
            self.embedding_url = os.environ["EMBEDDING_API_URL"]
            self.llm_url = None
            self.llm_model = os.environ["LLM_MODEL"]
            self.llm_description_len_threshold = 10  # in tokens
            self.llm_max_tokens = 300
            self.llm_temperature = 0.1

            # Every n product iteration update DB with product and product embedding data
            self.product_db_insert_interval = 100

            self.llm_client = AsyncOpenAI(
                base_url=os.environ["LLM_API_URL"],
                api_key="EMPTY"  # vLLM does not require a real key
            )

            logger.info("DataIngestService initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize DataIngestService: {e}")
            raise

    async def warmup_services(self):
        logger.info("Running LLM warmup test")

        start_time = time.time()
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system",
                 "content": LLM_SYS_PROMPT_DESCRIPTION},
                {"role": "user", "content": "Solar Guitars 6-string with Fishmann Fluence pickups."}
            ],
            max_tokens=self.llm_max_tokens,
            temperature=self.llm_temperature,
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
        logger.info(generated_output)

        logger.info("Performing test request for embedding service")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.embedding_url,
                json={"input": ["I am testing embedding request."]},
                timeout=10.0
            )

        response.raise_for_status()
        response_data = response.json()
        logger.info(np.array(response_data["data"][0]["embedding"]).shape)
        logger.info(self.embedding_url)

    async def create_retriever_from_corpus(self):
        try:
            all_products_emb_info: List[ProductEmbeddingRead] = await ProductService.get_all_product_emb_info()

            all_products_metadata_corpus = [{
                "id": product.product_id,
                "name": product.name,
                "text": product.generated_description
            } for product in all_products_emb_info]
            all_products_corpus_text = [product.generated_description for product in all_products_emb_info]

            logger.info(f"Training BM25s model on corpus of size: {len(all_products_metadata_corpus)} samples")

            corpus_tokens = bm25s.tokenize(all_products_corpus_text, stopwords="en")

            # Create the BM25 model and index the corpus
            keyword_based_retriever = bm25s.BM25(corpus=all_products_metadata_corpus)
            keyword_based_retriever.index(corpus_tokens)

            keyword_based_retriever.save(f"{BM25_CACHE_DIR}/thomann_product_index_bm25", corpus=all_products_metadata_corpus)

            logger.info(f"Successfully saved BM25s keyword retriever to path: {BM25_CACHE_DIR}/thomann_product_index_bm25")
        except Exception as e:
            logger.error(f"Failed to create bm25s retriever with error: {e}")

    async def llm_generate_description(
            self,
            product_name: str,
            product_description: str,
            main_category: str
    ) -> str:

        user_prompt = LLM_DESCRIPTION_TEMPLATE.format(
            name=product_name,
            category=main_category,
            description=product_description
        )

        start_time = time.time()
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": LLM_SYS_PROMPT_DESCRIPTION},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=self.llm_max_tokens,
            temperature=self.llm_temperature,
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

        # TODO implement hallucination filtering
        ### e.g::  built with factory strings in .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .046 .0

        logger.info(description_out)

        return description_out

    def build_structured_text(self, product: ProductSource) -> str:
        availability = "In Stock" if product.in_stock else "Out of Stock"
        bullet_points_str = "\n".join([f"- {point}" for point in product.description_points])

        structured_text_description = STRUCTURED_TEXT_TEMPLATE.format(
            name=product.name,
            price=product.price,
            availability=availability,
            bullet_points=bullet_points_str
        )
        return structured_text_description

    def get_product_category(self, category_values: List[str]):
        category_str = None

        try:
            filtered = [c for c in category_values if c != "All Categories"]

            # Slice the list to remove the last item (the brand/specific item) This leaves us with ['Software', 'Virtual Instruments + Samplers']
            structural_categories = filtered[0]
            category_str = " ".join(structural_categories).strip()
        except Exception as e:
            logger.error(e)

        return category_str

    async def get_embedding_async(self, client: httpx.AsyncClient, text: str):
        retry_delays = [2.0, 4.0, 6.0]
        total_attempts = 1 + len(retry_delays)  # Initial attempt + 3 retries

        output_embedding = None
        for attempt in range(total_attempts):
            try:
                response = await client.post(
                    self.embedding_url,
                    json={"input": [text]},
                    timeout=10.0
                )
                response.raise_for_status()
                response_data = response.json()

                output_embedding = np.array(response_data["data"][0]["embedding"])

                break
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                # If we have retries left, log a warning and sleep
                if attempt < len(retry_delays):
                    delay = retry_delays[attempt]
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {total_attempts} attempts failed. Final error: {e}")

        return output_embedding

    async def ingest_and_index_async(self, file_path: str):
        try:
            logger.info("Starting async data ingestion process...")

            # Fetch existing URLs that have been processed. Can be important for updating/skipping
            logger.info("Fetching existing product URLs from database...")
            existing_urls = await ProductService.get_all_product_urls()
            logger.info(f"Found {len(existing_urls)} existing products in database.")

            products_list = FileUtils.load_jsonl_file(file_path)

            products_list = [
                ProductSource(
                    url=product["url"],
                    name=product["name"],
                    price=product["price"],
                    in_stock=product["in_stock"],
                    description_points=product["description_points"],
                    image_urls=product["image_urls"],
                    breadcrumbs=product["breadcrumbs"],
                    reviews=[
                        ProductReview(
                            title=review["title"],
                            author=review["author"],
                            date=review["date"],
                            rating=review["rating"],
                            text=review["text"],
                            votes_up=review["votes_up"],
                            votes_down=review["votes_down"]
                        )
                        for review in product["reviews"]
                        if len(review["title"].strip()) > 0 and len(review["text"].strip())
                    ]
                )
                for product in products_list
                if product["url"] not in existing_urls
            ]

            total_product_count = len(products_list)

            logger.info(f"Starting processing of {total_product_count} total products.")

            product_emb_to_insert = []
            products_to_insert = []
            product_reviews_to_insert = []
            async with httpx.AsyncClient() as client:
                for idx, product in tqdm(enumerate(products_list)):
                    product_price_str = product.price.strip().replace(" ", "")
                    product_price = float(product_price_str.replace("€", "").replace(",", "."))
                    description = "\n".join(product.description_points)
                    category_str = self.get_product_category(product.breadcrumbs)

                    if category_str is None or not category_str.strip():
                        logger.warning("Couldn't get category for product. Skipping.")
                        continue

                    structured_text = self.build_structured_text(product)
                    llm_description_text = await self.llm_generate_description(
                        product_name=product.name,
                        product_description=description,
                        main_category=category_str
                    )

                    if len(llm_description_text.split(" ")) <= self.llm_description_len_threshold:
                        logger.warning(f"Skipping short llm_description_text: len={len(llm_description_text.split(" "))}")
                        continue

                    # Run async calls to the same embedding service
                    embedding_task = self.get_embedding_async(client, structured_text)
                    llm_embedding_task = self.get_embedding_async(client, llm_description_text)
                    embedding_vector, llm_embedding_vector = await asyncio.gather(
                        embedding_task,
                        llm_embedding_task
                    )

                    if embedding_vector is None or llm_embedding_vector is None:
                        logger.warning("One of embeddings were None. Skipping indexing for this sample.")
                        continue

                    product_record = ProductCreate(
                        name=product.name,
                        url=product.url,
                        description=description,
                        category_name=category_str,
                        price=product_price,
                        store_name=Store.thomann.value,  # ! currently only one store supported
                        in_stock=product.in_stock,
                    )
                    product_embedding_record = ProductEmbeddingCreate(
                        name=product.name,
                        structured_text=structured_text,
                        generated_description=llm_description_text,
                        structured_embedding=embedding_vector,
                        generated_description_embedding=llm_embedding_vector
                    )

                    reviews = []
                    for review in product.reviews:
                        review_create = ReviewCreate(
                            author=review.author,
                            title=review.title,
                            text=review.text,
                            rating=review.rating,
                            votes_up=review.votes_up,
                            votes_down=review.votes_down
                        )
                        reviews.append(review_create)

                    # Perform data tracking after all requests are made
                    products_to_insert.append(product_record)
                    product_emb_to_insert.append(product_embedding_record)
                    product_reviews_to_insert.append(reviews if len(reviews) > 0 else [])  # keep length mapping

                    # Perform saving every nth item batch
                    if idx % self.product_db_insert_interval == 0:
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

                        await self.create_retriever_from_corpus()

                if len(products_to_insert) > 0:
                    inserted_product_ids = await ProductService.insert_products_with_embeddings_and_reviews(
                        products_create=products_to_insert,
                        embeddings_create=product_emb_to_insert,
                        reviews_create=product_reviews_to_insert
                    )
                    logger.info(f"Inserted {len(inserted_product_ids)} last products into database.")

            await self.create_retriever_from_corpus()

        except Exception as ex:
            raise RuntimeError(f"Encountered error in {self.__class__.__name__} pipeline processing:: {ex}\n Stopping...")


async def main():
    ingest_service = DataIngestService()
    await ingest_service.warmup_services()

    await ingest_service.ingest_and_index_async(file_path=PRODUCTS_JSONL_PATH)


if __name__ == "__main__":
    asyncio.run(main())
