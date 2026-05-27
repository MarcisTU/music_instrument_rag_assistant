import asyncio
import json
import os
import time
from typing import List, Tuple

import bm25s
import httpx
import numpy as np
from loguru import logger
from openai import AsyncOpenAI

from src.db.service import ProductService
from src.db.models import ProductEmbedding
from src.models.prompts import LLM_PRODUCT_SUGGESTION_TEMPLATE, AiProductResults, LLM_PRODUCT_ITEM_TEMPLATE, \
    LLM_USER_QUERY_TRANSFORM_TEMPLATE, AiUserQuery, LLM_SYS_PROMPT_USER_QUERY_NORMALIZATION, LLM_SYS_NSHOT_PROMPT
from src.models.schemas import ProductRead, RAGDocument
from src.modules.constants import BM25_CACHE_DIR
from src.modules.reciprocal_rank_fusion import reciprocal_rank_fusion


class LLMService:
    def __init__(self, use_reranker: bool = False):
        try:
            self.llm_url = os.environ["LLM_API_URL"]
            self.embedding_url = os.environ["EMBEDDING_API_URL"]
            self.reranker_url = os.environ["RERANKER_API_URL"]
            self.llm_model_name = os.environ["LLM_MODEL"]

            self.llm_model = os.environ["LLM_MODEL"]
            self.llm_client = AsyncOpenAI(
                base_url=os.environ["LLM_API_URL"],
                api_key="EMPTY"  # no key for locally/remote deployed vLLM
            )

            self.use_reranker = use_reranker
            self.top_k_vectors = 50   # initial similar vector count from vector DB
            self.top_n_rerank = 25    # how many items to take after reranking step

            self.keyword_based_retriever = bm25s.BM25.load(BM25_CACHE_DIR, load_corpus=True)

        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {e}")
            raise

    async def warmup(self):
        logger.info("Performing test request for LLM service")

        response = await self.llm_normalize_user_query(
            user_query="I’m looking for a versatile MIDI keyboard controller for music production and film scoring, preferably with 49 or 61 semi-weighted keys, velocity sensitivity, aftertouch, and assignable pads/knobs for DAW control."
        )

        logger.info(response)

        logger.info("Performing test request for embedding service")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.embedding_url,
                json={"input": ["I am testing embedding request."]},
                timeout=10.0
            )

        response.raise_for_status()
        response_data = response.json()
        logger.debug(np.array(response_data["data"][0]["embedding"]).shape)
        logger.debug(self.embedding_url)

        logger.info(f"Warmup done: {response}")

    async def get_query_embedding_async(self, client: httpx.AsyncClient, text: str) -> list[float]:
        try:
            response = await client.post(
                self.embedding_url,
                json={"input": [text]},
                timeout=10.0
            )
            response.raise_for_status()
            response_data = response.json()

            return response_data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise

    async def rerank_documents(
        self,
        client: httpx.AsyncClient,
        query: str,
        documents: List[RAGDocument],
        top_n: int = 25
    ) -> List[RAGDocument]:
        reranked_docs = []

        logger.info(f"Reranking {len(documents)} documents down to top {top_n}...")
        try:
            payload = {
                "query": query,
                "documents": [doc.text for doc in documents]
            }

            response = await client.post(self.reranker_url, json=payload, timeout=15.0)
            response.raise_for_status()
            rerank_data = response.json()

            # Sorting based on response scores
            sorted_results = sorted(rerank_data["results"], key=lambda x: x["relevance_score"], reverse=True)[:top_n]

            reranked_docs = [documents[item["index"]] for item in sorted_results]
        except Exception as e:
            logger.error(f"Reranker request failed: {e}. Falling back to initial vector order.")

        return reranked_docs

    async def _run_llm_request(self, user_prompt: str, system_prompt: str, schema, max_tokens: int = 400):
        start_time = time.time()
        response = await self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "description",
                    "schema": schema.model_json_schema()
                },
            }
        )
        end_time = time.time()
        logger.info(f"Elapsed time: {end_time - start_time:4f} seconds")

        return response.choices[0].message.content

    async def llm_normalize_user_query(
        self,
        user_query: str,
    ) -> str:
        user_prompt = LLM_USER_QUERY_TRANSFORM_TEMPLATE.format(
            text=user_query
        )

        try:
            generated_output = await self._run_llm_request(
                user_prompt=user_prompt,
                system_prompt=LLM_SYS_PROMPT_USER_QUERY_NORMALIZATION,
                schema=AiUserQuery,
                max_tokens=400
            )

            generated_output_json = json.loads(generated_output)
            ai_user_query = AiUserQuery(**generated_output_json)

            return ai_user_query.product_description
        except Exception as e:
            logger.error(f"LLM product description generation failed: {e}")
            raise

    async def llm_generate_result(
        self,
        user_query: str,
        context: str
    ) -> AiProductResults:
        user_prompt = LLM_PRODUCT_SUGGESTION_TEMPLATE.format(
            context=context,
            user_query=user_query
        )

        generated_output = await self._run_llm_request(
            user_prompt=user_prompt,
            system_prompt=LLM_SYS_NSHOT_PROMPT,
            schema=AiProductResults,
            max_tokens=256
        )
        generated_output_json = json.loads(generated_output)
        ai_product_results = AiProductResults(**generated_output_json)

        return ai_product_results

    def format_product_recommendations(self, products: List[ProductRead]) -> str:
        if not products:
            output_msg = "I couldn't find any products matching your request right now."
        else:
            output = ["### 🎹 Here are some products you might be interested in:\n"]

            for product in products:
                stock_status = "🟢 In Stock" if product.in_stock else "🔴 Out of Stock"
                clean_desc = "\n".join(f"  * {line.strip()}" for line in product.description.split("\n") if line.strip())

                product_block = f"""
#### [{product.name}]({product.url})
Price: {product.price:,.2f}€ | Status: {stock_status}
Key Features:
{clean_desc}
 ---
"""

                output.append(product_block)

            output_msg = "\n".join(output)

        return output_msg

    async def vector_search(
        self,
        user_query: str,
        query_vector: list[float]
    ) -> Tuple[list[RAGDocument], list[RAGDocument]]:
        logger.info("Running vector and bm25 document search")
        # Perform Vector based semantic search
        retrieved_product_docs: List[ProductEmbedding] = await ProductService.get_similarity_search_query(
            embedding_vector=query_vector, limit=self.top_k_vectors)
        retrieved_vector_rag_docs = [RAGDocument(
            product_id=product_doc.product_id,
            name=product_doc.name,
            text=product_doc.generated_description
        ) for product_doc in retrieved_product_docs]

        # Perform BM25s keyword based search
        query_tokens = bm25s.tokenize(user_query)
        retrieved_products, scores = self.keyword_based_retriever.retrieve(query_tokens, k=self.top_k_vectors)
        retrieved_products, scores = retrieved_products.squeeze(), scores.squeeze()  # Reduce results from (1, n) to (n,)
        retrieved_bm25_rag_docs = [RAGDocument(
            product_id=p_result["id"],
            name=p_result["name"],
            text=p_result["text"]
        ) for p_result in retrieved_products]

        return retrieved_vector_rag_docs, retrieved_bm25_rag_docs

    async def inference(
        self,
        user_query: str
    ) -> str:
        logger.info(f"Received user query: '{user_query}'")

        async with httpx.AsyncClient() as client:
            logger.info("Running user query normalization")
            ai_user_query = await self.llm_normalize_user_query(user_query=user_query)

            query_vector = await self.get_query_embedding_async(client, ai_user_query)

            retrieved_vector_docs, retrieved_bm25_docs = await self.vector_search(
                user_query=user_query,
                query_vector=query_vector
            )

            if not retrieved_vector_docs and not retrieved_bm25_docs:
                logger.warning("No context documents matched your query vector.")
                context = "No specific product context found matching your query."
            else:
                logger.info(f"Retrieved {len(retrieved_vector_docs + retrieved_bm25_docs)} documents.")

                retrieved_hybrid_ranked_docs = reciprocal_rank_fusion(
                    bm25_results=retrieved_bm25_docs, vector_results=retrieved_vector_docs
                )

                if self.use_reranker:
                    final_ranked_docs = await self.rerank_documents(client, ai_user_query, retrieved_hybrid_ranked_docs, top_n=self.top_n_rerank)
                else:
                    final_ranked_docs = retrieved_hybrid_ranked_docs[:self.top_n_rerank]

                context = "".join([
                    f"{LLM_PRODUCT_ITEM_TEMPLATE.format(id=doc.product_id, name=doc.name, description=doc.text)}\n"
                    for doc in final_ranked_docs
                ])

            logger.info("Generating product suggestions")
            result_products = await self.llm_generate_result(context=context, user_query=ai_user_query)

            suggested_products = await ProductService.get_products(result_products.product_ids)
            result_text = self.format_product_recommendations(products=suggested_products)

            return result_text


async def main():
    llm_service = LLMService(use_reranker=False)
    await llm_service.warmup()

    logger.info("LLMService initialized successfully.")

    # query = "I’m looking for a matched pair of condenser microphones specifically for drum overhead recording in a studio setup. Preferably small-diaphragm condensers with a detailed high-end response, low self-noise, and good stereo imaging for capturing cymbals and room ambience in rock and fusion mixes."
    query = "I’m looking for a modern electric guitar for progressive metal and hard rock, with a roasted maple neck, stainless steel frets, and active humbuckers like Fishman Fluence. I’d like models similar with floyd rose locking tremolo."
    # query = "I’m looking for a versatile MIDI keyboard controller for music production and film scoring, preferably with 49 or 61 semi-weighted keys, velocity sensitivity, aftertouch, and assignable pads/knobs for DAW control."
    response = await llm_service.inference(user_query=query)

    logger.info(response)


if __name__ == "__main__":
    asyncio.run(main())
