# Advanced RAG Product Recommendation Engine

A Retrieval-Augmented Generation (RAG) system built to recommend musical instruments and products based on a user's query that contains description of what user has used/bought and what he is looking for.

The platform leverages an asynchronous microservices architecture powered by **vLLM** for fast local LLM and embedding inference, **RabbitMQ** for reliable message queuing and event-driven architecture, **pgvector** for semantic vector searches, and **uv** for ultra-fast, deterministic Python dependency resolution.

---

## 🏗️ Architecture & How It Works

The application is structured as a collection of decoupled services communicating asynchronously.

### Business Logic Flow
1. **User Query Intake**: The client submits a query describing a user's scenario (e.g., historical interactions, financial goals/wishes, or profile descriptions).
2. **Task Enqueuing**: The `api` worker validates the request and publishes a task to **RabbitMQ**, returning an immediate tracking token to the client.
3. **Background Processing**: The `llm_service` background consumer picks up the task and executes the core RAG pipeline:
   - **Vectorization**: Transforms the user's query into dense embeddings using the `vllm_emb` engine.
   - **Vector Retrieval**: Performs a cosine similarity search against metadata and index points stored inside the **Postgres (pgvector)** database to fetch matching products/instruments.
   - **Reranking**: Filters and optimizes relevance scores of candidate elements via the cross-encoder (`vllm_reranker`).
   - **Context Construction & Generation**: Assembles top-ranked context artifacts alongside historical notes, passing them into the structural LLM generation cluster (`vllm_llm`) to produce localized, tailored recommendations.

---

## 🛠️ Prerequisites

Before launching the pipeline, ensure your system meets the following infrastructure configurations:

- **Docker & Docker Compose**
- **NVIDIA Container Toolkit**: Required to pass GPU resources into the `vllm` containers. Ensure your host has functional NVIDIA drivers and standard `nvidia-smi` access.
- **Hardware Allocations**:
  - VRAM is statically budgeted via `--gpu-memory-utilization` split among the LLM (40%), Reranker (25%), and Embedding (15%) modules. Currently tested on local PC GPU (max 12GB GPU)

---

## ⚙️ Environment Configuration

The containers use defined configurations through a root `.env` file. Copy the template below, name it `.env`, and configure it appropriately before spinning up the infrastructure.
Use .env.example as reference of how locally deployed system config would look.