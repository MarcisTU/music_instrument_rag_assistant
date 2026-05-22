# Advanced RAG Product Recommendation Engine

A Retrieval-Augmented Generation (RAG) system built to recommend musical instruments and products based on a user's query that contains description of what user has used/bought and what he is looking for.

The platform leverages an asynchronous microservices architecture powered by **vLLM** for fast local LLM and embedding inference, **RabbitMQ** for reliable message queuing and event-driven architecture, **pgvector** for semantic vector searches, and **uv** for ultra-fast, deterministic Python dependency resolution.

---

## 🏗️ Architecture & How It Works

The application is structured as a collection of decoupled services communicating asynchronously in event-driven way::

### Logic Flow
1. **User Query Intake**: The client submits a query describing a user's scenario (e.g., historical interactions, financial goals/wishes, or profile descriptions).
2. **Task Enqueuing**: The `api` worker validates the request and publishes a task to **RabbitMQ**, returning an immediate tracking token to the client.
3. **Background Processing**: The `llm_service` background consumer picks up the task and executes the core RAG pipeline:
   - **Vectorization**: Transforms the user's query into dense embeddings using the `vllm_emb` engine.
   - **Vector Retrieval**: Performs a cosine similarity search against metadata and index points stored inside the **Postgres (pgvector)** database to fetch matching products/instruments.
   - **Reranking**: Filters and optimizes relevance scores of candidate elements via the cross-encoder (`vllm_reranker`).
   - **Context Construction & Generation**: Assembles top-ranked context artifacts alongside historical notes, passing them into the structural LLM generation cluster (`vllm_llm`) to produce localized, tailored recommendations.

---

## 🛠️ Prerequisites

Before launching the pipeline, ensure your system meets the following configuration requirements:

- **Docker & Docker Compose**
- **NVIDIA Container Toolkit**: Required to pass GPU resources into the `vllm` containers. Ensure your host has functional NVIDIA drivers and standard `nvidia-smi` access.
- **Hardware Allocations**:
  - GPU VRAM is statically budgeted via `--gpu-memory-utilization` split among the vLLM model services. Currently tested on local PC GPU (Nvidia RTX3060 - max 12GB GPU)

---

## ⚙️ Environment Configuration

The containers use defined configurations through a root `.env` file. Copy the template from `.env.example`, name it `.env`, and configure it appropriately before spinning up the infrastructure.

---

## 🚀 Docker Startup

### Data Ingest step
1) Build the Dockerfile
```bash
docker compose build
```

2) Run api docker service (since migration files exist no need to rerun `docker compose --profile tools run --rm alembic_gen uv run alembic revision --autogenerate -m "initial migration"`)
```bash
docker compose up -d api  
```

3) Then after succesfull api service setup we run ./src/services/data_ingest_service.py to create embeddings and ingest them into the system db:
```bash
docker compose run --rm api uv run python -m src.services.data_ingest_service
```


### Main AI services system

1) Start rest of the services in order (first api to run migrations and postgres/rabbitmq service)
```bash
docker compose up -d vllm_llm
docker compose up -d vllm_emb
docker compose up -d vllm_reranker
docker compose up -d llm_worker
docker compose up -d broker_worker
docker compose up -d callback_worker
```

2) Test the API with scripts under ./tests/ directory.

---

## 🚀 Data Pipeline Result Examples

1) Electric guitar suggestion::

*User query*: ``I’m looking for a modern super-strat electric guitar for progressive metal and hard rock, preferably with a roasted maple neck, stainless steel frets, and active humbuckers like Fishman Fluence or EMGs. My budget is around €1,500–€2,000, and I’d like models similar to the Ibanez Prestige, ESP LTD Deluxe, or Schecter SLS series with a fixed bridge or locking tremolo``

*System output (top-5 suggestions)*:
```text
#### [ESP LTD TE-1000 Silver Blast](https://www.thomann.de/intl/esp_ltd_te_1000_silver_blast.htm?type=category)
Price: 1.70€ | Status: 🟢 In Stock

Key Features:
  * Body: Swamp ash
  * Bolt-on neck: Roasted maple
  * Fingerboard: Ebony
  * Neck profile: Thin U
  * Fingerboard radius: 350 mm
  * Scale: 648 mm (25.5")
  * Frets: 24 XJ stainless steel
  * Nut width: 42 mm
  * Moulded nut
  * Tonabnehmer: Seymour Duncan APH-1N (neck) and Seymour Duncan Custom 14 (bridge) Humbucker
  * Volume control with push/pull function
  * Tone control with push/pull function
  * 3-Way switch
  * Tailpiece/bridge: Hipshot with string guide through the body
  * Machine heads: LTD locking
  * Black hardware
  * Strings: D'Addario XL110
  * Colour: Silver Blast
---

#### [ESP LTD Viper-1001 ET CHMS](https://www.thomann.de/intl/esp_ltd_viper_1001_et_chms.htm?type=category)
Price: 1.80€ | Status: 🟢 In Stock

Key Features:
  * Body: Mahogany
  * Top: Maple
  * 3-Piece neck-thru-body: Mahogany
  * Fingerboard: Macassar ebony
  * Neck profile: Thin "U"
  * Fingerboard radius: 350 mm (13.78")
  * Scale: 629 mm (24.75")
  * Nut width: 42 mm (1.65")
  * 22 XJ stainless steel frets
  * Pickup: 1 EMG 81 Black Humbucker (bridge)
  * Master volume control
  * EverTune bridge
  * LTD locking machine heads
  * Black hardware
  * Strings: D'Addario XL120 (.009/.011/.016/.024/.032/.042)
  * Colour: Charcoal Metallic matt
---

#### [ESP LTD MSV-1 BLK Mike Schleibaum](https://www.thomann.de/intl/esp_ltd_msv_1_blk_mike_schleibaum.htm?type=category)
Price: 6.80€ | Status: 🟢 In Stock

Key Features:
  * Mike Schleibaum (Darkest Hour) signature model
  * Body: Mahogany
  * Top: Maple
  * 3-Piece neck-thru-body: Mahogany
  * Fingerboard: Ebony
  * Neck profile: Thin U
  * Fingerboard radius: 350 mm
  * Scale: 628 mm (24.75")
  * Nut width: 42 mm
  * Locking nut
  * 24 XJ stainless steel frets
  * Pickup: 1 EMG JH Brushed Gold Humbucker (bridge)
  * Master volume control
  * Floyd Rose 1000 Gold Tremolo
  * Grover machine heads
  * Golden hardware
  * Strings: DAddario XL120
  * Colour: Black
---

#### [ESP LTD M-1000 CARS](https://www.thomann.de/intl/esp_ltd_m_1000_cars.htm?type=category)
Price: 1.70€ | Status: 🟢 In Stock

Key Features:
  * Body: Alder
  * 3-Piece neck-through-body neck: Maple
  * Fingerboard: Ebony
  * Neck profile: Extra Thin U
  * Fretboard radius: 300 - 400 mm compound (11.8 - 15.7")
  * Scale: 648 mm (25.5")
  * Nut width: 43 mm (1.7")
  * Locking nut
  * 24 XJ Stainless steel frets
  * Pickups: Fishman Fluence Modern Humbucker Alnico Black (neck) and Fishman Fluence Modern Humbucker Ceramic Black (bridge)
  * Volume control
  * Tone control with push/pull function
  * 3-Way switch
  * Floyd Rose 1000 SE tremolo
  * Grover tuners
  * Black hardware
  * Original strings: D'Addarío XL120
  * Colour: Candy Apple Red Satin
---

#### [Spector Euro 5 CST Spalted Maple Ltd](https://www.thomann.de/intl/spector_euro_5_cst_spalted_maple_ltd.htm?type=category)
Price: 3.50€ | Status: 🟢 In Stock

Key Features:
  * Top: Spalted maple
  * Body: European ash with a walnut stripe
  * 3-piece neck-thru-body: Maple
  * Fingerboard: Ebony
  * Scale length: 889 mm (35")
  * Nut width: 46 mm (1.81")
  * Brass nut
  * 24 frets
  * Pickups: Active EMG X P split coil (middle) and J single coil (bridge)
  * Spector Legacy preamp by Darkglass (Bass: ±12 dB @ 60 Hz, Treble: ± 12 dB @ 1 kHz)
  * Controls: Volume, Mix, Treble +/-, Bass +/-
  * Aluminium locking bridge with brass saddles
  * String spacing: 17 mm (0.67")
  * Gotoh GB-350 tuners
  * Chrome hardware
  * Colour: Natural high-gloss
  * Includes Spector gig bag
```