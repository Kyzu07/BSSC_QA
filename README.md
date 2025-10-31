# BSSC-QA Framework

BSSC-QA is a LangChain/LangGraph based framework for building high-quality question–answer (QA) datasets from domain documents. It combines automated chunking, retrieval-augmented answer synthesis, and multi-dimensional evaluation into a single reproducible pipeline that can be tuned for Bloom-level coverage or custom scoring strategies.

## Key Features
- Multi-agent orchestration with dedicated generator, synthesis, and evaluator agents.
- Pluggable LLM provider configuration (Gemini, DeepSeek, Mistral, Hugging Face) resolved through environment variables.
- Document ingestion helpers for TXT, PDF, HTML, DOCX, and URLs with robust cleaning and chunking.
- ChromaDB vector store management with an offline-friendly hashing embedding fallback.
- Rule-based and LLM-based QA validation to track quality metrics and export clean datasets.
- Notebook workflow (`bssc_qa/notebooks/01_BSSC_QA_Complete.ipynb`) for exploratory runs and experimentation.

## Repository Layout
```text
.
├── run_pipeline.py              # CLI entry point for the full QA pipeline
├── config.json                  # Central configuration for LLMs, agents, vector store, export
├── requirements.txt             # Captured package versions for the working environment
├── bssc_qa/
│   ├── src/
│   │   ├── agents/              # Generator, synthesis, evaluator agents
│   │   ├── core/                # Config loader, LLM factory, embeddings, vector store manager
│   │   ├── pipeline/            # Chunking, document loaders, ingestion, orchestrator
│   │   └── utils/               # Text normalization helpers and shared data structures
│   └── notebooks/               # End-to-end notebook walkthrough and version checkpoints
├── data/
│   ├── chroma_db/               # Persisted Chroma vector store (created after ingestion)
│   ├── output/                  # QA dataset exports
│   └── travel_scraped/          # Sample source corpus used during development
└── BSSC_QA_blueprint.txt        # Design notes and planning reference
```

## Prerequisites
- Python 3.12 (the project has been developed and tested on Python 3.12.9).
- A virtual environment (recommended) with access to the packages listed in `requirements.txt`.
- API keys for at least one supported LLM provider (Gemini, DeepSeek, Mistral, or Hugging Face Inference Endpoints).

## Installation
```bash
git clone <your-fork-or-repo-url>.git
cd BSSC-QA
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `requirements.txt` mirrors the full development environment. If you want a leaner deployment, create a trimmed dependency set once you confirm which packages the project actually needs for your use case.

## Environment Configuration
Set the API keys for the providers you plan to use before running the pipeline. The configuration loader expects the following environment variables:

```bash
export BSSC_QA_GEMINI_API_KEY="your-google-gemini-key"
export BSSC_QA_DEEPSEEK_API_KEY="your-deepseek-key"
export BSSC_QA_MISTRAL_API_KEY="your-mistral-key"
export BSSC_QA_HUGGINGFACE_API_KEY="your-huggingface-token"
```

If an API key is unavailable, either switch the corresponding agent to another provider in `config.json` or supply an offline-compatible embedding model (`offline-hash`) in the vector store settings.

## Preparing Data
Populate the ChromaDB vector store with your source documents before generating QA pairs.

```python
from pathlib import Path
from core.config import load_config
from core.vector_store import VectorStoreManager
from pipeline.ingestion import IngestionPipeline

cfg = load_config(Path("config.json"))
vs_manager = VectorStoreManager(
    persist_directory="data/chroma_db",
    collection_name=cfg.vector_store.collection_name,
    embedding_model=cfg.vector_store.embedding_model,
    embedding_dimension=cfg.vector_store.embedding_dimension,
)

ingestor = IngestionPipeline(
    vector_store_manager=vs_manager,
    chunk_size=cfg.chunking.chunk_size,
    chunk_overlap=cfg.chunking.chunk_overlap,
)

results = ingestor.ingest_directory(
    directory=Path("data/travel_scraped"),
    pattern="*.txt"
)
print(results)
```

You can ingest individual files or URLs via `ingestor.ingest_document(<path-or-url>)`. The process cleans, normalizes, and chunks content before adding it to ChromaDB.

## Running the Pipeline
Once documents have been ingested and API keys are configured, run the orchestrated multi-agent pipeline:

```bash
python run_pipeline.py
```

The script will:
1. Load `config.json`.
2. Initialize the vector store, tools, and agents.
3. Sample chunks from the vector store.
4. Generate questions, synthesize answers, and evaluate quality.
5. Export validated QA pairs to `data/output/qa_dataset_<timestamp>.json`.

Console output lists intermediate progress and summary statistics (QA counts, pass rate, average score).

## Working in Notebooks
The notebook `bssc_qa/notebooks/01_BSSC_QA_Complete.ipynb` mirrors the full workflow with rich explanations. Launch it inside the activated environment:

```bash
jupyter notebook bssc_qa/notebooks/01_BSSC_QA_Complete.ipynb
```

Use the notebook to experiment with different chunk sizes, agent prompts, or evaluation thresholds before committing changes to the Python modules.

## Configuration Tips
- Adjust chunking behavior under `chunking` in `config.json` to fine-tune granularity and overlap.
- Swap LLM providers per agent (generator, synthesis, evaluator) by editing the `agents` map.
- Enable Bloom taxonomy coverage or human review in the config to extend scoring dimensions.
- To reinitialize the vector store from scratch, call `VectorStoreManager.reset_collection()` before ingesting new documents.

## Troubleshooting
- **Empty or low-quality QA output:** Ensure the ChromaDB collection contains relevant documents and verify that your API keys are valid.
- **Provider authentication errors:** Double-check environment variables and provider-specific rate limits.
- **Large dependency footprint:** Start from `requirements.txt`, then incrementally remove packages not required in your environment to slim the install.

## Next Steps
- Write integration tests around the orchestrator to guard against regressions.
- Introduce CLI arguments to `run_pipeline.py` for chunk counts, provider overrides, or output locations.
- Containerize the pipeline (Docker) for reproducible deployments across environments.
