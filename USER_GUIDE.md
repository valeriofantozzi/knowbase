# USER_GUIDE — Technical Guide for KnowBase

This guide explains how to use KnowBase at a technical level: setup, CLI commands, Python API, model management, and details on how embedding collections are organized.

**Requirements**

- Python 3.9+ (recommended 3.11)
- Virtualenv or venv
- (Optional) CUDA / MPS for acceleration

## Environment setup

1. Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Copy the `.env` example and modify main variables:

```
cp .env.example .env
```

- `MODEL_NAME`: default model (e.g., `BAAI/bge-large-en-v1.5`)
- `DEVICE`: `cpu`, `cuda`, or `auto`
- `MODEL_CACHE_DIR`: directory for model cache

## Essential project structure

- `src/preprocessing/` — parsing (SRT), cleaning, chunking
- `src/embeddings/` — model adapters, loading management, and pipeline
- `src/vector_store/` — ChromaDB wrapper, naming for collections
- `scripts/` — CLI utilities for processing and querying

## Main CLI commands

- Process a folder of SRTs (using default model):

```
python scripts/process_subtitles.py --input subtitles/ --output data/processed
```

- Process with a specific model:

```
python scripts/process_subtitles.py --input subtitles/ --model "google/embeddinggemma-300m" --output data/processed
```

- Execute a text query via CLI script:

```
python scripts/query_subtitles.py "how to care for an orchid?"
```

Common options in scripts (if exposed): `--input`, `--output`, `--model`, `--batch-size`, `--device`.

## Programmatic usage (Python API)

Minimal example to get embeddings and save them:

```python
from src.embeddings.pipeline import EmbeddingPipeline
from src.vector_store.chroma_store import ChromaStore

# Initialize pipeline with specific model
pipeline = EmbeddingPipeline(model_name="BAAI/bge-large-en-v1.5", device="auto")

# Generate embeddings for a list of texts
texts = ["Text 1", "Text 2"]
embs = pipeline.embed_batch(texts)

# Save to Chroma (collection name determined by model)
store = ChromaStore(model_name=pipeline.model_name)
store.add_documents(texts, embs, metadatas=[{"source":"demo"}] * len(texts))
```

### Notes on `EmbeddingPipeline`

- Handles tokenization, chunking (if required), and batching.
- Supports model caching to avoid continuous reloading.

## Chroma collection naming

- Convention: `document_embeddings_<model_tag>`
- Examples:
  - BGE: `document_embeddings_bge_large`
  - Gemma: `document_embeddings_gemma_300m`

This prevents conflicts between different dimensions and allows cross-model comparisons without overwriting.

## Practical tips and troubleshooting

- If you see OOM on GPU: try `DEVICE=cpu` or reduce `--batch-size`.
- For quick local tests: use reduced sizes or data subsets.
- Removing collections: use helpers in `src/vector_store/` (warning: destructive operation).

## Advanced examples

- Comparative evaluation: process the same dataset with two different models and compare retrieval scores or qualitatively review results in `streamlit_app.py`.
- Embedding length mismatch: if comparing BGE(1024) with Gemma(768), keep collections separate and perform analysis after normalization or external projection.

## Quick UI deployment

1. Ensure the vector DB (Chroma) is accessible and persistent in `data/vector_db`.
2. Start the UI:

```
./start_viewer.sh
```

## Testing and development

- Run tests with `pytest`:

```
pytest -q
```

- Add tests for new adapters in the `tests/` folder.

## Contacts and resources

To extend support for new models, add an adapter in `src/embeddings/adapters/` and register the model in `src/embeddings/model_registry.py`.

---

If needed, I can add `.env` configuration examples, or generate a Docker/Compose deployment script to run the UI and vector DB in production.
