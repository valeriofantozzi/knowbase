# Subtitle Embedding & Retrieval System

A semantic search system for YouTube subtitle files that enables natural language queries over video content using multiple state-of-the-art embedding models.

## Overview

This system processes YouTube subtitle files (SRT format), generates high-quality embeddings using configurable embedding models, and stores them in a local ChromaDB vector database for fast semantic search. Supports multiple embedding models including BGE-large-en-v1.5 and EmbeddingGemma-300m.

## Features

- **Preprocessing Pipeline**: Parse SRT files, clean text, and create semantic chunks
- **Multi-Model Embedding**: Choose from multiple state-of-the-art embedding models (BGE, EmbeddingGemma, and extensible to others)
- **Dynamic Model Switching**: Switch between models at runtime with intelligent caching
- **Model-Specific Collections**: Automatic isolation of embeddings by model to prevent conflicts
- **Adaptive Dimensions**: Support for different embedding dimensions (1024 for BGE, 768 for EmbeddingGemma with MRL support)
- **Vector Storage**: Store embeddings with rich metadata in ChromaDB
- **Semantic Search**: Query video content using natural language
- **CLI Interface**: Command-line tools for processing and querying with model selection
- **Web Interface**: Streamlit app for interactive search with model selection

## Technology Stack

- **Language**: Python 3.9+
- **Embedding Models**:
  - BAAI/bge-large-en-v1.5 (1024-dim, instruction-tuned)
  - Google/embeddinggemma-300m (768-dim with MRL, structured prompts)
  - Extensible adapter system for additional models
- **Vector Database**: ChromaDB (local, persistent, model-isolated collections)
- **Deep Learning**: PyTorch (with CUDA/MPS support)
- **Model Management**: Intelligent caching and memory management

## Quick Start

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Process subtitle files (default BGE model):
   ```bash
   python scripts/process_subtitles.py --input subtitles/
   ```

2. Process with specific model:
   ```bash
   # Use EmbeddingGemma model
   python scripts/process_subtitles.py --input subtitles/ --model "google/embeddinggemma-300m"

   # Use BGE model explicitly
   python scripts/process_subtitles.py --input subtitles/ --model "BAAI/bge-large-en-v1.5"
   ```

3. Query the indexed content:
   ```bash
   python scripts/query_subtitles.py "your search query"
   ```

4. Launch web interface for interactive search:
   ```bash
   ./start_viewer.sh
   ```

## Project Structure

```
project_root/
├── src/                    # Source code
│   ├── preprocessing/      # SRT parsing, text cleaning, chunking
│   ├── embeddings/         # Multi-model embedding system
│   │   ├── adapters/       # Model-specific adapters (BGE, EmbeddingGemma)
│   │   ├── model_loader.py # Model loading with adapter pattern
│   │   ├── model_registry.py # Centralized model metadata & adapters
│   │   ├── model_manager.py # Multi-model caching & management
│   │   └── pipeline.py     # Embedding generation pipeline
│   ├── vector_store/       # ChromaDB management, model-isolated collections
│   ├── retrieval/          # Query engine, similarity search
│   └── utils/              # Configuration, logging, utilities
├── scripts/                # CLI scripts (with --model support)
├── data/                   # Data directories
│   ├── raw/               # Original subtitle files
│   ├── processed/         # Processed chunks
│   └── vector_db/        # ChromaDB storage (model-specific collections)
├── docs/                   # Documentation
│   └── model_selection_guide.md # Detailed model comparison guide
├── tests/                 # Test suite (unit & integration)
└── streamlit_app.py       # Web interface with model selection
```

## Model Selection

This system supports multiple embedding models with different characteristics:

### Supported Models

| Model | Dimensions | Max Length | Precision | Use Case |
|-------|------------|------------|-----------|----------|
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | float32, float16 | General purpose, high quality |
| `google/embeddinggemma-300m` | 768 (MRL: 512, 256, 128) | 2048 | float32, bfloat16 | Long contexts, flexible dimensions |

### Model Configuration

#### Environment Variable
Set the default model in your `.env` file:
```bash
MODEL_NAME=BAAI/bge-large-en-v1.5
# or
MODEL_NAME=google/embeddinggemma-300m
```

#### CLI Usage
Override the default model for specific operations:
```bash
# Processing with BGE
python scripts/process_subtitles.py --input subtitles/ --model "BAAI/bge-large-en-v1.5"

# Processing with EmbeddingGemma
python scripts/process_subtitles.py --input subtitles/ --model "google/embeddinggemma-300m"
```

#### Programmatic Usage
```python
from src.embeddings.pipeline import EmbeddingPipeline

# Use BGE model
bge_pipeline = EmbeddingPipeline(model_name="BAAI/bge-large-en-v1.5")

# Use EmbeddingGemma model
gemma_pipeline = EmbeddingPipeline(model_name="google/embeddinggemma-300m")
```

### Collection Management

Each model stores embeddings in separate ChromaDB collections to prevent conflicts:

- **BGE collections**: `subtitle_embeddings_bge_large`
- **EmbeddingGemma collections**: `subtitle_embeddings_gemma_300m`

Collections are automatically created and managed based on the selected model.

### Model Switching

The system supports dynamic model switching with intelligent caching:

- Models are cached in memory to avoid reloading
- Automatic cleanup of unused models to manage memory
- Seamless switching between models in the web interface

## Configuration

Copy `.env.example` to `.env` and configure as needed:

```bash
cp .env.example .env
```

Available configuration options:
- `MODEL_NAME`: Default embedding model
- `DEVICE`: Computation device (cpu/cuda/auto)
- `MODEL_CACHE_DIR`: Cache directory for downloaded models

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

