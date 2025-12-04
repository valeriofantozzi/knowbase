# KnowBase Architecture

Complete system architecture documentation for KnowBase CLI and core components.

**Table of Contents:**

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Module Organization](#module-organization)
4. [Command Execution Flow](#command-execution-flow)
5. [Data Flow](#data-flow)
6. [Configuration System](#configuration-system)
7. [Integration Points](#integration-points)
8. [Design Patterns](#design-patterns)

---

## Overview

KnowBase is a semantic search and knowledge extraction system built on:

- **Vector Embeddings**: BAAI/bge-large-en-v1.5 (1024-dim) for semantic understanding
- **Vector Database**: ChromaDB for fast retrieval
- **Clustering**: HDBSCAN + UMAP for pattern discovery
- **RAG**: Retrieval-Augmented Generation with LLM integration
- **CLI**: Click framework for user interface
- **Configuration**: Pydantic models for validation and type safety

**Core Capabilities:**

- Load and index documents with semantic embeddings
- Search by semantic similarity (not keyword matching)
- Cluster documents into thematic groups
- Ask questions using RAG (context-aware LLM responses)
- Export data for external analysis
- Reindex with different embedding models

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Layer                               │
│  (Click Commands: load, search, ask, cluster, export, ...)  │
└──────────┬──────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Pipelines                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │Preprocessing │→ │  Embeddings  │→ │ Vector Store │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└────────────────┬────────────────────────┬────────────────────┘
                 │                        │
                 ▼                        ▼
        ┌─────────────────┐      ┌─────────────────┐
        │   AI Search     │      │  Clustering &   │
        │   (RAG + LLM)   │      │   Analytics     │
        └─────────────────┘      └─────────────────┘
                 │                        │
                 └────────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  ChromaDB Vector DB │
                    │  (Persistence)      │
                    └─────────────────────┘
```

### Layered Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Presentation Layer                    │
│           (CLI Commands via Click Framework)            │
├────────────────────────────────────────────────────────┤
│                    Business Logic Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Load Cmd   │  │  Search Cmd  │  │   Ask Cmd    │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤  │
│  │  Cluster Cmd │  │  Export Cmd  │  │ Reindex Cmd  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├────────────────────────────────────────────────────────┤
│                    Service Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │Preprocessing │  │  Embeddings  │  │Vector Store  │  │
│  │  Pipeline    │  │  Pipeline    │  │ Pipeline     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├────────────────────────────────────────────────────────┤
│                   Integration Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ChromaDB API │  │  Transformers│  │  LLM APIs    │  │
│  │ (Vector DB)  │  │ (Embeddings) │  │ (OpenAI,etc) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├────────────────────────────────────────────────────────┤
│                   Data Persistence Layer                │
│    (ChromaDB Collections + Local File System)           │
└────────────────────────────────────────────────────────┘
```

---

## Module Organization

### Directory Structure

```
knowbase/
├── src/
│   ├── cli/                          # CLI Layer
│   │   ├── main.py                   # Entry point, command registration
│   │   ├── config.py                 # CLI configuration, Config singleton
│   │   ├── commands/                 # Command implementations
│   │   │   ├── hello.py              # Test command
│   │   │   ├── load.py               # Load and index documents
│   │   │   ├── search.py             # Semantic search
│   │   │   ├── ask.py                # RAG with LLM
│   │   │   ├── cluster.py            # Clustering analysis
│   │   │   ├── export.py             # Data export
│   │   │   ├── reindex.py            # Model migration
│   │   │   └── info.py               # System info
│   │   └── __init__.py
│   │
│   ├── preprocessing/                 # Preprocessing Pipeline
│   │   ├── __init__.py
│   │   └── preprocessing.py           # Text cleaning, chunking
│   │
│   ├── embeddings/                    # Embedding Pipeline
│   │   ├── __init__.py
│   │   └── embeddings.py              # Model loading, embedding generation
│   │
│   ├── vector_store/                  # Vector Store Pipeline
│   │   ├── __init__.py
│   │   └── vector_store.py            # ChromaDB integration
│   │
│   ├── clustering/                    # Clustering Analysis
│   │   ├── __init__.py
│   │   └── clustering.py              # HDBSCAN + UMAP
│   │
│   ├── ai_search/                     # AI Search & RAG
│   │   ├── __init__.py
│   │   ├── llm_factory.py             # Multi-provider LLM integration
│   │   ├── thinking_displayer.py      # Extended thinking visualization
│   │   └── rag_search.py              # RAG pipeline
│   │
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   └── logger.py                  # Logging configuration
│   │
│   └── __init__.py
│
├── tests/                            # Test Suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration
│   ├── test_cli_commands.py           # CLI unit tests
│   ├── test_clustering.py             # Clustering tests
│   ├── test_embeddings.py             # Embedding tests
│   └── ...
│
├── docs/                             # Documentation
│   ├── CLI_GUIDE.md                  # CLI reference (1,200+ lines)
│   ├── INSTALLATION.md               # Installation guide
│   ├── ARCHITECTURE.md               # This file
│   ├── model_selection_guide.md       # Embedding model selection
│   ├── MULTI_AGENT_CONFIG.md         # Multi-agent setup
│   └── plans/                        # Implementation plans
│
├── data/                             # Data Directory
│   ├── vector_db/                    # ChromaDB storage
│   └── cache/                        # Model cache
│
├── logs/                             # Log files
│   └── app.log                       # Application logs
│
├── streamlit_app.py                  # Web UI (legacy)
├── pyproject.toml                    # Package configuration
├── requirements.txt                  # Dependencies
├── README.md                          # Project overview
└── .env.example                      # Example environment variables
```

### Key Modules

#### CLI Module (`src/cli/`)

**Responsibility**: User interface and command routing

**Key Files**:

- `main.py`: Entry point, command registration, global options
- `config.py`: Configuration management (Config singleton)
- `commands/*.py`: Individual command implementations

**Pattern**: Click decorators + Pydantic models for validation

**Example**:

```python
# src/cli/commands/search.py
@click.command()
@click.option('--query', required=True)
@click.option('--top-k', default=5)
def search(query: str, top_k: int):
    config = Config()
    # Use config for database path, device, etc.
    results = ai_search.search(query, top_k)
    # Display results
```

#### Preprocessing Pipeline (`src/preprocessing/`)

**Responsibility**: Document preparation

**Key Operations**:

1. Read documents from various formats (txt, pdf, md)
2. Clean text (remove special chars, normalize whitespace)
3. Split into chunks (configurable chunk size, overlap)
4. Deduplicate chunks

**Interface**:

```python
pipeline = PreprocessingPipeline()
chunks = pipeline.process(documents, chunk_size=256, overlap=50)
```

#### Embedding Pipeline (`src/embeddings/`)

**Responsibility**: Generate semantic vectors

**Key Operations**:

1. Load embedding model (BAAI/bge-large-en-v1.5 by default)
2. Move to device (auto, cuda, mps, cpu)
3. Batch encode text to vectors
4. Handle device memory efficiently

**Interface**:

```python
pipeline = EmbeddingPipeline(model_name="BAAI/bge-large-en-v1.5")
embeddings = pipeline.embed(chunks, batch_size=32)  # Returns list[list[float]]
```

#### Vector Store Pipeline (`src/vector_store/`)

**Responsibility**: Persistence and retrieval

**Key Operations**:

1. Create/open ChromaDB collection
2. Add embeddings with metadata
3. Search by similarity
4. Delete/update embeddings

**Interface**:

```python
pipeline = VectorStorePipeline(collection_name="subtitle_embeddings_bge_large")
pipeline.add(documents, embeddings, metadatas)
results = pipeline.search(query_embedding, top_k=5)
```

#### AI Search Module (`src/ai_search/`)

**Responsibility**: RAG and LLM integration

**Components**:

- `llm_factory.py`: Multi-provider LLM support (OpenAI, Anthropic, Groq, Azure, Ollama)
- `thinking_displayer.py`: Extended thinking visualization with streaming
- `rag_search.py`: Retrieval-Augmented Generation pipeline

**Interface**:

```python
# Search and retrieve context
results = ai_search.search(query, top_k=5)

# Generate answer with LLM
llm = LLMFactory.create(provider="openai", model="gpt-4")
answer = llm.generate(prompt=f"Context: {results}\n\nQuestion: {query}")
```

#### Clustering Module (`src/clustering/`)

**Responsibility**: Pattern discovery and analysis

**Key Operations**:

1. Use HDBSCAN for density-based clustering
2. Apply UMAP for dimensionality reduction
3. Calculate cluster statistics
4. Generate visualization data

**Interface**:

```python
clusters = clustering.cluster(embeddings, min_cluster_size=5)
# Returns: cluster_labels, cluster_sizes, noise_points
```

---

## Command Execution Flow

### Load Command Flow

```
User Input: knowbase load --input ./docs
                  │
                  ▼
        ┌─────────────────────┐
        │ Click CLI Handler   │
        │ (Validates args)    │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ PreprocessingPipeline
        │ - Read files        │
        │ - Clean text        │
        │ - Chunk (256 char)  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ EmbeddingPipeline   │
        │ - Load model        │
        │ - Encode chunks     │
        │ - Move to device    │
        │ (batch_size=32)     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ VectorStorePipeline │
        │ - Create collection │
        │ - Add embeddings    │
        │ - Persist to disk   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Output Results      │
        │ (Progress, summary) │
        └─────────────────────┘
```

### Search Command Flow

```
User Input: knowbase search --query "..."
                  │
                  ▼
        ┌─────────────────────┐
        │ Click CLI Handler   │
        │ (Validate query)    │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ EmbeddingPipeline   │
        │ - Encode query      │
        │ (same model as docs)│
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ VectorStorePipeline │
        │ - Similarity search │
        │ - Return top-k (5)  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Format Results      │
        │ (Rich table, JSON)  │
        └─────────────────────┘
```

### Ask Command Flow (RAG)

```
User Input: knowbase ask "question"
                  │
                  ▼
        ┌─────────────────────┐
        │ Click CLI Handler   │
        └──────────┬──────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
    ▼                             ▼
┌────────────┐         ┌──────────────────┐
│ Embedding  │         │ Retrieve Context │
│ Query      │         │ (Vector search)  │
└──────┬─────┘         └────────┬─────────┘
       │                        │
       └────────────┬───────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ Build RAG Prompt    │
        │ Context + Query     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Call LLM API        │
        │ (OpenAI/Anthropic)  │
        │ (Stream thinking)   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Display Thinking    │
        │ + Final Answer      │
        └─────────────────────┘
```

---

## Data Flow

### Document Ingestion Pipeline

```
Input Documents
├── Text Files (.txt)
├── PDFs (.pdf)
├── Markdown (.md)
└── Subtitles (.srt)
           │
           ▼
    Preprocessing
    ├── Parse format
    ├── Extract text
    ├── Clean & normalize
    └── Split into chunks (256 chars, 50 overlap)
           │
           ▼
    Chunks (e.g., 66 chunks from 35 documents)
           │
           ▼
    Embedding Generation
    ├── Load model (BAAI/bge-large-en-v1.5)
    ├── Move to device (auto-detected)
    └── Batch encode → 1024-dim vectors
           │
           ▼
    Embeddings (e.g., 66 × 1024 matrix)
           │
           ▼
    Vector Store (ChromaDB)
    ├── Create collection
    ├── Store embeddings + metadata
    ├── Build indices
    └── Persist to ./data/vector_db/
           │
           ▼
    Ready for Search/RAG
```

### Search & Retrieval Flow

```
User Query
    │
    ▼
Embed Query
    │
    ▼
Vector (1024-dim)
    │
    ▼
ChromaDB Similarity Search
    ├── Compute cosine similarity
    ├── Return top-k results
    └── Include metadata (source, chunk_id)
    │
    ▼
Retrieved Context
    │
    ├─→ [For Search] Display results
    │
    └─→ [For Ask/RAG] Build LLM prompt
        │
        ▼
        LLM Response
```

---

## Configuration System

### Config Singleton Pattern

**File**: `src/cli/config.py`

**Purpose**: Centralized configuration management accessible from all commands

**Key Properties**:

```python
class Config:
    VECTOR_DB_PATH: str = "./data/vector_db"
    DEFAULT_MODEL: str = "BAAI/bge-large-en-v1.5"
    DEVICE: str = "auto"  # auto, cuda, mps, cpu
    BATCH_SIZE: int = 32
    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 50
```

**Usage**:

```python
# Any command can access global config
config = Config()
db_path = config.VECTOR_DB_PATH
device = config._resolve_device()  # auto → mps/cuda/cpu
```

### Environment Variables

**Supported**:

```bash
# Core
MODEL_NAME=BAAI/bge-large-en-v1.5
DEVICE=auto
BATCH_SIZE=32
VECTOR_DB_PATH=./data/vector_db

# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

---

## Integration Points

### External Services

#### ChromaDB (Vector Database)

- **Purpose**: Persistent storage of embeddings
- **Connection**: Local filesystem (./data/vector_db/)
- **Collections**: Model-specific (e.g., "subtitle_embeddings_bge_large")
- **Operations**: add, search, delete, query

#### Hugging Face Transformers

- **Purpose**: Loading embedding models
- **Models Supported**:
  - BAAI/bge-large-en-v1.5 (1024-dim, primary)
  - google/embedding-gemma-300m (768-dim)
  - Any HuggingFace sentence-transformer
- **Device Support**: auto, cuda, mps, cpu

#### LLM APIs

- **OpenAI**: gpt-4, gpt-3.5-turbo, text-embedding-3-large
- **Anthropic**: claude-3-sonnet, claude-3-opus (with extended thinking)
- **Groq**: llama2-70b-4096, mixtral-8x7b-32768 (fast inference)
- **Azure OpenAI**: Enterprise deployments
- **Ollama**: Local inference (optional)

### Data Formats

#### Input Formats

- Text files (.txt)
- PDFs (.pdf) - via pdf2image + OCR
- Markdown (.md)
- SRT subtitles (.srt)

#### Output Formats

- **Search Results**: Markdown table, JSON
- **Clustering**: JSON (clusters, stats), UMAP visualization
- **Export**: JSON (full chunks), CSV (simplified)

---

## Design Patterns

### 1. Pipeline Pattern

Each processing stage is a separate pipeline class:

```python
class Pipeline:
    def process(self, input_data) -> output_data:
        pass

# Usage
preprocessing = PreprocessingPipeline()
embeddings = EmbeddingPipeline()
vector_store = VectorStorePipeline()

chunks = preprocessing.process(documents)
vectors = embeddings.process(chunks)
vector_store.add(chunks, vectors)
```

### 2. Singleton Pattern

Configuration is a singleton accessible globally:

```python
# src/cli/config.py
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Usage
config = Config()  # Same instance everywhere
```

### 3. Factory Pattern

LLM creation uses factory for multi-provider support:

```python
# src/ai_search/llm_factory.py
class LLMFactory:
    @staticmethod
    def create(provider: str, model: str) -> LLM:
        if provider == "openai":
            return OpenAILLM(model)
        elif provider == "anthropic":
            return AnthropicLLM(model)
        # ...
```

### 4. Validation with Pydantic

Input validation for all CLI commands:

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=100)
    model: str = "BAAI/bge-large-en-v1.5"

# Usage
input_data = SearchInput(query="...", top_k=10)  # Validated!
```

### 5. Observer Pattern (Streaming)

LLM responses stream via callback:

```python
def stream_callback(token: str):
    print(token, end="", flush=True)

llm.generate(prompt=..., stream_callback=stream_callback)
```

---

## Performance Considerations

### Memory Management

**Device Resolution**:

```python
def _resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"
```

**Batch Processing**:

- Default: 32 chunks per batch
- Configurable: `--batch-size 8` for limited RAM
- ~150 embeddings fit in 4GB RAM (1024-dim vectors)

### Indexing Strategy

**ChromaDB Persistence**:

- Automatic index creation on add
- Fast similarity search (~100ms for 66 chunks)
- Replication on disk for durability

**Collection Naming**:

- Model-specific: "subtitle_embeddings_bge_large"
- Allows multiple models coexist
- Clean separation of data

### Search Optimization

- **Query Embedding**: ~50ms (same model as docs)
- **Similarity Search**: ~10ms (k=5)
- **Total Search Latency**: ~100ms (cached model)

---

## Extension Points

### Adding New Commands

1. Create file: `src/cli/commands/my_command.py`
2. Implement with Click decorators
3. Register in `src/cli/main.py`:
   ```python
   from src.cli.commands.my_command import my_command
   cli.add_command(my_command)
   ```

### Adding New Embedding Models

1. Update model list in config
2. ChromaDB auto-creates collection with model name
3. Example: `--model google/embedding-gemma-300m`

### Adding New LLM Provider

1. Create class in `src/ai_search/llm_factory.py`
2. Implement `generate()` and `stream()` methods
3. Register in LLMFactory.create()

---

## Testing Architecture

### Test Structure

```
tests/
├── test_cli_commands.py      # CLI command tests (CliRunner)
├── test_embeddings.py         # Embedding pipeline tests
├── test_clustering.py         # Clustering tests
└── conftest.py                # Pytest fixtures
```

### Test Strategy

- **Unit Tests**: Individual pipelines in isolation
- **Integration Tests**: Full command execution
- **CLI Tests**: Click CliRunner for command validation

**Example**:

```python
def test_search_command():
    runner = CliRunner()
    result = runner.invoke(search, ['--query', 'test'])
    assert result.exit_code == 0
```

---

## Deployment Considerations

### Local Development

- Install with `pip install -e .`
- Commands accessible via `knowbase` CLI
- Database persists in `./data/vector_db/`

### Production

- Use Docker container (Dockerfile provided)
- Mount volume for `./data/` persistence
- Set environment variables for API keys
- Configure logging to external service

### Scaling

- **Single Machine**: Current implementation optimal
- **Distributed**: Would require:
  - PostgreSQL backend for ChromaDB
  - Separate embedding server
  - Task queue for batch processing (Celery)

---

## Security Considerations

### API Keys

**Management**:

- Store in `.env` file (not in git)
- Read via environment variables
- Never log or expose keys

**File**:

```bash
# .env (example)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Data Privacy

- Embeddings stored locally (./data/vector_db/)
- No external vector database calls
- Metadata remains with embeddings

### Input Validation

- All CLI inputs validated with Pydantic
- Query length limits (1-1000 chars)
- Batch size bounds (1-256)
- Device whitelist (auto, cuda, mps, cpu)

---

## Monitoring & Observability

### Logging

**Configuration**: `src/utils/logger.py`

**Levels**:

- DEBUG: Detailed pipeline steps
- INFO: Command execution, batch progress
- WARNING: Model loading issues, API retries
- ERROR: Command failures, API errors

**Usage**:

```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Loaded {len(chunks)} chunks")
```

### Metrics

**Collected**:

- Documents processed
- Chunks generated
- Embeddings created
- Search latency
- Cluster count
- Noise points percentage

**Example**:

```bash
$ knowbase info
System Information:
- Device: mps (Apple Silicon)
- Default Model: BAAI/bge-large-en-v1.5
- Collections: 1 (subtitle_embeddings_bge_large)
- Total Embeddings: 66
```

---

## Future Enhancements

### Phase 5 (Planned)

1. **Caching**: Cache embeddings to reduce reprocessing
2. **Streaming Export**: Streaming JSON export for large datasets
3. **Web API**: FastAPI wrapper for programmatic access
4. **Performance**: Quantized models for faster inference
5. **Multi-Language**: Support for non-English documents

### Phase 6 (Considered)

1. **Distributed Processing**: Kubernetes deployment
2. **Real-time Indexing**: Continuous document stream processing
3. **Fine-tuning**: Domain-specific model adaptation
4. **Multi-Modal**: Image + text embeddings

---

## References

- [ChromaDB Documentation](https://docs.trychroma.com)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Click Documentation](https://click.palletsprojects.com)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [PyTorch Device Management](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)

---

**Last Updated:** December 4, 2025  
**Status:** Architecture Complete
**Maintainer:** KnowBase Team
