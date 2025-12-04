# 🧠 KnowBase — Immersive Semantic Search for Documents

A practical and straightforward toolkit to transform document collections (SRT, PDF, TXT, Markdown...) into a semantically searchable knowledge base. Uses multiple embedding models, model-isolated collections, and an integrated web UI to explore results.

For developers and power users: easy to extend, designed for testing different models and pipelines without breaking existing indices.

**✨ Highlights**

- 🤖 **Multi-model**: support for `BAAI/bge-large-en-v1.5` and `google/embeddinggemma-300m` (and others via adapters)
- 🔐 **Isolated collections**: each model writes to separate ChromaDB collections
- 🔄 **Modular pipeline**: parsing → chunking → embeddings → store → retrieval
- 🎛️ **Interfaces**: CLI scripts for batch, programmatic API, and Streamlit interface for exploration

**⚡ Ready for prototyping and experimentation**: model caching, dynamic device selection (CPU, CUDA, MPS), and helpers for quality comparison between models.

**🚀 Quick Start (Installation & Usage)**

### Setup

1. 📦 Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 📥 Install dependencies:

```bash
pip install -r requirements.txt
```

3. 🔧 Install KnowBase CLI:

```bash
pip install -e .
```

### Using the CLI (Recommended)

The new **KnowBase CLI** provides an easy command-line interface for all operations:

```bash
# Load documents, generate embeddings, and index
knowbase load --input ./subtitles --model BAAI/bge-large-en-v1.5

# Search for documents
knowbase search --query "how to care for orchids?" --top-k 5

# Ask questions using RAG (Retrieval-Augmented Generation)
knowbase ask "What are the best practices for growing orchids?"

# Analyze document clustering
knowbase cluster --min-cluster-size 5

# Export indexed documents to JSON/CSV
knowbase export --output documents.json --format json

# Reindex with a different embedding model
knowbase reindex --new-model google/embeddinggemma-300m

# View system information and statistics
knowbase info
```

**📚 For detailed CLI examples and options, see [`docs/CLI_GUIDE.md`](docs/CLI_GUIDE.md)**

### Using the Web UI (Alternative)

```bash
./start_viewer.sh
```

Opens Streamlit interface at `http://localhost:8501`

### Legacy Script Interface (Deprecated)

For backward compatibility, legacy scripts are still available:

```bash
# Process files
python scripts/process_subtitles.py --input subtitles/ --output data/processed

# Search
python scripts/query_subtitles.py "how to care for an orchid?"
```

⚠️ **Note**: CLI commands are now the recommended interface. Legacy scripts may be removed in future versions.

**💡 Why it's awesome?**

- ⚡ Swap models on the fly: compare embeddings from different models without mixing your data.
- 🔌 Easily extensible: the adapter pattern makes adding a new model minimal.
- ⏱️ Built for SRT and temporal documents (subtitle-aware chunking).

**📁 Key repository structure**

- 🎯 `src/cli/` — **New!** Click-based CLI commands (load, search, ask, cluster, export, reindex, info)
- 🧠 `src/embeddings/` — adapters, loaders, and pipelines to generate embeddings.
- 🔤 `src/preprocessing/` — SRT parser, chunker, text normalization.
- 🗄️ `src/vector_store/` — ChromaDB management, naming for model-specific collections.
- 🤖 `src/ai_search/` — LLM integration, RAG chains, agent thinking state
- 🛠️ `scripts/` — Legacy CLI scripts (deprecated, use CLI instead).
- 📚 `docs/` — Documentation (CLI guide, architecture, etc.)
- 🎨 `streamlit_app.py` — web interface to explore searches and switch models.

**📌 Quick Reference**

**CLI Commands** (new recommended interface):

- `knowbase load` — Load documents and generate embeddings
- `knowbase search` — Semantic search with ranking
- `knowbase ask` — RAG queries with LLM (requires API key)
- `knowbase cluster` — Document clustering analysis
- `knowbase export` — Export to JSON/CSV
- `knowbase reindex` — Model migration
- `knowbase info` — System statistics

**Collections**:

- BGE: `subtitle_embeddings_bge_large` (1024-dim)
- Gemma: `document_embeddings_gemma_300m` (768-dim)

**Configuration**:

- Environment file: `.env` (load via dotenv)
- LLM providers: OpenAI, Anthropic, Groq, Azure, Ollama
- Devices: auto, cpu, cuda, mps (Apple Silicon)

📖 **Documentation**:

- 📚 [`USER_GUIDE.md`](USER_GUIDE.md) — Detailed technical instructions, Python API, advanced usage
- 🎯 [`docs/CLI_GUIDE.md`](docs/CLI_GUIDE.md) — **Complete CLI reference** with examples for all 7 commands
- 🔧 [`docs/INSTALLATION.md`](docs/INSTALLATION.md) — **Step-by-step installation** guide for all platforms
- 🏗️ [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — **System design & architecture**, pipeline flows, module organization
- 📋 [`PHASE_2_COMPLETE.md`](PHASE_2_COMPLETE.md) — Phase 2 (Core Commands) implementation summary
- 📋 [`PHASE_3_COMPLETE.md`](PHASE_3_COMPLETE.md) — Phase 3 (Advanced Commands) implementation summary
- 📋 [`PHASE_4_COMPLETE.md`](PHASE_4_COMPLETE.md) — **Phase 4 (Documentation & Packaging) implementation summary** ✅ PRODUCTION READY
