# 🧠 KnowBase — Immersive Semantic Search for Documents

A practical and straightforward toolkit to transform document collections (SRT, PDF, TXT, Markdown...) into a semantically searchable knowledge base. Uses multiple embedding models, model-isolated collections, and an integrated web UI to explore results.

For developers and power users: easy to extend, designed for testing different models and pipelines without breaking existing indices.

**✨ Highlights**

- 🤖 **Multi-model**: support for `BAAI/bge-large-en-v1.5` and `google/embeddinggemma-300m` (and others via adapters)
- 🔐 **Isolated collections**: each model writes to separate ChromaDB collections
- 🔄 **Modular pipeline**: parsing → chunking → embeddings → store → retrieval
- 🎛️ **Interfaces**: CLI scripts for batch, programmatic API, and Streamlit interface for exploration

**⚡ Ready for prototyping and experimentation**: model caching, dynamic device selection (CPU, CUDA, MPS), and helpers for quality comparison between models.

**🚀 Quick TL;DR (quick example)**

1. 📦 Create and activate a virtualenv:

```
python -m venv .venv
source .venv/bin/activate
```

2. 📥 Install dependencies:

```
pip install -r requirements.txt
```

3. ⚙️ Process files (default model set in `.env`):

```
python scripts/process_subtitles.py --input subtitles/ --output data/processed
```

4. 🔍 Search in indexed data:

```
python scripts/query_subtitles.py "how to care for an orchid?"
```

5. 🌐 Start the web UI:

```
./start_viewer.sh
```

**💡 Why it's awesome?**

- ⚡ Swap models on the fly: compare embeddings from different models without mixing your data.
- 🔌 Easily extensible: the adapter pattern makes adding a new model minimal.
- ⏱️ Built for SRT and temporal documents (subtitle-aware chunking).

**📁 Key repository structure**

- 🧠 `src/embeddings/` — adapters, loaders, and pipelines to generate embeddings.
- 🔤 `src/preprocessing/` — SRT parser, chunker, text normalization.
- 🗄️ `src/vector_store/` — ChromaDB management, naming for model-specific collections.
- 🛠️ `scripts/` — CLI scripts for processing, migrating, and querying the DB.
- 🎨 `streamlit_app.py` — web interface to explore searches and switch models.

**📌 Quick reference**

- 📚 Collections:
  - BGE: `document_embeddings_bge_large`
  - Gemma: `document_embeddings_gemma_300m`
- 📄 Useful files: `requirements.txt`, `start_viewer.sh`, `scripts/process_subtitles.py`

📖 Want to go deeper? Open `USER_GUIDE.md` for detailed technical instructions, CLI examples, and snippets for using pipelines from Python code.
