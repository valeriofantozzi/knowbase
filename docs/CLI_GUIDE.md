# KnowBase CLI Guide

Complete reference for the KnowBase command-line interface. All commands are accessible via the `knowbase` command after installation.

**Table of Contents:**

1. [Installation](#installation)
2. [Global Options](#global-options)
3. [Commands](#commands)
4. [Examples](#examples)
5. [Environment Variables](#environment-variables)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd knowbase

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install KnowBase CLI
pip install -e .
```

### Verify Installation

```bash
knowbase --version
knowbase --help
```

---

## Global Options

These options work with all commands:

```bash
knowbase [OPTIONS] COMMAND [ARGS]

Options:
  --version                    Show CLI version
  -v, --verbose               Enable verbose output (detailed logs)
  -c, --config PATH           Path to configuration YAML file
  -f, --format TEXT           Default output format (text/json/csv/table)
  --help                      Show help for command
```

### Examples

```bash
# Show version
knowbase --version

# Run command with verbose output
knowbase load --input ./docs -v

# Use custom config file
knowbase search --query "test" -c /path/to/config.yaml

# Set default output format to JSON
knowbase info -f json
```

---

## Commands

### `load` — Load Documents and Generate Embeddings

Load documents from a directory, preprocess, generate embeddings, and index into ChromaDB.

**Usage:**

```bash
knowbase load --input INPUT_PATH [OPTIONS]
```

**Required Arguments:**

- `-i, --input PATH` — Input file or directory path

**Options:**

- `-m, --model TEXT` — Embedding model (default: `BAAI/bge-large-en-v1.5`)
- `-d, --device TEXT` — Device: auto/cpu/cuda/mps (default: `auto`)
- `-b, --batch-size INT` — Batch size for embeddings (default: 32, max: 256)
- `--chunk-size INT` — Document chunk size (default: 512)
- `--chunk-overlap INT` — Overlap between chunks (default: 50)
- `--skip-duplicates` — Skip documents already in database
- `-c, --config PATH` — Configuration file
- `-v, --verbose` — Verbose output

**Examples:**

```bash
# Load all SRT files from subtitles directory
knowbase load --input ./subtitles

# Load with custom model
knowbase load --input ./docs --model google/embeddinggemma-300m

# Load with specific GPU and smaller batch size
knowbase load --input ./docs --device cuda --batch-size 16

# Load with custom chunk parameters
knowbase load --input ./docs --chunk-size 256 --chunk-overlap 25

# Skip duplicate documents
knowbase load --input ./new_docs --skip-duplicates

# Show detailed progress
knowbase load --input ./docs -v
```

**Output:**

```
Configuration:
  Model: BAAI/bge-large-en-v1.5
  Device: auto
  Batch Size: 32

🚀 Starting document load pipeline...
✓ Load completed successfully!

Summary:
  Documents processed: 35
  Total chunks: 66
  Chunks indexed: 66
  Elapsed time: 7.8s
  Average speed: 8 chunks/sec
```

---

### `search` — Semantic Search

Search for documents semantically similar to a query.

**Usage:**

```bash
knowbase search --query QUERY [OPTIONS]
```

**Required Arguments:**

- `-q, --query TEXT` — Search query (1-2000 characters)

**Options:**

- `-k, --top-k INT` — Number of results (default: 5, max: 50)
- `-m, --model TEXT` — Embedding model to use
- `-t, --threshold FLOAT` — Similarity threshold (0.0-1.0, default: 0.0)
- `-f, --format TEXT` — Output format: text/json/csv/table (default: text)
- `-c, --config PATH` — Configuration file
- `-v, --verbose` — Verbose output

**Examples:**

```bash
# Basic search
knowbase search --query "how to grow orchids?"

# Search with more results
knowbase search --query "orchid care" --top-k 10

# Search with specific model
knowbase search --query "orchid" --model google/embeddinggemma-300m

# Filter by similarity threshold
knowbase search --query "orchid" --threshold 0.5

# Export results as JSON
knowbase search --query "orchid" --format json > results.json

# Export as CSV
knowbase search --query "orchid" --format csv > results.csv

# Pretty table output
knowbase search --query "orchid" --format table

# Verbose mode shows which model/collection used
knowbase search --query "orchid" -v
```

**Output (text format):**

```
Query: "how to grow orchids?"
Model: BAAI/bge-large-en-v1.5
Results:

1. [0.89] Document title...
   Source: 20231028_*.srt

2. [0.86] Another document...
   Source: 20231015_*.srt
```

---

### `ask` — RAG Query (Question Answering)

Ask questions about the knowledge base. Uses Retrieval-Augmented Generation (RAG) with an LLM to provide comprehensive answers based on retrieved context.

**Usage:**

```bash
knowbase ask QUESTION [OPTIONS]
```

**Required Arguments:**

- `QUESTION` — Question to ask (positional argument)

**Options:**

- `-m, --model TEXT` — Embedding model for retrieval (default: `BAAI/bge-large-en-v1.5`)
- `-k, --top-k INT` — Number of documents to retrieve (default: 5)
- `-p, --llm-provider TEXT` — LLM provider: openai/anthropic/groq/azure/ollama
- `-t, --temperature FLOAT` — LLM temperature (0.0-2.0, default: 0.7)
- `--show-thinking / --no-thinking` — Display agent thinking process
- `--stream / --no-stream` — Stream output (experimental)
- `-f, --format TEXT` — Output format: text/json
- `-c, --config PATH` — Configuration file
- `-v, --verbose` — Verbose output

**Prerequisites:**

Set up LLM API keys in `.env`:

```bash
# OpenAI (default)
OPENAI_API_KEY=sk-...

# Or other providers
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
```

**Examples:**

```bash
# Basic question (requires OPENAI_API_KEY)
knowbase ask "What are best practices for orchid care?"

# Show thinking process
knowbase ask "How to grow orchids faster?" --show-thinking

# Use Anthropic instead of OpenAI
knowbase ask "..." --llm-provider anthropic

# Retrieve more context
knowbase ask "..." --top-k 10

# Lower temperature for focused answers
knowbase ask "..." --temperature 0.3

# Higher temperature for creative answers
knowbase ask "..." --temperature 1.5

# Get answer as JSON with sources
knowbase ask "..." --format json > answer.json

# Verbose mode shows retrieved documents
knowbase ask "..." -v
```

**Output (text format):**

```
💭 Thinking Process:
🔍 Analyzing query...
📚 Retrieving documents...
✍️ Generating answer...

Question: "What are best practices for orchid care?"

Answer:
Based on the knowledge base, here are the key practices:
1. Proper watering...
2. Nutrient-rich fertilizers...

Sources:
1. 20231028_*.srt
2. 20231015_*.srt
```

---

### `cluster` — Document Clustering Analysis

Analyze document embeddings using HDBSCAN clustering and optional UMAP visualization.

**Usage:**

```bash
knowbase cluster [OPTIONS]
```

**Options:**

- `-m, --model TEXT` — Embedding model (default: `BAAI/bge-large-en-v1.5`)
- `--min-cluster-size INT` — Minimum cluster size for HDBSCAN (default: 5)
- `--min-samples INT` — Minimum samples for HDBSCAN (default: 5)
- `--export-umap PATH` — Export UMAP projection to JSON file
- `--export-metadata PATH` — Export cluster metadata to JSON file
- `-f, --format TEXT` — Output format: text/json/table (default: text)
- `-c, --config PATH` — Configuration file
- `-v, --verbose` — Verbose output

**Prerequisites:**

```bash
# Install optional dependencies
pip install hdbscan umap-learn
```

**Examples:**

```bash
# Basic clustering analysis
knowbase cluster

# More lenient clustering (smaller min size)
knowbase cluster --min-cluster-size 3

# Stricter clustering (larger min size)
knowbase cluster --min-cluster-size 10

# Export UMAP projection for visualization
knowbase cluster --export-umap clusters.json

# Export full cluster metadata
knowbase cluster --export-metadata clusters_meta.json

# JSON output
knowbase cluster --format json > analysis.json

# Verbose mode shows cluster details
knowbase cluster -v
```

**Output:**

```
Total documents: 132
Number of clusters: 4
Noise points: 71

Cluster Details:

  Cluster 0:
    Size: 7 (5.3%)
    Avg distance to centroid: 0.3290
    Sample documents:
      - document1.srt
      - document2.srt

  Cluster 1:
    Size: 18 (13.6%)
    ...
```

---

### `export` — Export Collections

Export indexed documents to JSON or CSV format. Supports streaming for large datasets.

**Usage:**

```bash
knowbase export --output OUTPUT_PATH [OPTIONS]
```

**Required Arguments:**

- `-o, --output PATH` — Output file path (JSON or CSV)

**Options:**

- `-m, --model TEXT` — Embedding model (default: `BAAI/bge-large-en-v1.5`)
- `-f, --format TEXT` — Format: json/csv (auto-detected from file extension)
- `--include-embeddings` — Include embeddings in export (JSON only, large file)
- `--batch-size INT` — Batch size for streaming (default: 100)
- `-c, --config PATH` — Configuration file
- `-v, --verbose` — Verbose output

**Examples:**

```bash
# Export to JSON
knowbase export --output documents.json

# Export to CSV
knowbase export --output documents.csv --format csv

# Include embeddings (warning: large file)
knowbase export --output with_embeddings.json --include-embeddings

# Custom batch size for large datasets
knowbase export --output large.json --batch-size 50

# Verbose output shows progress
knowbase export --output data.json -v
```

**Output Files:**

JSON structure:

```json
{
  "documents": [
    {
      "id": "...",
      "content": "...",
      "metadata": {
        "filename": "...",
        "chunk_index": 0,
        "token_count": 297
      }
    }
  ]
}
```

CSV columns: `id, content, metadata`

---

### `reindex` — Reindex with Different Model

Reindex all documents using a different embedding model. Preserves all metadata while updating embeddings.

**Usage:**

```bash
knowbase reindex --new-model MODEL [OPTIONS]
```

**Required Arguments:**

- `-n, --new-model TEXT` — Target embedding model

**Options:**

- `--from-model TEXT` — Source model (default: `BAAI/bge-large-en-v1.5`)
- `-b, --batch-size INT` — Batch size for processing (default: 32)
- `-d, --device TEXT` — Device: auto/cpu/cuda/mps
- `--skip-backup` — Skip creating backup of original collection
- `-c, --config PATH` — Configuration file
- `-v, --verbose` — Verbose output

**Examples:**

```bash
# Reindex with different model
knowbase reindex --new-model google/embeddinggemma-300m

# Reindex from one model to another
knowbase reindex --from-model google/embeddinggemma-300m --new-model BAAI/bge-large

# Use GPU for reindexing
knowbase reindex --new-model BAAI/bge-large --device cuda

# Custom batch size
knowbase reindex --new-model BAAI/bge-large --batch-size 16

# Verbose mode shows progress
knowbase reindex --new-model BAAI/bge-large -v
```

**Output:**

```
⚠️  Reindexing Operation
From model: BAAI/bge-large-en-v1.5
To model:   google/embeddinggemma-300m

✓ Retrieved 66 documents
✓ Reindexing completed successfully!

Summary:
  Source collection: subtitle_embeddings_bge_large
  Target collection: document_embeddings_gemma_300m
  Documents reindexed: 66
  Model: BAAI/bge-large-en-v1.5 → google/embeddinggemma-300m
```

---

### `info` — System Information

Display system information, database statistics, and configuration.

**Usage:**

```bash
knowbase info [OPTIONS]
```

**Options:**

- `-c, --config PATH` — Configuration file
- `-v, --verbose` — Verbose output

**Examples:**

```bash
# Show system information
knowbase info

# Verbose mode shows detailed info
knowbase info -v
```

**Output:**

```
KnowBase System Information

Database
  Location: data/vector_db
  Size: 125 MB

Models Available
  ✓ BAAI/bge-large-en-v1.5 (66 documents)
  ✓ google/embeddinggemma-300m (0 documents)

Hardware
  Device: mps (Apple Silicon)
  CPU cores: 16
  Memory: 48.0 GB total, 26.6 GB available
  GPU: Metal Performance Shaders (MPS) available
```

---

## Examples

### Typical Workflow

```bash
# 1. Load documents
knowbase load --input ./documents

# 2. Search for information
knowbase search --query "topic of interest"

# 3. Get detailed answers using RAG
knowbase ask "What are the key findings about this topic?"

# 4. Analyze document clusters
knowbase cluster

# 5. Export for downstream processing
knowbase export --output results.json
```

### Batch Processing

```bash
# Load multiple document sources
knowbase load --input ./docs/source1
knowbase load --input ./docs/source2

# Search and export results
knowbase search --query "something" --format json > results.json

# Process with external tools
cat results.json | jq '.' | ...
```

### Model Comparison

```bash
# Index with Model A
knowbase load --input ./docs --model BAAI/bge-large

# Search with Model A
knowbase search --query "test" --model BAAI/bge-large

# Index with Model B
knowbase load --input ./docs --model google/embeddinggemma-300m

# Compare results
knowbase search --query "test" --model google/embeddinggemma-300m
```

### RAG Pipeline

```bash
# Load documents
knowbase load --input ./knowledge_base

# Ask questions with thinking visible
knowbase ask "Complex question?" --show-thinking --top-k 10

# Export retrieved context
knowbase ask "What did the documents say about X?" --format json > context.json
```

---

## Environment Variables

Configure KnowBase via environment variables or `.env` file:

```bash
# Embedding configuration
MODEL_NAME=BAAI/bge-large-en-v1.5
DEVICE=auto
BATCH_SIZE=32

# ChromaDB
VECTOR_DB_PATH=./data/vector_db
COLLECTION_NAME=documents

# LLM / RAG Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Agent behavior
AI_QUERY_CLARITY_THRESHOLD=0.85
AI_CONVERSATION_WINDOW=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

---

## Troubleshooting

### "command not found: knowbase"

**Solution:** Ensure CLI is installed:

```bash
pip install -e .
```

### "No documents found in database"

**Solution:** Load documents first:

```bash
knowbase load --input ./your_documents
```

### "API key not found"

**Solution:** Set environment variables:

```bash
export OPENAI_API_KEY=sk-...
# Or in .env file
echo "OPENAI_API_KEY=sk-..." >> .env
```

### "Collection not found"

**Solution:** Check available models:

```bash
knowbase info
```

Reindex with correct model:

```bash
knowbase reindex --new-model BAAI/bge-large-en-v1.5
```

### "Out of memory during load"

**Solution:** Reduce batch size:

```bash
knowbase load --input ./docs --batch-size 8
```

Or use CPU:

```bash
knowbase load --input ./docs --device cpu
```

### "Slow performance"

**Solution:** Check hardware and device:

```bash
knowbase info
```

Force GPU usage:

```bash
knowbase load --input ./docs --device cuda  # or mps
```

---

## Getting Help

Show help for any command:

```bash
knowbase --help              # General help
knowbase load --help         # Load command help
knowbase search --help       # Search command help
knowbase ask --help          # Ask command help
```

For detailed technical information, see:

- 📖 [`USER_GUIDE.md`](../USER_GUIDE.md) — Python API and advanced usage
- 🏗️ [`ARCHITECTURE.md`](ARCHITECTURE.md) — System design and pipelines
- 📋 [`PHASE_3_COMPLETE.md`](../PHASE_3_COMPLETE.md) — Implementation details

---

**Last Updated:** December 4, 2025  
**CLI Version:** 0.1.0  
**Status:** Production Ready
