# KnowBase Installation Guide

Complete installation instructions for KnowBase on different platforms and environments.

**Table of Contents:**

1. [System Requirements](#system-requirements)
2. [Quick Install](#quick-install)
3. [Detailed Installation](#detailed-installation)
4. [Platform-Specific Notes](#platform-specific-notes)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.11 or later
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: 5 GB (for models and database)
- **Internet**: Required for downloading models

### Optional Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8+ (for `--device cuda`)
- **Apple Silicon**: Native MPS support for M1/M2/M3 (auto-detected)
- **Docker**: For containerized deployment

---

## Quick Install

For users who want to get started quickly:

```bash
# 1. Clone repository
git clone https://github.com/valeriofantozzi/knowbase.git
cd knowbase

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. Install KnowBase CLI
pip install -e .

# 5. Verify installation
knowbase --help
```

That's it! You can now use the CLI.

---

## Detailed Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/valeriofantozzi/knowbase.git
cd knowbase
```

### Step 2: Python Virtual Environment

**Linux/macOS:**

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Upgrade pip and Tools

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Optional: Install dev dependencies for testing
pip install -r requirements-dev.txt
```

### Step 5: Install KnowBase CLI

```bash
# Install in development mode (recommended for testing/development)
pip install -e .

# Or install in normal mode
pip install .
```

### Step 6: Verify Installation

```bash
# Check CLI is available
knowbase --version
knowbase --help

# Test basic command
knowbase info
```

---

## Platform-Specific Notes

### macOS with Apple Silicon (M1/M2/M3)

**MPS (Metal Performance Shaders) is automatically detected:**

```bash
# Device will be auto-detected as 'mps'
knowbase load --input ./docs --device auto

# Or explicitly use MPS
knowbase load --input ./docs --device mps
```

**Known Issues:**

- Some versions of PyTorch may have issues with MPS. If you encounter problems:
  ```bash
  pip install torch==2.0.1 -f https://download.pytorch.org/whl/torch_stable.html
  ```

### Linux with NVIDIA GPU

**Requirements:**

- NVIDIA GPU with CUDA Compute Capability 3.5+
- NVIDIA Driver 450.0+
- CUDA Toolkit 11.8+
- cuDNN 8.0+

**Installation:**

```bash
# PyTorch will be installed with CUDA support via requirements.txt
# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Use CUDA device
knowbase load --input ./docs --device cuda
```

### Linux without GPU (CPU Only)

```bash
# CPU processing is supported by default
# No special installation needed

# Force CPU usage
knowbase load --input ./docs --device cpu
```

### Windows

**Visual C++ Build Tools Required:**

- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Or install Microsoft C++ Build Tools

**Installation:**

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install KnowBase
pip install -e .

# Verify
knowbase --help
```

---

## Optional Dependencies

### For Clustering Analysis

Required for `knowbase cluster` command:

```bash
pip install hdbscan umap-learn
```

### For Development and Testing

```bash
pip install -r requirements-dev.txt
```

Includes:

- `pytest` — Unit testing
- `black` — Code formatting
- `mypy` — Type checking
- `pylint` — Linting

### For Specific LLM Providers

```bash
# Anthropic Claude
pip install anthropic

# Groq
pip install groq

# Azure OpenAI
# (included in langchain-openai which is already installed)

# Ollama local inference
# (requires Ollama to be running separately)
# Download from: https://ollama.ai
```

---

## Configuration Setup

### Environment Variables

Create a `.env` file in the project root:

```bash
# Core Configuration
MODEL_NAME=BAAI/bge-large-en-v1.5
DEVICE=auto
BATCH_SIZE=32
VECTOR_DB_PATH=./data/vector_db

# LLM API Keys (for 'ask' command)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Getting API Keys

**OpenAI:**

1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Add to `.env`: `OPENAI_API_KEY=sk-...`

**Anthropic:**

1. Go to https://console.anthropic.com
2. Get your API key
3. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

**Groq:**

1. Go to https://console.groq.com
2. Create API key
3. Add to `.env`: `GROQ_API_KEY=gsk_...`

---

## Verification

### Check Installation

```bash
# Verify Python version
python --version  # Should be 3.11+

# Verify virtual environment is active
which python  # Should show path to .venv

# Verify CLI is installed
knowbase --version
knowbase --help
```

### Test Commands

```bash
# Test system info
knowbase info

# Test with sample data (if you have documents)
knowbase load --input ./sample_docs
knowbase search --query "test"
```

### Verify GPU/Device

```bash
# Check available devices
knowbase info

# Explicitly test device
knowbase load --input /tmp --device auto
```

---

## Troubleshooting

### "command not found: knowbase"

**Problem**: CLI command not recognized after installation.

**Solutions:**

1. Ensure virtual environment is activated:

   ```bash
   source .venv/bin/activate
   ```

2. Reinstall CLI:

   ```bash
   pip install -e .
   ```

3. Check Python path:
   ```bash
   which python
   python -m pip show knowbase
   ```

### "ModuleNotFoundError: No module named 'src'"

**Problem**: Python imports fail.

**Solutions:**

1. Ensure you're in the correct directory:

   ```bash
   pwd  # Should be knowbase root
   ```

2. Reinstall in editable mode:

   ```bash
   pip install -e .
   ```

3. Check PYTHONPATH:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

### "torch.cuda.is_available() returns False"

**Problem**: GPU not detected despite having NVIDIA GPU.

**Solutions:**

1. Check NVIDIA driver:

   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch with CUDA:

   ```bash
   pip uninstall torch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Force CPU as fallback:
   ```bash
   knowbase load --input ./docs --device cpu
   ```

### "CUDA out of memory"

**Problem**: GPU memory exhausted during processing.

**Solutions:**

1. Reduce batch size:

   ```bash
   knowbase load --input ./docs --batch-size 8
   ```

2. Use CPU instead:

   ```bash
   knowbase load --input ./docs --device cpu
   ```

3. Process smaller datasets:
   ```bash
   knowbase load --input ./small_docs
   ```

### "APIError: invalid_api_key"

**Problem**: API key error when using `ask` command.

**Solutions:**

1. Verify API key is set:

   ```bash
   echo $OPENAI_API_KEY
   ```

2. Check key format (should start with `sk-`):

   ```bash
   cat .env | grep OPENAI
   ```

3. Get new key from API provider dashboard

4. Test connection:
   ```bash
   python -c "from openai import OpenAI; print('OK')"
   ```

### "Permission denied" on Linux/macOS

**Problem**: Cannot access database or log files.

**Solutions:**

```bash
# Fix directory permissions
chmod -R 755 data/
chmod -R 755 logs/

# Or run with sudo (not recommended)
sudo knowbase load --input ./docs
```

### Dependencies Conflict

**Problem**: "Requirement already satisfied" with version conflicts.

**Solutions:**

1. Clear pip cache:

   ```bash
   pip cache purge
   ```

2. Reinstall in clean environment:

   ```bash
   deactivate
   rm -rf .venv
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. Use specific versions:
   ```bash
   pip install 'torch==2.0.1'
   ```

---

## Docker Installation (Optional)

For containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Set entrypoint
ENTRYPOINT ["knowbase"]
```

**Build and run:**

```bash
# Build image
docker build -t knowbase .

# Run container
docker run -v $(pwd)/data:/app/data knowbase load --input /data
```

---

## Next Steps

After installation, check out:

1. **Quick Start**: [`CLI_GUIDE.md`](docs/CLI_GUIDE.md)
2. **Examples**: See "Examples" section in CLI_GUIDE
3. **Advanced Usage**: [`USER_GUIDE.md`](USER_GUIDE.md)
4. **API Reference**: Python code in `src/`

---

## Getting Help

- 📖 Check documentation
- 🐛 Open an issue on GitHub
- 💬 Check existing discussions

---

**Last Updated:** December 4, 2025  
**Status:** Production Ready
