# Multi-Agent Configuration Guide

## Overview

Il sistema AI Search utilizza un'architettura **multi-agent** dove ogni agente può utilizzare un modello LLM diverso con un provider diverso. Ciò consente di ottimizzare i costi e le prestazioni per ogni task specifico.

## Architettura degli Agenti

### 1. **Query Analyzer Agent**

- **Ruolo**: Analizza la chiarezza e specificità della domanda dell'utente
- **Output**: JSON con analisi di qualità
- **Configurazione ideale**: Modello veloce e leggero (es. gpt-4-mini, claude-3-haiku)
- **Temperature**: Bassa (0.3) - analisi deterministica

### 2. **Clarification Agent**

- **Ruolo**: Genera richieste di chiarimento amichevoli e suggerisce domande migliori
- **Output**: Testo conversazionale
- **Configurazione ideale**: Modello creativo (es. gpt-3.5-turbo, claude-3-sonnet)
- **Temperature**: Alta (0.7) - generazione più variegata

### 3. **Query Rewriter Agent**

- **Ruolo**: Normalizza la domanda per la ricerca vettoriale
- **Output**: Domanda riscritta
- **Configurazione ideale**: Modello veloce (es. gpt-3.5-turbo)
- **Temperature**: Bassa (0.3) - riscrittura coerente

### 4. **RAG Response Generator Agent**

- **Ruolo**: Genera la risposta finale basata su documenti recuperati
- **Output**: Risposta sintetica basata su contesto
- **Configurazione ideale**: Modello intelligente (es. gpt-4, claude-3-opus)
- **Temperature**: Molto bassa (0.2) - risposte accurate

## Configurazione nel `.env`

Ogni agente ha una propria sezione di configurazione indipendente:

```env
# QUERY ANALYZER AGENT
AI_QUERY_ANALYZER_PROVIDER=openai          # Provider: openai, anthropic, groq, azure, ollama
AI_QUERY_ANALYZER_MODEL=gpt-4-mini         # Nome del modello
AI_QUERY_ANALYZER_TEMPERATURE=0.3          # Temperatura (0.0-1.0)
AI_QUERY_ANALYZER_MAX_TOKENS=500           # Massimo tokens per risposta
AI_QUERY_ANALYZER_API_KEY=${OPENAI_API_KEY} # API key (eredita da default se non specificata)

# CLARIFICATION AGENT
AI_CLARIFICATION_PROVIDER=openai
AI_CLARIFICATION_MODEL=gpt-3.5-turbo
AI_CLARIFICATION_TEMPERATURE=0.7
AI_CLARIFICATION_MAX_TOKENS=800
AI_CLARIFICATION_API_KEY=${OPENAI_API_KEY}

# QUERY REWRITER AGENT
AI_QUERY_REWRITER_PROVIDER=openai
AI_QUERY_REWRITER_MODEL=gpt-3.5-turbo
AI_QUERY_REWRITER_TEMPERATURE=0.3
AI_QUERY_REWRITER_MAX_TOKENS=300
AI_QUERY_REWRITER_API_KEY=${OPENAI_API_KEY}

# RAG RESPONSE GENERATOR AGENT
AI_RAG_GENERATOR_PROVIDER=openai
AI_RAG_GENERATOR_MODEL=gpt-4
AI_RAG_GENERATOR_TEMPERATURE=0.2
AI_RAG_GENERATOR_MAX_TOKENS=500
AI_RAG_GENERATOR_API_KEY=${OPENAI_API_KEY}
```

## Provider Supportati

### OpenAI

```env
OPENAI_API_KEY=sk-...
AI_QUERY_ANALYZER_PROVIDER=openai
AI_QUERY_ANALYZER_MODEL=gpt-4-mini  # o gpt-4, gpt-3.5-turbo, etc
```

### Anthropic (Claude)

```env
ANTHROPIC_API_KEY=sk-ant-...
AI_QUERY_ANALYZER_PROVIDER=anthropic
AI_QUERY_ANALYZER_MODEL=claude-3-haiku-20240307  # o claude-3-sonnet, claude-3-opus
```

### Groq (Inference API a bassa latenza)

```env
GROQ_API_KEY=gsk-...
AI_QUERY_ANALYZER_PROVIDER=groq
AI_QUERY_ANALYZER_MODEL=mixtral-8x7b-32768  # o altri modelli Groq
```

### Azure OpenAI

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AI_QUERY_ANALYZER_PROVIDER=azure
AI_QUERY_ANALYZER_MODEL=your-deployment-name
```

### Ollama (Local Inference)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral  # o altri modelli locali
AI_QUERY_ANALYZER_PROVIDER=ollama
AI_QUERY_ANALYZER_MODEL=mistral  # opzionale se vuoi override
```

## Esempi di Configurazione

### Configurazione Budget-Friendly

```env
# Usa modelli piccoli e veloci per ridurre costi
AI_QUERY_ANALYZER_PROVIDER=groq
AI_QUERY_ANALYZER_MODEL=mixtral-8x7b-32768  # Veloce e gratis

AI_CLARIFICATION_PROVIDER=groq
AI_CLARIFICATION_MODEL=mixtral-8x7b-32768

AI_QUERY_REWRITER_PROVIDER=groq
AI_QUERY_REWRITER_MODEL=mixtral-8x7b-32768

AI_RAG_GENERATOR_PROVIDER=groq
AI_RAG_GENERATOR_MODEL=mixtral-8x7b-32768
```

### Configurazione Ibrida (OpenAI + Anthropic)

```env
# Usa il migliore per ogni task
AI_QUERY_ANALYZER_PROVIDER=anthropic
AI_QUERY_ANALYZER_MODEL=claude-3-haiku-20240307  # Veloce per analisi

AI_CLARIFICATION_PROVIDER=anthropic
AI_CLARIFICATION_MODEL=claude-3-sonnet-20240229  # Creativo per chiarimenti

AI_QUERY_REWRITER_PROVIDER=openai
AI_QUERY_REWRITER_MODEL=gpt-3.5-turbo  # Efficiente

AI_RAG_GENERATOR_PROVIDER=anthropic
AI_RAG_GENERATOR_MODEL=claude-3-opus-20240229  # Migliore per risposte complesse
```

### Configurazione Locale (Ollama)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

AI_QUERY_ANALYZER_PROVIDER=ollama
AI_QUERY_ANALYZER_MODEL=mistral

AI_CLARIFICATION_PROVIDER=ollama
AI_CLARIFICATION_MODEL=mistral

AI_QUERY_REWRITER_PROVIDER=ollama
AI_QUERY_REWRITER_MODEL=mistral

AI_RAG_GENERATOR_PROVIDER=ollama
AI_RAG_GENERATOR_MODEL=mistral
```

## Comportamento degli Agenti

### Agent Behavior Configuration

```env
# Soglia di confidenza per richiedere chiarimenti (0.0-1.0)
# Più alto = più rigido nel richiedere chiarimenti
AI_QUERY_CLARITY_THRESHOLD=0.85

# Massimo numero di domande suggerite da mostrare
AI_MAX_SUGGESTED_QUERIES=5

# Numero di messaggi precedenti da includere nel contesto conversazionale
AI_CONVERSATION_WINDOW=10
```

## Implementazione nella Codebase

### 1. LLM Factory (`src/ai_search/llm_factory.py`)

Gestisce la creazione e il caching degli LLM per ogni agente:

```python
from src.ai_search.llm_factory import LLMFactory

# Inizializzare la factory
LLMFactory.initialize(config)

# Creare LLM per un agente specifico
llm_analyzer = LLMFactory.create_query_analyzer_llm()
llm_rag = LLMFactory.create_rag_generator_llm()

# O creare manualmente
llm = LLMFactory.create_llm(
    provider="openai",
    model="gpt-4",
    temperature=0.2,
    max_tokens=500,
    api_key="sk-..."
)
```

### 2. Chains (`src/ai_search/chains.py`)

Usa automaticamente i LLM configurati per ogni agente:

```python
from src.ai_search.chains import (
    query_analyzer_chain,      # Usa LLM Query Analyzer
    clarification_chain,        # Usa LLM Clarification
    query_rewriter_chain,       # Usa LLM Query Rewriter
    rag_chain                   # Usa LLM RAG Generator
)
```

### 3. Graph (`src/ai_search/graph.py`)

Il graph legge CLARITY_THRESHOLD dalla configurazione:

```python
from src.utils.config import Config

config = Config()
CLARITY_THRESHOLD = config.AI_QUERY_CLARITY_THRESHOLD
```

## Miglioramento delle Prestazioni

### Riduzione Costi

1. Usa Groq per agenti veloci e poco critici
2. Usa OpenAI GPT-3.5-turbo invece di GPT-4 dove possibile
3. Usa Claude Haiku per task semplici

### Miglioramento Qualità

1. Usa GPT-4 o Claude-3-Opus per RAG Generator
2. Aumenta temperature per Clarification (0.7-0.9)
3. Diminuisci temperature per Query Rewriter (0.1-0.3)

### Latenza Bassa

1. Usa Groq per tutti gli agenti
2. Riduci max_tokens per ogni agente
3. Usa modelli più piccoli (7B-13B range)

## Troubleshooting

### Errore: "OPENAI_API_KEY is not set"

```env
# Assicurati che la API key sia configurata
OPENAI_API_KEY=sk-...
```

### Errore: "Unsupported LLM provider"

Controlla che il provider sia uno di: `openai`, `anthropic`, `groq`, `azure`, `ollama`

### Errore: "ImportError: langchain-anthropic is not installed"

```bash
pip install langchain-anthropic
```

## Testing della Configurazione

```python
from src.ai_search.llm_factory import LLMFactory
from src.utils.config import Config

# Inizializzare
config = Config()
LLMFactory.initialize(config)

# Testare un agente
llm = LLMFactory.create_query_analyzer_llm()
response = llm.invoke("Ciao, come stai?")
print(response)
```

## Best Practices

1. **Usa provider diversi per agenti diversi** - Ottimizza costi/qualità per ogni task
2. **Configura temperature appropriata** - Bassa per analisi, alta per creatività
3. **Cache LLM instances** - La factory lo fa automaticamente
4. **Monitora i costi** - Traccia l'usage di token per ogni agente
5. **Testa offline** - Usa Ollama localmente in development
6. **Documenta le scelte** - Spiega perché hai scelto quel modello per quell'agente
