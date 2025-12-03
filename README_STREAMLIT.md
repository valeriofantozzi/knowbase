# Vector Database Web Viewer

Interfaccia web interattiva per visualizzare e esplorare il vector database con supporto multi-modello per gli embedding.

## Installazione

Le dipendenze sono già installate se hai eseguito `pip install -r requirements.txt`.

Se necessario:
```bash
source .venv/bin/activate
pip install streamlit plotly
```

## Avvio

```bash
./start_viewer.sh
```

oppure manualmente:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8601
```

L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8601`

## Funzionalità

- **Selezione Modello**: Scegli tra diversi modelli di embedding (BGE, EmbeddingGemma) dalla sidebar
- **Isolamento Collection**: Ogni modello utilizza collection separate per evitare conflitti
- **Statistiche per Modello**: Visualizza statistiche specifiche del modello selezionato
- **Grafici**: Distribuzione dei chunks per data con grafici interattivi
- **Ricerca semantica**: Cerca nel database usando query in linguaggio naturale con il modello selezionato
- **Browser documenti**: Esplora i documenti con filtri per video ID e data
- **Lista video**: Visualizza tutti i video nel database con conteggio chunks
- **Switching Dinamico**: Cambia modello al volo senza riavviare l'applicazione

## Uso

1. Avvia l'applicazione con `./start_viewer.sh` o `streamlit run streamlit_app.py --server.port 8601`
2. **Seleziona il modello** dalla barra laterale (BGE o EmbeddingGemma)
3. Usa la barra laterale per navigare tra le diverse sezioni
4. Inserisci una query nella sezione "Semantic Search" e clicca "Search"
5. Esplora i documenti usando i filtri nella sezione "Document Browser"
6. Visualizza la lista completa dei video nella sezione "Video List"
7. **Cambia modello** in qualsiasi momento - l'applicazione passerà automaticamente alla collection corrispondente

## Modelli Supportati

### BAAI/bge-large-en-v1.5
- **Dimensioni**: 1024
- **Lunghezza massima**: 512 token
- **Ideale per**: Ricerca generale di alta qualità

### Google/embeddinggemma-300m
- **Dimensioni**: 768 (con supporto MRL per dimensioni flessibili)
- **Lunghezza massima**: 2048 token
- **Ideale per**: Contenuti lunghi e ricerca contestuale

## Note Importanti

- **Collection Separate**: Ogni modello utilizza la propria collection nel database
- **Ricerca Isolati**: I risultati di ricerca sono specifici del modello selezionato
- **Switching**: Il cambio di modello è immediato ma richiede dati precedentemente processati con quel modello
- **Performance**: EmbeddingGemma può essere più lento ma gestisce meglio testi lunghi

