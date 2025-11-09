# 🌐 Visualizzazione 3D del Vector Database

## Funzionalità

L'applicazione Streamlit include una **visualizzazione 3D interattiva** degli embeddings che permette di:

### 🎨 Visualizzazione 3D Interattiva

- **Riduzione dimensionale**: Usa UMAP o t-SNE per ridurre gli embeddings da 1024D a 3D
- **Interattività**: 
  - Zoom e rotazione del grafico 3D
  - Hover sui punti per vedere metadata e testo
  - Colorazione per Video ID o Date
- **Personalizzazione**:
  - Scegli il numero di punti da visualizzare (100-2000)
  - Seleziona il metodo di riduzione (UMAP o t-SNE)
  - Colora i punti per Video ID, Date o nessun colore

### 📊 Cosa mostra la visualizzazione 3D

- **Punti vicini** = contenuti simili semanticamente
- **Cluster** = gruppi di video/chunks con argomenti simili
- **Distanze** = similarità semantica tra i testi

### 🚀 Come usare

1. Avvia l'app: `streamlit run streamlit_app.py`
2. Scorri fino alla sezione "🌐 3D Embedding Visualization"
3. Scegli:
   - Numero di punti (consigliato: 500-1000 per iniziare)
   - Metodo di riduzione (UMAP è più veloce, t-SNE più accurato)
   - Colore per categorizzare i punti
4. Clicca "🎨 Generate 3D Visualization"
5. Interagisci con il grafico:
   - **Rotazione**: Click e trascina
   - **Zoom**: Scroll del mouse
   - **Pan**: Click destro e trascina
   - **Hover**: Passa sopra i punti per vedere i dettagli

### 💡 Suggerimenti

- **UMAP** è più veloce e mantiene meglio le strutture globali
- **t-SNE** è più lento ma mostra meglio i cluster locali
- Usa 500-1000 punti per un buon equilibrio tra velocità e dettaglio
- Colora per Video ID per vedere se video simili si raggruppano insieme
- Colora per Date per vedere l'evoluzione temporale dei contenuti

### 📥 Export

Puoi scaricare le coordinate 3D in formato CSV per analisi ulteriori.

