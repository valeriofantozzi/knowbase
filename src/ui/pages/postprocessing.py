"""
PostProcessing Page Module

Implements vector database visualization and statistical analysis:
- Overview tab: Database statistics and distributions
- Visualization tab: 2D/3D embedding space visualization
- Clustering tab: HDBSCAN clustering with evaluation metrics
- Export tab: Data export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

from ..state import reset_clustering_state
from ..theme import ICONS, COLORS, CLUSTERING_PRESETS, format_number
from ..components.feedback import (
    show_empty_state,
    show_info_callout,
    show_success_summary,
)
from ..components.metric_card import metric_card, metric_row


def get_collection_data(limit: int = 1000) -> Dict[str, Any]:
    """
    Get sample data from ChromaDB collection.

    Args:
        limit: Maximum number of documents to retrieve

    Returns:
        Dictionary with collection data
    """
    if "collection" not in st.session_state or st.session_state.collection is None:
        return {"error": "No collection loaded"}

    try:
        sample = st.session_state.collection.get(
            limit=limit, include=["embeddings", "metadatas", "documents"]
        )
        return sample
    except Exception as e:
        return {"error": str(e)}


def analyze_metadata(metadatas: List[Dict]) -> Dict[str, Any]:
    """
    Analyze metadata for statistics.

    Args:
        metadatas: List of metadata dictionaries

    Returns:
        Analysis results
    """
    source_ids = []
    dates = []
    titles = []

    for meta in metadatas:
        if meta:
            if "source_id" in meta:
                source_ids.append(meta["source_id"])
            if "date" in meta:
                dates.append(meta["date"])
            if "title" in meta:
                titles.append(meta["title"])

    return {
        "source_ids": source_ids,
        "dates": dates,
        "titles": titles,
        "unique_sources": len(set(source_ids)),
        "date_counter": Counter(dates),
        "source_counter": Counter(source_ids),
    }


def render_overview_tab() -> None:
    """Render the Overview tab with database statistics."""
    total_docs = st.session_state.get("total_docs", 0)

    if total_docs == 0:
        show_empty_state(
            title="No Documents Loaded",
            message="Upload documents to get started with analysis.",
            icon="📭",
            action_label="Go to Load Documents",
            action_page="📥 Load Documents",
        )
        return

    # Get sample data for analysis
    sample_size = min(1000, total_docs)
    sample = get_collection_data(limit=sample_size)

    if "error" in sample:
        st.error(f"Error loading data: {sample['error']}")
        return

    # Analyze metadata
    analysis = analyze_metadata(sample.get("metadatas", []))

    # Key metrics row
    st.subheader(f"{ICONS['metrics']} Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Documents", format_number(total_docs), icon="📄")
    with col2:
        metric_card("Unique Sources", analysis["unique_sources"], icon="🎬")
    with col3:
        avg_chunks = (
            total_docs / analysis["unique_sources"]
            if analysis["unique_sources"] > 0
            else 0
        )
        metric_card("Avg Chunks/Source", f"{avg_chunks:.1f}", icon="📏")
    with col4:
        embeddings = sample.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            emb_dim = len(embeddings[0])
            metric_card("Embedding Dim", emb_dim, icon="🔢")

    st.markdown("---")

    # Date distribution chart
    if analysis["dates"]:
        st.subheader(f"{ICONS['calendar']} Date Distribution")

        date_df = pd.DataFrame(
            [
                {"Date": date, "Count": count}
                for date, count in analysis["date_counter"].most_common(20)
            ]
        )

        fig = px.bar(
            date_df,
            x="Date",
            y="Count",
            title="Chunks per Date (Top 20)",
            labels={"Count": "Number of Chunks", "Date": "Date"},
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Source distribution
    if analysis["source_ids"]:
        st.subheader(f"{ICONS['database']} Top Sources by Chunks")

        source_df = pd.DataFrame(
            [
                {"Source ID": sid, "Chunks": count}
                for sid, count in analysis["source_counter"].most_common(15)
            ]
        )

        fig = px.bar(
            source_df,
            y="Source ID",
            x="Chunks",
            orientation="h",
            title="Top 15 Sources by Chunk Count",
            color_discrete_sequence=[COLORS["info"]],
        )
        fig.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)


def render_visualization_tab() -> None:
    """Render the Visualization tab with 2D/3D plots."""
    total_docs = st.session_state.get("total_docs", 0)

    if total_docs < 10:
        st.warning(
            f"⚠️ Not enough documents for visualization. "
            f"Need at least 10 documents, but only {total_docs} found."
        )
        return

    # Controls
    st.subheader(f"{ICONS['settings']} Visualization Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        slider_min = min(100, total_docs)
        slider_max = min(2000, total_docs)
        slider_default = min(500, total_docs)
        num_points = st.slider(
            "Number of points",
            min_value=slider_min,
            max_value=slider_max,
            value=slider_default,
            step=100,
        )

    with col2:
        reduction_method = st.selectbox(
            "Algorithm",
            ["UMAP", "t-SNE", "PCA"],
            index=0,
            help="UMAP is faster, t-SNE gives better cluster separation",
        )

    with col3:
        dimensions = st.radio("Dimensions", ["2D", "3D"], horizontal=True)

    # Generate visualization button
    if st.button(
        f"{ICONS['chart']} Generate Visualization",
        type="primary",
        use_container_width=True,
    ):
        compute_visualization_data(
            num_points=num_points,
            reduction_method=reduction_method,
            dimensions=dimensions,
        )

    # Render plot if data is available
    if "viz_data" in st.session_state:
        render_visualization_plot()


def compute_visualization_data(
    num_points: int, 
    reduction_method: str, 
    dimensions: str
) -> None:
    """
    Compute visualization data (fetch + reduce) and store in session state.

    Args:
        num_points: Number of points
        reduction_method: UMAP/PCA/t-SNE
        dimensions: 2D or 3D
    """
    with st.spinner(f"Computing {dimensions} space with {reduction_method}..."):
        try:
             # Get data
            sample = st.session_state.collection.get(
                limit=num_points, include=["embeddings", "metadatas", "documents"]
            )

            embeddings = sample.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                st.error("No embeddings found in database")
                return

            embeddings_array = np.array(embeddings)
            metadatas = sample.get("metadatas", [{}] * len(embeddings))
            documents = sample.get("documents", [""] * len(embeddings))
            ids = sample.get("ids", [""] * len(embeddings))

            # Progress tracking
            progress = st.progress(0, text="Reducing dimensions...")

            # Dimensionality reduction
            n_components = 3 if dimensions == "3D" else 2

            if reduction_method == "UMAP":
                try:
                    import umap

                    reducer = umap.UMAP(
                        n_components=n_components,
                        random_state=42,
                        n_neighbors=15,
                        min_dist=0.1,
                    )
                    reduced = reducer.fit_transform(embeddings_array)
                except ImportError:
                    st.error("UMAP not installed. Install with: pip install umap-learn")
                    return
            elif reduction_method == "t-SNE":
                from sklearn.manifold import TSNE

                progress.progress(0.3, text="Running t-SNE (this may take a while)...")
                reducer = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=30,
                    max_iter=1000,
                )
                reduced = reducer.fit_transform(embeddings_array)
            else:  # PCA
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings_array)

            progress.progress(1.0, text="✅ Computation complete")
            
            # Store in session state
            st.session_state["viz_data"] = {
                "reduced": reduced,
                "ids": ids,
                "metadatas": metadatas,
                "documents": documents,
                "dimensions": dimensions,
                "method": reduction_method,
                "num_points": num_points,
                "original_dim": embeddings_array.shape[1]
            }
            st.rerun()

        except Exception as e:
            st.error(f"Error computing visualization: {e}")
            st.exception(e)


def render_visualization_plot() -> None:
    """Render the interactive visualization plot from stored data."""
    data = st.session_state["viz_data"]
    reduced = data["reduced"]
    ids = data["ids"]
    metadatas = data["metadatas"]
    documents = data["documents"]
    dimensions = data["dimensions"]
    
    # 1. Controls Row (Filter & Color)
    col1, col2 = st.columns([2, 1])
    
    filtered_indices = list(range(len(ids))) # Default: all
    
    with col1:
        # Filter by cluster (if available)
        cluster_results = st.session_state.get("cluster_results")
        cluster_labels = st.session_state.get("cluster_labels")
        cluster_chunk_ids = st.session_state.get("cluster_chunk_ids")
        
        # Create ID map
        id_to_label = {}
        if cluster_chunk_ids and cluster_labels is not None:
             id_to_label = dict(zip(cluster_chunk_ids, cluster_labels))

        if cluster_labels is not None:
            unique_labels = sorted(list(set(cluster_labels)))
            cluster_options = []
            cluster_map = {} # Display Name -> ID
            
            for lbl in unique_labels:
                if lbl == -1:
                    name = "Outlier"
                else:
                    name = f"Cluster {lbl}"
                    if cluster_results and cluster_results.cluster_names and lbl in cluster_results.cluster_names:
                        name = f"{cluster_results.cluster_names[lbl]} ({lbl})"
                
                cluster_options.append(name)
                cluster_map[name] = lbl
                
            selected_clusters = st.multiselect(
                "Filter by Cluster",
                options=cluster_options,
                default=[],
                help="Show points only from selected clusters"
            )
            
            if selected_clusters:
                target_labels = set([cluster_map[s] for s in selected_clusters])
                # Filter indices
                filtered_indices = []
                for i, cid in enumerate(ids):
                    lbl = id_to_label.get(cid)
                    if lbl in target_labels:
                        filtered_indices.append(i)
                        
                if not filtered_indices:
                     st.warning("No points found for selected clusters in the current sample.")

    with col2:
        color_option = st.selectbox(
            "Color by", ["Cluster", "Source ID", "Date", "None"], index=0
        )

    # 2. Prepare Data for Plotting (Filtered)
    if not filtered_indices:
        return

    plot_reduced = reduced[filtered_indices]
    plot_metadatas = [metadatas[i] for i in filtered_indices]
    plot_documents = [documents[i] for i in filtered_indices]
    plot_ids = [ids[i] for i in filtered_indices]
    
    # Prepare Colors
    colors = None
    if color_option == "Source ID":
        colors = [m.get("source_id", "unknown") if m else "unknown" for m in plot_metadatas]
    elif color_option == "Date":
        colors = [m.get("date", "unknown") if m else "unknown" for m in plot_metadatas]
    elif color_option == "Cluster":
        colors = []
        for cid in plot_ids:
            label = id_to_label.get(cid)
            if label is None:
                    colors.append("Unknown")
            elif label == -1:
                colors.append("Outlier")
            else:
                if cluster_results and cluster_results.cluster_names and label in cluster_results.cluster_names:
                        colors.append(f"{cluster_results.cluster_names[label]} ({label})")
                else:
                        colors.append(f"Cluster {label}")
    
    # Prepare Hover
    hover_texts = []
    for i, (meta, doc) in enumerate(zip(plot_metadatas, plot_documents)):
        parts = []
        if meta:
            if "title" in meta:
                parts.append(f"Title: {meta['title'][:50]}")
            if "source_id" in meta:
                parts.append(f"Source: {meta['source_id']}")
            if "date" in meta:
                parts.append(f"Date: {meta['date']}")
        
        cid = plot_ids[i]
        if cid in id_to_label:
            lbl = id_to_label[cid]
            c_name = "Outlier" if lbl == -1 else f"Cluster {lbl}"
            if lbl != -1 and cluster_results and cluster_results.cluster_names and lbl in cluster_results.cluster_names:
                 c_name = f"{cluster_results.cluster_names[lbl]} ({lbl})"
            parts.append(f"Cluster: {c_name}")
            
        parts.append(f"Text: {doc[:100]}...")
        hover_texts.append("<br>".join(parts))

    # 3. Render Plot
    if dimensions == "3D":
        fig = go.Figure(
            data=go.Scatter3d(
                x=plot_reduced[:, 0],
                y=plot_reduced[:, 1],
                z=plot_reduced[:, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    color=[hash(c) % 256 for c in colors] if colors else "blue",
                    colorscale="Viridis" if colors else None,
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_texts,
                hovertemplate="<b>%{text}</b><extra></extra>",
                name="Embeddings",
            )
        )
        fig.update_layout(
            title=f"3D Space ({data['method']}) - {len(plot_reduced)} points",
            scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
            height=700,
        )
    else:
        df = pd.DataFrame({
            "x": plot_reduced[:, 0],
            "y": plot_reduced[:, 1],
            "color": colors if colors else "All",
            "hover": hover_texts
        })
        fig = px.scatter(
            df, x="x", y="y", color="color" if colors else None,
            hover_data={"hover": True, "x": False, "y": False, "color": False},
            title=f"2D Space ({data['method']}) - {len(plot_reduced)} points",
        )
        fig.update_layout(height=600)
        fig.update_traces(marker=dict(size=6, opacity=0.7))

    st.plotly_chart(fig, use_container_width=True)
    
    # 4. Statistics
    st.subheader(f"{ICONS['chart']} Visualization Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Points visualized", len(plot_reduced))
    with col2:
        st.metric("Original Data Points", data["num_points"])
    with col3:
        st.metric("Reduced dimensions", 3 if dimensions == "3D" else 2)

    # 5. Download
    coords_df = pd.DataFrame({
        "ID": plot_ids,
        "X": plot_reduced[:, 0],
        "Y": plot_reduced[:, 1],
    })
    if dimensions == "3D":
        coords_df["Z"] = plot_reduced[:, 2]
    
    st.download_button(
        label=f"{ICONS['download']} Download Coordinates (CSV)",
        data=coords_df.to_csv(index=False),
        file_name=f"embedding_{dimensions.lower()}_coordinates.csv",
        mime="text/csv",
    )


def render_clustering_tab() -> None:
    """Render the Clustering tab with HDBSCAN analysis."""
    total_docs = st.session_state.get("total_docs", 0)

    if total_docs < 50:
        st.warning(
            f"⚠️ Need at least 50 documents for meaningful clustering. "
            f"Currently have {total_docs} documents."
        )
        return

    # Attempt to load existing results if not present
    if st.session_state.get("cluster_results") is None:
        try:
            from ...clustering.cluster_manager import ClusterManager
            
            # Use the active collection name
            collection_name = None
            if "collection" in st.session_state and st.session_state.collection:
                collection_name = st.session_state.collection.name
                
            manager = ClusterManager(collection_name=collection_name)
            # Try to load
            loaded_result = manager.load_clustering_results(limit=5000)
            if loaded_result:
                st.session_state["cluster_results"] = loaded_result
                st.session_state["cluster_labels"] = loaded_result.labels
                # Populate chunk_ids and metadatas to support UI interactions
                # Ideally, we should fetch these properly or have load_clustering_results return them.
                # Since we fetched a sample in load_clustering_results, let's minimally ensure
                # we don't crash. But for full functionality (Highlighting, Filtering), 
                # we rely on st.session_state.collection.get() which was likely done elsewhere 
                # OR we might see a mismatch. 
                # For now, let's assume the user doesn't need perfect sync of the *exact* sample unless they re-run.
                # However, the Visualization tab uses 'viz_data'.
                
                # Fetch data to populate support variables for the list view
                sample = st.session_state.collection.get(
                   limit=5000,
                   include=["metadatas", "documents", "embeddings"]
                )
                st.session_state["cluster_chunk_ids"] = sample["ids"]
                st.session_state["cluster_metadatas"] = sample["metadatas"]
                
                st.success("Loaded saved clustering results from database.")
        except Exception as e:
            st.warning(f"Could not load saved clustering results: {e}")

    # Clustering parameters
    st.subheader(f"{ICONS['settings']} Clustering Parameters")

    # Presets
    st.write("**Quick Presets:**")
    preset_cols = st.columns(3)
    for i, (name, params) in enumerate(CLUSTERING_PRESETS.items()):
        with preset_cols[i]:
            if st.button(name, help=params["description"], use_container_width=True):
                st.session_state["clustering_params"] = {
                    "min_cluster_size": params["min_cluster_size"],
                    "min_samples": params["min_samples"],
                    "metric": "cosine",
                }
                st.rerun()

    st.markdown("---")

    # Manual parameters
    col1, col2, col3 = st.columns(3)

    clustering_params = st.session_state.get(
        "clustering_params",
        {"min_cluster_size": 15, "min_samples": 5, "metric": "cosine"},
    )

    with col1:
        min_cluster_size = st.slider(
            "min_cluster_size",
            min_value=5,
            max_value=100,
            value=clustering_params.get("min_cluster_size", 15),
            help="Minimum number of points to form a cluster",
        )

    with col2:
        min_samples = st.slider(
            "min_samples",
            min_value=1,
            max_value=50,
            value=clustering_params.get("min_samples", 5),
            help="Number of samples in neighborhood for core point",
        )

    with col3:
        metric = st.selectbox(
            "Distance metric",
            ["cosine", "euclidean"],
            index=0 if clustering_params.get("metric", "cosine") == "cosine" else 1,
        )

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        run_clustering = st.button(
            f"{ICONS['cluster']} Run Clustering",
            type="primary",
            use_container_width=True,
        )
    with col2:
        save_clustering = st.button(
            f"{ICONS['download']} Save to DB",
            use_container_width=True,
            disabled=st.session_state.get("cluster_results") is None,
        )

    # Run clustering
    if run_clustering:
        perform_clustering(min_cluster_size, min_samples, metric)
        
    # Save clustering
    if save_clustering:
        save_clustering_to_db()

    # Display results if available
    if st.session_state.get("cluster_results") is not None:
        render_clustering_results()


def perform_clustering(min_cluster_size: int, min_samples: int, metric: str) -> None:
    """
    Perform clustering using the pipeline.

    Args:
        min_cluster_size: Minimum cluster size
        min_samples: Min samples for core point
        metric: Distance metric
    """
    with st.spinner("Running Clustering Pipeline (Reduction + HDBSCAN + Topic Extraction)..."):
        try:
            from ...clustering.pipeline import ClusteringPipeline
            from ...clustering.cluster_evaluator import ClusterEvaluator

            # Get embeddings and documents
            sample = st.session_state.collection.get(
                limit=min(5000, st.session_state.total_docs),
                include=["embeddings", "metadatas", "documents"],
            )

            embeddings = np.array(sample["embeddings"])
            documents = sample.get("documents", [])

            # Run Pipeline
            pipeline = ClusteringPipeline(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                use_reduction=True, # Always use reduction for better results
                reduction_components=10,
                extract_topics=True
            )

            result = pipeline.fit_predict(embeddings, documents)

            # Store results
            st.session_state["cluster_results"] = result
            st.session_state["cluster_labels"] = result.labels
            st.session_state["cluster_metadatas"] = sample["metadatas"]
            st.session_state["cluster_chunk_ids"] = sample["ids"]

            # Evaluate clustering
            evaluator = ClusterEvaluator()
            metrics = evaluator.evaluate(embeddings, result.labels, metric)
            st.session_state["cluster_metrics"] = metrics

            # Update params
            st.session_state["clustering_params"] = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "metric": metric,
            }

            st.success(f"✅ Clustering complete! Found {len(result.topics)} topics.")
            st.rerun()

        except Exception as e:
            st.error(f"Clustering failed: {e}")
            st.exception(e)

def save_clustering_to_db() -> None:
    """Save clustering results to ChromaDB."""
    with st.spinner("Saving clustering results to database..."):
        try:
            from ...clustering.cluster_manager import ClusterManager
            
            cluster_manager = ClusterManager(
                collection_name=st.session_state.collection.name
            )
            
            chunk_ids = st.session_state.get("cluster_chunk_ids", [])
            result = st.session_state.get("cluster_results")
            
            if not chunk_ids or not result:
                st.error("No clustering results to save.")
                return
                
            count = cluster_manager.store_clustering_results(chunk_ids, result)
            
            st.success(f"✅ Successfully saved results for {count} documents!")
            
        except Exception as e:
            st.error(f"Failed to save results: {e}")
            st.exception(e)


def render_clustering_results() -> None:
    """Render clustering results."""
    result = st.session_state.get("cluster_results")
    labels = st.session_state.get("cluster_labels")
    metrics = st.session_state.get("cluster_metrics")
    metadatas = st.session_state.get("cluster_metadatas", [])

    if labels is None:
        return

    st.markdown("---")
    st.subheader(f"{ICONS['success']} Clustering Results")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metric_card("Clusters", n_clusters, icon="🔬")
    with col2:
        n_outliers = int(np.sum(labels == -1))
        metric_card("Outliers", n_outliers, icon="📍")
    with col3:
        if metrics:
            metric_card("Silhouette", f"{metrics.silhouette_score:.3f}", icon="📊")
    with col4:
        if metrics:
            metric_card(
                "Davies-Bouldin", f"{metrics.davies_bouldin_index:.3f}", icon="📈"
            )

    # Cluster list details
    st.subheader("📋 Cluster Details")

    cluster_sizes = Counter(labels)
    # Remove outliers from count display
    if -1 in cluster_sizes:
        del cluster_sizes[-1]
    
    cluster_data_list = []
    unique_labels = sorted(list(set(labels)))
    
    for label in unique_labels:
        if label == -1: 
            continue
            
        name = f"Cluster {label}"
        keywords = []
        if result and result.cluster_names and label in result.cluster_names:
            name = result.cluster_names[label]
        
        if result and result.topics and label in result.topics:
            keywords = [k[0] for k in result.topics[label][:5]]
            
        cluster_data_list.append({
            "ID": label,
            "Name": name,
            "Size": cluster_sizes[label],
            "Top Keywords": ", ".join(keywords)
        })
        
    if cluster_data_list:
        st.dataframe(
            pd.DataFrame(cluster_data_list),
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Name": st.column_config.TextColumn("Topic Name", width="medium"),
                "Size": st.column_config.ProgressColumn("Size", format="%d", min_value=0, max_value=max(cluster_sizes.values())),
                "Top Keywords": st.column_config.TextColumn("Keywords", width="large"),
            },
            hide_index=True,
            use_container_width=True
        )

    # Cluster size distribution (Visual)
    st.subheader("📊 Cluster Distribution")
    if cluster_sizes:
        size_df = pd.DataFrame(
            [
                {"Cluster": result.cluster_names.get(k, f"Cluster {k}") if result and result.cluster_names else f"Cluster {k}" , "Size": v}
                for k, v in sorted(cluster_sizes.items(), key=lambda x: -x[1])
            ]
        )

        fig = px.bar(
            size_df,
            x="Cluster",
            y="Size",
            title="Documents per Cluster",
            color_discrete_sequence=[COLORS["info"]],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cluster exploration
    with st.expander(f"{ICONS['search']} Explore Cluster Contents", expanded=False):
        cluster_ids = sorted(set(labels))
        # Use semantic names for selection box
        cluster_options = []
        for i in cluster_ids:
            if i >= 0:
                name = f"Cluster {i}"
                if result and result.cluster_names and i in result.cluster_names:
                    name = f"{result.cluster_names[i]} ({i})"
                cluster_options.append(name)
        
        if cluster_options:
            selected_option = st.selectbox("Select Cluster", options=cluster_options)

            # Parse back ID from "Name (ID)" or "Cluster ID"
            if "(" in selected_option and selected_option.endswith(")"):
                cluster_idx = int(selected_option.split("(")[-1].strip(")"))
            else:
                cluster_idx = int(selected_option.split()[-1])
            
            # Show Keywords
            if result and result.topics and cluster_idx in result.topics:
                keywords = [k[0] for k in result.topics[cluster_idx][:10]]
                st.info(f"**Keywords:** {', '.join(keywords)}")

            cluster_mask = labels == cluster_idx
            cluster_docs_idx = np.where(cluster_mask)[0][:10]  # First 10

            st.write(f"**Sample documents from {selected_option}:**")
            for idx in cluster_docs_idx:
                if idx < len(metadatas):
                    meta = metadatas[idx]
                    title = meta.get("title", "Untitled") if meta else "Untitled"
                    st.markdown(f"- {title[:80]}...")


def render_export_tab() -> None:
    """Render the Export tab."""
    st.subheader(f"{ICONS['download']} Export Data")

    total_docs = st.session_state.get("total_docs", 0)

    if total_docs == 0:
        st.info("No data to export. Load documents first.")
        return

    # Source list export
    st.write("**📹 Source List**")
    if st.button("Generate Source List"):
        with st.spinner("Generating source list..."):
            all_data = st.session_state.collection.get()

            sources = {}
            for meta in all_data.get("metadatas", []):
                if meta and "source_id" in meta:
                    source_id = meta["source_id"]
                    if source_id not in sources:
                        sources[source_id] = {
                            "title": meta.get("title", "N/A"),
                            "date": meta.get("date", "N/A"),
                            "chunks": 0,
                        }
                    sources[source_id]["chunks"] += 1

            source_df = pd.DataFrame(
                [
                    {
                        "Source ID": sid,
                        "Title": info["title"],
                        "Date": info["date"],
                        "Chunks": info["chunks"],
                    }
                    for sid, info in sorted(
                        sources.items(), key=lambda x: -x[1]["chunks"]
                    )
                ]
            )

            st.dataframe(source_df, use_container_width=True, height=400)

            st.download_button(
                label=f"{ICONS['download']} Download Source List (CSV)",
                data=source_df.to_csv(index=False),
                file_name="source_list.csv",
                mime="text/csv",
            )

    st.markdown("---")

    # Clustering results export
    if st.session_state.get("cluster_labels") is not None:
        st.write("**🔬 Clustering Results**")

        labels = st.session_state["cluster_labels"]
        cluster_df = pd.DataFrame({"Index": range(len(labels)), "Cluster": labels})

        st.download_button(
            label=f"{ICONS['download']} Download Cluster Labels (CSV)",
            data=cluster_df.to_csv(index=False),
            file_name="cluster_labels.csv",
            mime="text/csv",
        )


def render_postprocessing_page() -> None:
    """
    Render the PostProcessing page.

    This is the main entry point for the page module.
    """
    # Initialize total_docs from collection if not set
    if (
        "total_docs" not in st.session_state
        or st.session_state.get("total_docs", 0) == 0
    ):
        if "collection" in st.session_state and st.session_state.collection is not None:
            try:
                st.session_state.total_docs = st.session_state.collection.count()
            except Exception:
                st.session_state.total_docs = 0

    st.header(f"{ICONS['analysis']} PostProcessing")
    st.markdown("Analyze and visualize your embedding space.")

    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            f"{ICONS['chart']} Overview",
            "🌐 Visualization",
            f"{ICONS['cluster']} Clustering",
            f"{ICONS['download']} Export",
        ]
    )

    with tab1:
        render_overview_tab()

    with tab2:
        render_visualization_tab()

    with tab3:
        render_clustering_tab()

    with tab4:
        render_export_tab()
