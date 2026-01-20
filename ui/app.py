"""
RAG Evaluation Dashboard

Streamlit app to display experiment results and provide an interactive query interface.
Uses FAISS vector database for document retrieval.
"""
import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import anthropic

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    RESULTS_DIR,
    LLM_MODEL,
    ANTHROPIC_API_KEY,
    FAISS_INDEX_PATH
)
from faiss_retriever import FAISSRetriever

# Page configuration
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .search-result {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .dense-result {
        border-left-color: #2ca02c;
    }
    .sparse-result {
        border-left-color: #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)


def load_experiment_results():
    """Load experiment results from JSON files."""
    results = {}

    hybrid_path = RESULTS_DIR / "hybrid_retrieval_results.json"
    context_path = RESULTS_DIR / "context_window_results.json"

    if hybrid_path.exists():
        with open(hybrid_path, 'r') as f:
            results['hybrid_retrieval'] = json.load(f)

    if context_path.exists():
        with open(context_path, 'r') as f:
            results['context_window'] = json.load(f)

    return results


@st.cache_resource
def load_faiss_retriever():
    """Load and cache the FAISS retriever for interactive queries."""
    if not FAISS_INDEX_PATH.exists():
        return None

    retriever = FAISSRetriever()
    if retriever.load():
        return retriever
    return None


def generate_answer_with_context(client, question, context):
    """Generate answer using Claude."""
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": f"""Based on the following context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""}],
            system="You are a helpful assistant. Answer questions based on the provided context."
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def main():
    st.title("📊 RAG Evaluation Dashboard")
    st.markdown("---")

    # Load results
    results = load_experiment_results()

    if not results:
        st.warning("⚠️ No experiment results found. Please run experiments first:")
        st.code("cd src && python run_experiments.py", language="bash")
        st.info("After running experiments, refresh this page to see results.")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Experiment Results", "🔍 Interactive Query", "📋 Detailed Results"])

    # Tab 1: Experiment Results
    with tab1:
        if not results:
            st.info("Run experiments to see results here.")
        else:
            st.subheader("Experiment Comparison")

            # Define the 4 key metrics we want to show
            key_metrics = ['hit@3', 'recall@3', 'token_overlap', 'factual_consistency']
            metric_labels = {
                'hit@3': 'Hit@3',
                'recall@3': 'Recall@3',
                'token_overlap': 'Token Overlap',
                'factual_consistency': 'Factual Consistency'
            }

            # Extract metrics from both experiments
            table_data = []
            chart_data = []

            # Hybrid Retrieval experiment
            if 'hybrid_retrieval' in results:
                hybrid_metrics = results['hybrid_retrieval'].get('overall_metrics', {})
                for metric in key_metrics:
                    value = hybrid_metrics.get(metric, {}).get('mean', 0)
                    table_data.append({
                        'Experiment': 'Hybrid Retrieval',
                        'Metric': metric_labels.get(metric, metric),
                        'Value': f"{value:.4f}"
                    })
                    chart_data.append({
                        'Experiment': 'Hybrid Retrieval',
                        'Metric': metric_labels.get(metric, metric),
                        'Value': value
                    })

            # Context Window experiment (use best performing chunk count or default)
            if 'context_window' in results:
                results_by_chunk = results['context_window'].get('results_by_chunk_count', {})
                if results_by_chunk:
                    # Use the first chunk configuration's overall metrics
                    # (or you could pick a specific one like chunks_5)
                    first_chunk_key = list(results_by_chunk.keys())[0]
                    context_metrics = results_by_chunk[first_chunk_key].get('overall_metrics', {})

                    for metric in key_metrics:
                        value = context_metrics.get(metric, {}).get('mean', 0)
                        table_data.append({
                            'Experiment': 'Context Window',
                            'Metric': metric_labels.get(metric, metric),
                            'Value': f"{value:.4f}"
                        })
                        chart_data.append({
                            'Experiment': 'Context Window',
                            'Metric': metric_labels.get(metric, metric),
                            'Value': value
                        })

            # Display metrics table
            if table_data:
                st.markdown("### Metrics Summary")
                df_table = pd.DataFrame(table_data)
                # Pivot to show experiments as columns
                df_pivot = df_table.pivot(index='Metric', columns='Experiment', values='Value').reset_index()
                st.dataframe(df_pivot, use_container_width=True, hide_index=True)

            # Display bar chart with fixed y-axis (0 to 1)
            if chart_data:
                st.markdown("### Metrics Comparison")
                df_chart = pd.DataFrame(chart_data)

                fig = px.bar(
                    df_chart,
                    x='Metric',
                    y='Value',
                    color='Experiment',
                    barmode='group',
                    title='Experiment Metrics Comparison'
                )

                # Fix y-axis range from 0 to 1
                fig.update_layout(
                    yaxis=dict(
                        range=[0, 1],
                        title='Score'
                    ),
                    xaxis=dict(
                        title='Metric'
                    ),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Interactive Query
    with tab2:
        st.subheader("🔍 Interactive Query Interface")
        st.markdown("Search the FAISS vector database and generate answers with Claude.")

        # Load FAISS retriever
        retriever = load_faiss_retriever()

        if retriever is None:
            st.warning("⚠️ FAISS index not found. Please run the ingestion script first:")
            st.code("python scripts/ingest_articles_to_faiss.py", language="bash")
            st.info("This will scrape articles and build the vector database.")
        else:
            st.success(f"✅ FAISS index loaded with {retriever.index.ntotal} documents")

            # Query input
            query = st.text_area(
                "Enter your question:",
                placeholder="e.g., What is the Federal Reserve's balance sheet size?",
                height=100
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                top_k = st.slider("Number of results:", 1, 10, 5)

            if st.button("🔍 Search", type="primary"):
                if query:
                    with st.spinner("Searching FAISS vector database..."):
                        # Get results from all three methods
                        dense_results = retriever.dense_search(query, top_k)
                        sparse_results = retriever.sparse_search(query, top_k)
                        hybrid_results = retriever.hybrid_search(query, top_k)

                    # Display results in columns
                    col_dense, col_sparse = st.columns(2)

                    with col_dense:
                        st.markdown("### 🟢 Dense Search Results")
                        st.caption("Semantic similarity using embeddings")
                        for idx, score in dense_results:
                            doc = retriever.get_document(idx)
                            if doc:
                                title = doc.get('title', 'No title')[:50]
                                with st.expander(f"Score: {score:.4f} | {title}..."):
                                    st.markdown(f"**Title:** {doc.get('title', 'N/A')}")
                                    st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                                    st.markdown(f"**URL:** {doc.get('url', 'N/A')}")
                                    st.markdown(f"**Summary:** {doc.get('summary', 'N/A')}")
                                    text_preview = doc.get('text_preview', doc.get('full_text', 'N/A')[:300])
                                    st.markdown(f"**Content Preview:** {text_preview}")

                    with col_sparse:
                        st.markdown("### 🟠 Sparse Search Results")
                        st.caption("Keyword matching using BM25")
                        for idx, score in sparse_results:
                            doc = retriever.get_document(idx)
                            if doc:
                                title = doc.get('title', 'No title')[:50]
                                with st.expander(f"Score: {score:.4f} | {title}..."):
                                    st.markdown(f"**Title:** {doc.get('title', 'N/A')}")
                                    st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                                    st.markdown(f"**URL:** {doc.get('url', 'N/A')}")
                                    st.markdown(f"**Summary:** {doc.get('summary', 'N/A')}")
                                    text_preview = doc.get('text_preview', doc.get('full_text', 'N/A')[:300])
                                    st.markdown(f"**Content Preview:** {text_preview}")

                    st.markdown("---")
                    st.markdown("### 🔀 Hybrid Search Results")
                    st.caption("Combined dense and sparse search (used for answer generation)")

                    contexts = []
                    for idx, score, meta in hybrid_results:
                        doc = retriever.get_document(idx)
                        if doc:
                            title = doc.get('title', 'No title')[:50]
                            d_score = meta.get('dense_score', 0)
                            s_score = meta.get('sparse_score', 0)
                            with st.expander(f"Combined: {score:.4f} (Dense: {d_score:.4f}, Sparse: {s_score:.4f}) | {title}..."):
                                st.markdown(f"**Title:** {doc.get('title', 'N/A')}")
                                st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                                st.markdown(f"**URL:** {doc.get('url', 'N/A')}")
                                st.markdown(f"**Summary:** {doc.get('summary', 'N/A')}")
                                full_text = doc.get('full_text', doc.get('text_preview', 'N/A'))
                                st.markdown(f"**Full Content:** {full_text[:1000]}{'...' if len(full_text) > 1000 else ''}")

                            # Collect context for answer generation
                            contexts.append(doc.get('full_text', doc.get('text_preview', '')))

                    # Generate answer with context
                    st.markdown("---")
                    st.markdown("### 🤖 Generated Answer")

                    if ANTHROPIC_API_KEY:
                        # Build context from hybrid results
                        combined_context = "\n\n---\n\n".join(contexts)

                        with st.spinner("Generating answer with Claude..."):
                            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                            answer = generate_answer_with_context(client, query, combined_context)

                        st.success(answer)
                    else:
                        st.warning("⚠️ ANTHROPIC_API_KEY not set. Set the environment variable to enable answer generation.")
                        st.info("Using top retrieved context as answer:")
                        if contexts:
                            st.write(contexts[0][:500] + "..." if len(contexts[0]) > 500 else contexts[0])
                else:
                    st.warning("Please enter a query.")

    # Tab 3: Detailed Results
    with tab3:
        st.subheader("📋 Detailed Question-Level Results")

        if not results:
            st.info("Run experiments to see detailed results here.")
        else:
            # Experiment selector
            exp_for_details = st.selectbox(
                "Select Experiment for Details",
                list(results.keys()),
                format_func=lambda x: x.replace('_', ' ').title(),
                key="details_experiment"
            )

            exp_data = results[exp_for_details]

            # Get detailed results
            if exp_for_details == 'hybrid_retrieval':
                detailed = exp_data.get('detailed_results', [])
            else:
                # For context window, let user select chunk count
                results_by_chunk = exp_data.get('results_by_chunk_count', {})
                if results_by_chunk:
                    chunk_key = st.selectbox(
                        "Select chunk count:",
                        list(results_by_chunk.keys()),
                        key="chunk_detail_select"
                    )
                    detailed = results_by_chunk[chunk_key].get('detailed_results', [])
                else:
                    detailed = []

            if detailed:
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    diff_filter = st.multiselect(
                        "Filter by Difficulty",
                        ['easy', 'medium', 'hard'],
                        default=['easy', 'medium', 'hard'],
                        key="detail_diff"
                    )
                with col2:
                    type_filter_detail = st.multiselect(
                        "Filter by Type",
                        ['factual', 'conceptual', 'procedural'],
                        default=['factual', 'conceptual', 'procedural'],
                        key="detail_type"
                    )

                # Filter results
                filtered = [
                    r for r in detailed
                    if r.get('difficulty') in diff_filter and r.get('question_type') in type_filter_detail
                ]

                st.caption(f"Showing {len(filtered)} of {len(detailed)} results")

                # Display results
                for result in filtered[:50]:
                    metrics = result.get('metrics', {})
                    with st.expander(
                        f"Q{result['id']}: {result['question'][:60]}... | "
                        f"F1: {metrics.get('token_f1', 0):.2f} | "
                        f"Hit@1: {metrics.get('hit@1', 0):.0f}"
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Question:**")
                            st.write(result['question'])
                            st.markdown("**Ground Truth:**")
                            st.write(result['ground_truth'])

                        with col2:
                            st.markdown("**Generated Answer:**")
                            st.write(result.get('generated_answer', 'N/A'))

                        st.markdown("**Metrics:**")
                        metric_cols = st.columns(4)
                        metric_cols[0].metric("Token F1", f"{metrics.get('token_f1', 0):.4f}")
                        metric_cols[1].metric("Hit@1", f"{metrics.get('hit@1', 0):.0f}")
                        metric_cols[2].metric("Recall@3", f"{metrics.get('recall@3', 0):.4f}")
                        metric_cols[3].metric("Factual Consistency", f"{metrics.get('factual_consistency', 0):.4f}")

                        st.caption(f"Difficulty: {result.get('difficulty')} | Type: {result.get('question_type')}")


if __name__ == "__main__":
    main()
