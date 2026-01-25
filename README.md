# RAG Evaluation System

A comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) systems using a custom golden dataset and a FAISS vector database. This project evaluates how well AI can answer different questions using different retrieval strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG EVALUATION SYSTEM                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│  Golden Dataset  │     │  Article URLs    │     │    FAISS Vector DB       │
│  (100 Q&A pairs) │────▶│  (Source Links)  │────▶│  (Scraped Article Text)  │
│                  │     │                  │     │                          │
│  • Questions     │     │  Scraping via    │     │  • Dense Embeddings      │
│  • Ground Truth  │     │  httpx + Claude  │     │  • BM25 Index            │
│  • Source IDs    │     │  text extraction │     │  • Full Article Content  │
└──────────────────┘     └──────────────────┘     └──────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RETRIEVAL LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Dense Search   │  │  Sparse Search  │  │      Hybrid Search          │  │
│  │  (Embeddings)   │  │  (BM25)         │  │  (0.7 Dense + 0.3 Sparse)   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GENERATION LAYER                                   │
│                                                                              │
│    Retrieved Context (Top-K=3) ───▶ Claude API  ───▶ Generated Answer        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVALUATION LAYER                                   │
│                                                                              │
│    ┌────────────┐  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│    │  Hit@K     │  │  Recall@K  │  │ Token Overlap│  │Factual Consistency│  │
│    │ (Retrieval)│  │ (Retrieval)│  │  (Answer)    │  │     (Custom)      │  │
│    └────────────┘  └────────────┘  └──────────────┘  └───────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT DASHBOARD                                  │
│                                                                              │
│    📈 Experiment Results  │  🔍 Interactive Query  │  📋 Detailed Results    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Golden Dataset

The golden dataset contains **100 question-answer pairs** sourced from real news articles and publications. Questions are categorized by type and difficulty level.

### Question Types

| Type | Count | Description |
|------|-------|-------------|
| **Factual** | 61 | Questions with specific, verifiable answers (names, numbers, dates, facts) |
| **Conceptual** | 32 | Questions requiring understanding of ideas, theories, or relationships |
| **Procedural** | 7 | Questions about processes, methods, or how things work |

#### Factual Questions
Direct questions seeking specific information that can be verified against the source.

**Example:**
> "How many countries will have immigrant visa processing suspended according to the State Department announcement?"
>
> **Answer:** "The State Department announced it will suspend immigrant visa processing for nationals of 75 countries..."

#### Conceptual Questions
Questions requiring deeper understanding and explanation of ideas or relationships.

**Example:**
> "What is the 'barbell economy' concept and how does it relate to economic recession predictions?"
>
> **Answer:** "The barbell economy describes an economic structure where wealth is concentrated at the two extremes - the wealthy and the poor - while the middle class shrinks..."

#### Procedural Questions
Questions about processes, mechanisms, or step-by-step methods.

**Example:**
> "How does electron beam water treatment (EBWT) work to remediate PFAS contamination?"
>
> **Answer:** "Electron beam water treatment uses a compact, high-average-power superconducting radio-frequency accelerator to generate electron beams that are directed at contaminated water..."

### Difficulty Distribution

| Difficulty | Count |
|------------|-------|
| Easy | 39 |
| Medium | 35 |
| Hard | 26 |

### Dataset Entry Fields

Each entry in `golden_dataset.jsonl` contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier for the question |
| `difficulty` | string | Difficulty level: "easy", "medium", or "hard" |
| `question_type` | string | Category: "factual", "conceptual", or "procedural" |
| `question` | string | The evaluation question |
| `answer` | string | Ground truth answer used for evaluation |
| `source_articles` | array | List of source article references |
| `source_articles[].article_id` | string | URL of the source article |
| `source_articles[].title` | string | Title of the source article |
| `source_articles[].source` | string | Publication name |

## Vector Database Setup

The system uses FAISS (Facebook AI Similarity Search) for efficient vector storage and retrieval.

### Ingestion Pipeline

1. **URL Extraction**: Article URLs are extracted from the golden dataset's `source_articles` field

2. **Web Scraping**: Articles are fetched using `httpx` with proper headers and rate limiting

3. **Text Extraction**: Claude is used as an intelligent agent to extract clean article text from HTML, removing navigation, ads, and sidebars. Falls back to BeautifulSoup if Claude extraction fails.

4. **Embedding Generation**: Article text is converted to dense vectors using the `all-MiniLM-L6-v2` sentence transformer model (384 dimensions)

5. **Index Creation**:
   - **FAISS Index**: Dense embeddings stored in a flat L2 index
   - **BM25 Index**: Tokenized text stored for sparse keyword search
   - **Metadata**: Full article text, title, source, URL, and summary stored in pickle file

### Storage Files

| File | Description |
|------|-------------|
| `data/faiss_index` | FAISS vector index (binary) |
| `data/article_metadata.pkl` | Article metadata including full text |

### Running Ingestion

```bash
python scripts/ingest_articles_to_faiss.py
```

## Evaluation Metrics

The system evaluates RAG performance using 4 key metrics: 3 standard metrics and 1 custom metric.

### Standard Retrieval Metrics

#### 1. Hit@K
**Purpose:** Measures if at least one relevant document was retrieved in the top-K results.

**Formula:**
```
Hit@K = 1 if |Retrieved_K ∩ Relevant| > 0 else 0
```

**Interpretation:** Binary metric (0 or 1). A score of 1 means the retriever successfully found at least one relevant document.

#### 2. Recall@K
**Purpose:** Measures what fraction of all relevant documents were retrieved in the top-K results.

**Formula:**
```
Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|
```

**Interpretation:** Ranges from 0 to 1. Higher values indicate better coverage of relevant documents.

#### 3. Token Overlap (F1)
**Purpose:** Measures lexical similarity between generated answer and ground truth using token-level precision, recall, and F1.

**Formula:**
```
Precision = |Generated_tokens ∩ Truth_tokens| / |Generated_tokens|
Recall = |Generated_tokens ∩ Truth_tokens| / |Truth_tokens|
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation:** Ranges from 0 to 1. Higher values indicate more word overlap with ground truth.

### Custom Metric

#### 4. Factual Consistency (Anti-Hallucination Score)
**Purpose:** Measures how well the generated answer is grounded in the retrieved context and aligned with ground truth, helping detect hallucinations.

**Implementation:**
1. Tokenize generated answer, retrieved context, and ground truth
2. Remove common stopwords to focus on content words
3. Calculate what fraction of generated answer tokens appear in either context or ground truth

**Formula:**
```
Factual_Consistency = |Generated_content ∩ (Context_content ∪ Truth_content)| / |Generated_content|
```

**Interpretation:** Ranges from 0 to 1. Higher values indicate the answer is well-grounded in sources rather than hallucinated.

**Why This Metric:**
- Standard metrics like BLEU/ROUGE focus on n-gram overlap with ground truth only
- This metric also considers the retrieved context, rewarding answers that use information from retrieved documents
- Helps identify when the model generates plausible but unsupported information

## Running the System

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1: Build Vector Database

```bash
python scripts/ingest_articles_to_faiss.py
```

### Step 2: Run Experiments

```bash
cd src
python run_experiments.py
```

### Step 3: Launch Dashboard

```bash
streamlit run ui/app.py
```

## Project Structure

```
eval_system/
├── data/
│   ├── golden_dataset.jsonl    # 100 Q&A pairs with source articles
│   ├── faiss_index             # FAISS vector index (generated)
│   ├── article_metadata.pkl    # Article metadata (generated)
│   └── results/                # Experiment results (generated)
├── scripts/
│   ├── ingest_articles_to_faiss.py  # Article scraping & indexing
│   └── query_faiss.py               # CLI query tool
├── src/
│   ├── config.py                    # Configuration settings
│   ├── faiss_retriever.py           # FAISS retrieval module
│   ├── metrics.py                   # Evaluation metrics
│   ├── hybrid_retrieval_experiment.py
│   ├── context_window_experiment.py
│   └── run_experiments.py           # Main experiment runner
├── ui/
│   └── app.py                       # Streamlit dashboard
├── .env                             # API keys (not in git)
├── .gitignore
├── requirements.txt
└── README.md
```

## Configuration

Key settings in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer for embeddings |
| `LLM_MODEL` | claude-sonnet-4-20250514 | Claude model for generation |
| `DEFAULT_TOP_K` | 3 | Number of documents to retrieve |
| `SPARSE_WEIGHT` | 0.3 | BM25 weight in hybrid search |
| `DENSE_WEIGHT` | 0.7 | Embedding weight in hybrid search |
