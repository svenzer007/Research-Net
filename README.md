# Research-Net: Community Detection in Academic Papers

A scalable big-data framework for uncovering thematic communities in large-scale scholarly graphs. By combining distributed graph construction, semantic‐embedding–based similarity, and weighted PageRank, ResearchGraph detects cohesive groups of research papers and highlights influential works across millions of records.

---

## Features

* **Distributed Ingestion & ETL**

  * Fetch paper metadata and citation relations via the Semantic Scholar API
  * Keyword-based, paper-ID–based, and publication-year–based query workflows
  * Schema enforcement, deduplication, and Parquet-based storage for fault tolerance

* **Graph Construction**

  * **Vertices**: Each paper node carries `(id, title, year, venue, embedding vector)`
  * **Edges**:

    * **Citation edges** (directed)
    * **Similarity edges** (bidirectional) weighted by cosine similarity of embeddings

* **Weighted PageRank**

  * Custom Spark DataFrame–based PageRank implementation supporting edge weights
  * Configurable reset probability (`α`) and max iterations

* **Interactive Visualization**

  * PyVis-powered HTML network graphs
  * Node size ∝ PageRank score; edge thickness ∝ semantic similarity
  * Pan, zoom, and hover to explore citation trails and topical clusters

---

## Repository Structure

```
.
├── lib/                           # Shared utilities and helper functions
├── modules/                       # Core Python modules for each workflow
│   ├── keyword_search_module.py
│   ├── paper_id_search_module.py
│   └── publication_search_module.py
├── keyword_search/                # Keyword-based search artifacts & outputs
├── paper_id_search/               # Paper-ID–based search artifacts & outputs
├── publication_search/            # Publication-year–based search artifacts & outputs
├── driver.ipynb                   # Jupyter notebook orchestrating all workflows
└── README.md                      # This file
```

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/ResearchGraph.git
   cd ResearchGraph
   ```

2. **Create a Python virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install \
     pyspark \
     graphframes \
     pyarrow \
     ipywidgets \
     pyvis==0.3.1 \
     networkx \
     matplotlib
   ```

4. **Download Semantic Scholar embeddings**

   * Obtain API access and set your `SEMANTIC_SCHOLAR_API_KEY` environment variable.
   * Ensure you have access to precomputed paper embeddings (e.g., SBERT vectors saved in Parquet).

---

## Usage

All three workflows are driven by `driver.ipynb`. Launch it with:

```bash
jupyter notebook driver.ipynb
```

### 1. Initialize Spark

```python
from keyword_search_module import initialize_spark
spark, sc = initialize_spark(driver_memory="6g", shuffle_partitions=32)
```

### 2. Distribute Modules to Executors

```python
sc.addPyFile("modules/keyword_search_module.py")
sc.addPyFile("modules/publication_search_module.py")
sc.addPyFile("modules/paper_id_search_module.py")
```

### 3. Set Project Roots & Checkpointing

```python
import keyword_search_module as ksm
import publication_search_module as psm
import paper_id_search_module as pidm

# Keyword search
ksm.PROJECT_ROOT = "/home/jovyan/Final Project/keyword_search"
sc.setCheckpointDir(f"{ksm.PROJECT_ROOT}/checkpoints")

# Publication search
psm.PROJECT_ROOT = "/home/jovyan/Final Project/publication_search"

# Paper-ID search
pidm.PROJECT_ROOT = "/home/jovyan/Final Project/paper_id_search"
sc.setCheckpointDir(f"{pidm.PROJECT_ROOT}/checkpoints")
```

### 4. Run Workflows

* **Keyword-based Search**

  ```python
  from keyword_search_module import build_graph_widget
  build_graph_widget(spark, sc)
  ```

* **Paper-ID–based Search**

  ```python
  from paper_id_search_module import build_id_graph_widget
  build_id_graph_widget(spark, sc)
  ```

* **Publication–Year Search**

  ```python
  from publication_search_module import build_publication_graph_widget
  build_publication_graph_widget(spark, sc)
  ```

---

## Future Work

* **Multi-Hop Neighborhoods** for richer context
* **Streaming Updates** via Kafka or Spark Structured Streaming
* **Alternative Community Algorithms** (Louvain, Leiden, spectral clustering)
* **Heterogeneous Edge Types** (co-authorship, venue co-occurrence)
* **Web-Based Dashboards** (D3.js, Neo4j Bloom)
* **Cloud Deployment** (EMR, Dataproc, Databricks) with GPU-accelerated vector ops
* **Collaborative Features** (annotations, saved queries, exportable reports)

---

> 🔧 *Need help or want to contribute?*
>
> * Open an issue or pull request
> * Check the `modules/` folder for detailed docstrings and unit tests
> * Join the conversation in Discussions

---
