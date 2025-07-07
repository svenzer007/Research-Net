# paper_id_search_module.py

import os, time, json, math, gzip, pickle, requests
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf, lit, coalesce
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType,
    ArrayType, DoubleType
)

# for UI
from IPython.display import display, HTML
import ipywidgets as widgets
from pyvis.network import Network


# ─── 1. SPARK INITIALIZATION (reuse from keyword module) ─────────────────────

def initialize_spark(
    app_name: str = "ResearchGraphByID",
    driver_memory: str = "14g",
    shuffle_partitions: int = 32
):
    """Initialize SparkSession with GraphFrames support."""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    return spark, sc


# ─── 2. SCHEMA DEFINITION ─────────────────────────────────────────────────────

def define_schema():
    """Schema for Semantic-Scholar paper JSON (with embedding)."""
    return StructType([
        StructField("paperId", StringType(), False),
        StructField("title",   StringType(), True),
        StructField("year",    IntegerType(), True),
        StructField("references", ArrayType(
            StructType([StructField("paperId", StringType(), False)])
        ), True),
        StructField("citations", ArrayType(
            StructType([StructField("paperId", StringType(), False)])
        ), True),
        StructField("embedding", StructType([
            StructField("model",  StringType(), True),
            StructField("vector", ArrayType(DoubleType()), True)
        ]), True)
    ])


# ─── 3. API FETCH ────────────────────────────────────────────────────────────

API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
FIELDS  = "title,year,referenceCount,citationCount,references,citations,embedding"

def fetch_batch(id_lists):
    """Fetch metadata for batches of paper-IDs (including embeddings)."""
    session = requests.Session()
    out = []
    for batch in id_lists:
        time.sleep(2)  # throttle
        api_key = "X3R0FOZJ2Q57yiS3W6Sgp7F3mfJWna9B7K3vLy3N"
        headers = {"x-api-key": api_key}
        r = session.post(API_URL, params={"fields":FIELDS}, headers = headers, json={"ids":batch})
        r.raise_for_status()
        out.extend(r.json())
    return iter(out)


# ─── 4. COSINE‐SIMILARITY UDF ─────────────────────────────────────────────────

def calculate_cosine_similarity(a, b):
    """Safe cosine similarity for two Python lists."""
    if a is None or b is None:
        return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(y*y for y in b))
    return float(dot/(na*nb)) if na and nb else 0.0


# ─── 5. GRAPH BUILDERS ───────────────────────────────────────────────────────

def build_graph_from_ids(spark, sc, seed_ids, min_similarity=0.0):
    """
    Given a list of paper‐IDs, fetch their metadata + 1-hop references,
    build a directed, weighted citation graph.
    """
    schema = define_schema()
    cos_udf = udf(calculate_cosine_similarity, DoubleType())

    # 1) fetch seed metadata
    rdd = sc.parallelize([seed_ids], numSlices=1).mapPartitions(fetch_batch)
    df_seed = spark.read.schema(schema).json(rdd.map(json.dumps)).dropDuplicates(["paperId"])

    # 2) extract 1-hop referenced IDs
    cited = (df_seed
        .select(explode("references.paperId").alias("paperId"))
        .distinct()
        .rdd.map(lambda r:r.paperId).collect()
    )

    # 3) fetch cited metadata
    if cited:
        rdd2 = sc.parallelize([cited],1).mapPartitions(fetch_batch)
        df_cited = spark.read.schema(schema).json(rdd2.map(json.dumps)).dropDuplicates(["paperId"])
    else:
        df_cited = spark.createDataFrame([], schema)

    # 4) union into full vertex set
    df = df_seed.unionByName(df_cited).dropDuplicates(["paperId"]).filter(col("paperId").isNotNull())

    # 5) build vertices DF
    vertices = (df
        .select("paperId","title","year","embedding.vector")
        .withColumnRenamed("paperId","id")
        .dropDuplicates(["id"])
    )

    # 6) build directed citation edges A→B
    refs = (df
        .select(col("paperId").alias("src"), explode("references.paperId").alias("dst"))
        .dropDuplicates(["src","dst"])
    )
    # restrict to 1-hop
    refs = refs.join(vertices.select(col("id").alias("dst")), on="dst", how="inner")

    # attach embeddings & compute weight
    emb_s = vertices.select(col("id").alias("src"), col("vector").alias("emb_src"))
    emb_d = vertices.select(col("id").alias("dst"), col("vector").alias("emb_dst"))
    edges = (refs
        .join(emb_s, "src", "left")
        .join(emb_d, "dst", "left")
        .withColumn("weight", cos_udf("emb_src","emb_dst"))
        .select("src","dst","weight")
    )

    # 7) assemble GraphFrame
    from graphframes import GraphFrame
    g = GraphFrame(vertices, edges)
    g.vertices.cache(); g.edges.cache()
    return g


# ─── 6. PAGERANK ───────────────────────────────────────────────────────────────

def compute_pagerank(g, reset_prob=0.15, max_iter=10):
    """DataFrame‐based weighted PageRank on GraphFrame."""
    ranks = g.vertices.select("id").withColumn("rank", lit(1.0))
    for _ in range(max_iter):
        contribs = (g.edges
            .join(ranks.withColumnRenamed("id","src"), on="src")
            .withColumn("contrib", col("rank")*col("weight"))
            .select(col("dst").alias("id"), "contrib")
        )
        summed = contribs.groupBy("id").sum("contrib").withColumnRenamed("sum(contrib)","sumContrib")
        ranks = (g.vertices.select("id")
            .join(summed, on="id", how="left")
            .withColumn("rank", lit(reset_prob) + (1-reset_prob)*coalesce(col("sumContrib"),lit(0.0)))
            .select("id","rank")
        )
    return ranks.join(g.vertices.select("id","title"), on="id")


# ─── 7. VISUALIZATION ──────────────────────────────────────────────────────────

def generate_interactive_graph(g, output_file, project_root:Path, height="800px", width="100%"):
    """PyVis interactive viz of GraphFrame g (directed citation graph)."""
    pr = compute_pagerank(g).orderBy(col("rank").desc()).limit(50).collect()
    top_ids = {r.id for r in pr[:10]}
    net = Network(height=height, width=width, directed=True, notebook=True, cdn_resources="in_line")

    for v in g.vertices.collect():
        net.add_node(v.id,
                     label=v.title if v.id in top_ids else " ",
                     title=v.title or "",
                     size=20 if v.id in top_ids else 10)

    for e in g.edges.collect():
        thickness = max(e.weight*5, 1.0)
        net.add_edge(e.src, e.dst, value=thickness, title=f"{e.weight:.3f}")

    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150,
                          spring_strength=0.08, damping=0.4)

    out_dir = project_root/"output"
    out_dir.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    os.chdir(out_dir)
    net.write_html(output_file)
    os.chdir(old)

    full = out_dir/output_file
    display(HTML(str(full)))
    return str(full)


# ─── 8. PARQUET SAVE ──────────────────────────────────────────────────────────

def save_graph_parquet(g, name, project_root:Path, shuffle_partitions:int=32):
    """Persist vertices & edges as Parquet under project_root/processed/…"""
    v_out = project_root/"processed"/"vertices"
    e_out = project_root/"processed"/"edges_weighted"
    g.vertices.select("id","title","year").write.mode("overwrite").partitionBy("year").parquet(str(v_out))
    g.edges.repartition(shuffle_partitions).write.mode("overwrite").parquet(str(e_out))
    return str(v_out), str(e_out)


# ─── 9. WIDGETS ───────────────────────────────────────────────────────────────

def build_id_graph_widget(spark, sc):
    """Single UI to input seed IDs, build & visualize the citation graph."""
    txt = widgets.Text(
        description="IDs:",
        placeholder="e.g. 649def34f8be52c8b66281af98ae884c09aef38b, ARXIV:2106.15928"
    )
    btn = widgets.Button(description="Build Graph", button_style="primary")
    out = widgets.Output()

    def on_click(b):
        with out:
            out.clear_output()
            # parse IDs
            seed_ids = [i.strip() for i in txt.value.split(",") if i.strip()]
            if not seed_ids:
                print("Please enter at least one paper ID.")
                return

            print("Building graph for IDs:", seed_ids)
            # build the GraphFrame (uses default min_similarity)
            g = build_graph_from_ids(spark, sc, seed_ids)

            if g is None:
                print("Graph construction failed.")
                return

            # # show PageRank top‐10
            pr = compute_pagerank(g).orderBy(col("rank").desc()).limit(10)
            # print("\nTop 10 papers by PageRank:")
            # pr.show(truncate=False)

            # save to parquet
            v_path, e_path = save_graph_parquet(g, "id_graph", PROJECT_ROOT)
            print(f"Vertices written to {v_path}\nEdges written to {e_path}")

            # render interactive HTML
            html_path = generate_interactive_graph(g, "id_graph.html", PROJECT_ROOT)
            print(f"Interactive graph saved & displayed from:\n  {html_path}")

    btn.on_click(on_click)
    display(widgets.VBox([txt, btn, out]))
