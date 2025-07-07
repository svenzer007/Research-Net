# keyword_search_module.py

import os
import time
import json
import math
import gzip
import pickle
import requests

from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf, lit, coalesce
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType,
    ArrayType, DoubleType
)

# for UI functions
from IPython.display import display, HTML, IFrame
import ipywidgets as widgets
from pyvis.network import Network


# ─── 1. SPARK INITIALIZATION ──────────────────────────────────────────────────

def initialize_spark(
    app_name: str = "ResearchGraphKeyword",
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
    """Define Spark schema matching Semantic Scholar JSON (with embedding & tldr)."""
    return StructType([
        StructField("paperId", StringType(), False),
        StructField("title",   StringType(), True),
        StructField("abstract", StringType(), True),
        StructField("year",    IntegerType(), True),
        StructField("authors", ArrayType(
            StructType([
                StructField("authorId", StringType(), True),
                StructField("name",     StringType(), True)
            ])
        ), True),
        StructField("references", ArrayType(
            StructType([StructField("paperId", StringType(), False)])
        ), True),
        StructField("citations", ArrayType(
            StructType([StructField("paperId", StringType(), False)])
        ), True),
        StructField("embedding", StructType([
            StructField("model",  StringType(), True),
            StructField("vector", ArrayType(DoubleType()), True)
        ]), True),
        StructField("tldr", StructType([
            StructField("text", StringType(), True)
        ]), True)
    ])


# ─── 3. SEMANTIC SCHOLAR API HELPERS ──────────────────────────────────────────

def keyword_search(keywords, limit=100, offset=0, api_base=None, fields=None):
    """Search for papers by keywords via Semantic Scholar /paper/search."""
    if api_base is None:
        api_base = "https://api.semanticscholar.org/graph/v1"
    if fields is None:
        fields = "paperId,title,abstract,year,authors,references,citations,embedding,tldr"

    if isinstance(keywords, list):
        query = " ".join(keywords)
    else:
        query = keywords

    params = {
        "query": query,
        "fields": fields,
        "limit": min(limit, 100),
        "offset": offset
    }
    url = f"{api_base}/paper/search"
    session = requests.Session()
    try:
        api_key = "X3R0FOZJ2Q57yiS3W6Sgp7F3mfJWna9B7K3vLy3N"
        headers = {"x-api-key": api_key}
        r = session.get(url, params=params, headers = headers)
        r.raise_for_status()
        data = r.json().get("data", [])
        print(f"Found {len(data)} papers for query “{query}”")
        return data
    except Exception as e:
        print(f"keyword_search error: {e}")
        return []


def fetch_papers_batch(id_lists, api_base=None, fields=None):
    """Fetch full paper metadata (including embeddings) in batches."""
    id_lists = list(id_lists)
    if api_base is None:
        api_base = "https://api.semanticscholar.org/graph/v1"
    if fields is None:
        fields = "paperId,title,abstract,year,authors,references,citations,embedding,tldr"
    url = f"{api_base}/paper/batch"
    session = requests.Session()
    out = []
    total = len(id_lists)
    for i, batch in enumerate(id_lists):
        if i > 0:
            time.sleep(1)
        try:
            r = session.post(url, params={"fields": fields}, json={"ids": batch})
            r.raise_for_status()
            batch_data = r.json()
            print(f"Fetched batch {i+1}/{total}: {len(batch_data)} papers")
            out.extend(batch_data)
        except Exception as e:
            print(f"fetch_papers_batch error on batch {i+1}: {e}")
    return iter(out)


# ─── 4. COSINE SIMILARITY UDF ─────────────────────────────────────────────────

def calculate_cosine_similarity(a, b):
    """Safe cosine similarity for two Python lists."""
    if a is None or b is None:
        return 0.0
    try:
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        return float(dot/(na*nb)) if na>0 and nb>0 else 0.0
    except:
        return 0.0


# ─── 5. GRAPH BUILDERS ────────────────────────────────────────────────────────

def build_graph_from_keywords(
    spark, sc, keywords,
    limit=100, central_paper=None, min_similarity=0.0
):
    """Main entry: search by keywords → build GraphFrame (star or fully-connected)."""
    schema = define_schema()
    cos_sim_udf = udf(calculate_cosine_similarity, DoubleType())

    # 1) search
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    papers = keyword_search(keywords, limit=limit)
    if not papers:
        print("No papers found."); return None

    # 2) to DF
    json_rdd = sc.parallelize([papers]).flatMap(lambda x:x)
    df_seed = spark.read.schema(schema).json(json_rdd.map(json.dumps)).dropDuplicates(["paperId"])

    # 3) optionally add central paper
    ids = df_seed.select("paperId").rdd.map(lambda r:r.paperId).collect()
    if central_paper and central_paper not in ids:
        cent_rdd = sc.parallelize([[central_paper]]).mapPartitions(fetch_papers_batch)
        cent_df  = spark.read.schema(schema).json(cent_rdd.map(json.dumps))
        if cent_df.count()>0:
            print("Central Paper found")
            df_seed = df_seed.unionByName(cent_df).dropDuplicates(["paperId"])
            ids.append(central_paper)

    # 4) vertices
    vertices = (
        df_seed
        .select("paperId","title","year","abstract","embedding.vector","authors","tldr.text")
        .withColumnRenamed("paperId","id").withColumnRenamed("text","tldr")
        .dropDuplicates(["id"])
    )
    # 5) edges
    if central_paper and central_paper in ids:
        # edges = build_star_edges(vertices, central_paper, min_similarity, cos_sim_udf)
        edges = build_star_edges(vertices, df_seed, central_paper, cos_sim_udf)
    else:
        # edges = build_fully_connected_edges(vertices, min_similarity, cos_sim_udf)
         edges = build_citation_edges(vertices, df_seed, cos_sim_udf)
    # 6) GraphFrame
    from graphframes import GraphFrame
    g = GraphFrame(vertices, edges)
    g.vertices.cache(); g.edges.cache()
    print(f"Graph: {g.vertices.count()} vertices, {g.edges.count()} edges")
    return g

def build_star_edges(vertices, df_seed, central_paper, cos_sim_udf):
    """
    1) seed_cits: directed citation edges A→B among seed (excluding central)
    2) forward/back: directed edges between central & seed when a real citation exists
    3) fallback: central→seed for any seed not in (forward ∪ back), tagged for black coloring
    4) restrict all edges to IDs in `vertices`
    5) compute weight via cos_sim_udf
    """
    # 1) explode all A→B citations in seed
    all_cits = (
      df_seed
        .select(col("paperId").alias("src"), explode("references.paperId").alias("dst"))
        .dropDuplicates(["src","dst"])
    )
    
    # a) directed citation edges among the other seed papers
    seed_cits = (
      all_cits
        .filter((col("src") != central_paper) & (col("dst") != central_paper))
        .select("src","dst")
        .withColumn("edge_type", lit("seed_cite"))
    )
    
    # b) central→other when central cites other
    forward = (
      all_cits
        .filter(col("src") == central_paper)
        .select(lit(central_paper).alias("src"), col("dst"))
        .withColumn("edge_type", lit("central_cites"))
    )
    
    # c) other→central when other cites central
    backward = (
      all_cits
        .filter(col("dst") == central_paper)
        .select(col("src"), lit(central_paper).alias("dst"))
        .withColumn("edge_type", lit("cites_central"))
    )
    
    # 2) fallback: any seed paper not in forward.dst or backward.src
    others = vertices.select(col("id").alias("nid")).filter(col("nid") != central_paper)
    connected = (
      forward.select("dst").withColumnRenamed("dst","nid")
      .union(backward.select("src").withColumnRenamed("src","nid"))
      .distinct()
    )
    unconnected = others.join(connected, on="nid", how="left_anti")
    fallback = (
      unconnected
        .select(lit(central_paper).alias("src"), col("nid").alias("dst"))
        .withColumn("edge_type", lit("fallback"))
    )
    
    # 3) union all raw edges
    raw = seed_cits.union(forward).union(backward).union(fallback)
    
    # 4) restrict to vertices set on both ends
    valid = vertices.select(col("id").alias("vid"))
    raw = (
      raw
        .join(valid.withColumnRenamed("vid","src"), on="src")
        .join(valid.withColumnRenamed("vid","dst"), on="dst")
    )
    
    # 5) attach embeddings & compute weight
    emb_s = vertices.select(col("id").alias("src"), col("vector").alias("emb_src"))
    emb_d = vertices.select(col("id").alias("dst"), col("vector").alias("emb_dst"))
    edges = (
      raw
        .join(emb_s, on="src", how="left")
        .join(emb_d, on="dst", how="left")
        .withColumn("weight", cos_sim_udf("emb_src","emb_dst"))
        .select("src","dst","weight","edge_type")
    )
    return edges


def build_reference_based_edges(vertices):
    """Fallback MST-style edges if no embeddings present."""
    ids = [r.id for r in vertices.select("id").collect()]
    schema = StructType([StructField("src",StringType(),False),
                         StructField("dst",StringType(),False),
                         StructField("weight",DoubleType(),False)])
    if len(ids)<=1:
        return vertices.sparkSession.createDataFrame([], schema)
    data=[]
    for other in ids[1:]:
        data += [(ids[0],other,1.0),(other,ids[0],1.0)]
    return vertices.sparkSession.createDataFrame(data, schema)


def build_citation_edges(vertices, df_seed, cos_sim_udf):
    """
    Build directed 1-hop citation edges A→B for any A in seed whose references
    include B, keeping only B’s that are in our vertex set, and weighting by
    cosine-similarity of embeddings (0 if missing).
    """
    # 1) every citation A→B
    refs = (
        df_seed
        .select(col("paperId").alias("src"), explode("references.paperId").alias("dst"))
        .dropDuplicates(["src","dst"])
    )
    # 2) keep only those dst ∈ our vertices
    refs = refs.join(
        vertices.select(col("id").alias("dst")),
        on="dst", how="inner"
    )
    # 3) bring in embeddings on both ends
    emb_src = vertices.select(col("id").alias("src"), col("vector").alias("emb_src"))
    emb_dst = vertices.select(col("id").alias("dst"), col("vector").alias("emb_dst"))
    edges = (
        refs
        .join(emb_src, on="src", how="left")
        .join(emb_dst, on="dst", how="left")
        .withColumn("weight", cos_sim_udf("emb_src","emb_dst"))
        .withColumn("edge_type", lit("citation"))
        .select("src","dst","weight","edge_type")
    )
    return edges

# ─── 6. PAGERANK ───────────────────────────────────────────────────────────────

def compute_pagerank(g, reset_prob=0.15, max_iter=10):
    """Run weighted PageRank (DataFrame-based) on GraphFrame g."""
    ranks = g.vertices.select("id").withColumn("rank", lit(1.0))
    for i in range(max_iter):
        contribs = (
            g.edges
             .join(ranks.withColumnRenamed("id","src"), on="src")
             .withColumn("contrib", col("rank")*col("weight"))
             .select(col("dst").alias("id"), "contrib")
        )
        summed = contribs.groupBy("id").sum("contrib").withColumnRenamed("sum(contrib)","sumContrib")
        ranks = (
            g.vertices.select("id")
             .join(summed, on="id", how="left")
             .withColumn("rank", lit(reset_prob) + (1-reset_prob)*coalesce(col("sumContrib"),lit(0.0)))
             .select("id","rank")
        )
    return ranks.join(g.vertices.select("id","title"), on="id")


# ─── 7. VISUALIZATION ──────────────────────────────────────────────────────────

def generate_interactive_graph(
    g,
    output_file,
    project_root: Path,
    height="800px",
    width="100%",
    min_thickness: float = 1.0,    # ◀ minimum edge thickness
    scale: float = 5.0             # ◀ scale factor for non-zero weights
):
    """Render & save a PyVis interactive graph of GraphFrame g."""
    # compute PageRank for sizing/labels
    pr = compute_pagerank(g).orderBy(col("rank").desc()).limit(50).collect()
    top_ids = {r.id for r in pr[:10]}

    net = Network(height=height, width=width, directed=True, notebook=True, cdn_resources="in_line")
    for v in g.vertices.collect():
        net.add_node(v.id,
                     label=v.title if v.id in top_ids else " ",
                     title=v.title or "",
                     size=20 if v.id in top_ids else 10)

    # add edges with a floor on thickness and black color for zero-weight
    for e in g.edges.collect():
        w = e.weight or 0.0
        thickness = max(w * scale, min_thickness)
        
        if e.edge_type == "fallback":
            # black, undirected-look
            net.add_edge(e.src, e.dst,
                         value=thickness,
                         title=f"{w:.3f}",
                         color="black",
                         arrows="")  
        else:
            # directed citation or similarity edge
            net.add_edge(e.src, e.dst,
                         value=thickness,
                         title=f"{w:.3f}")
            
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150, spring_strength=0.08, damping=0.4)

    # ensure output folder exists
    out_dir = project_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remember original working-dir
    old_cwd = Path.cwd()
    # Switch into output folder
    os.chdir(out_dir)
    
    # Write just the file name
    net.write_html(output_file)  

    # Return to original working-dir
    os.chdir(old_cwd)

    # Now read & display
    full_path = out_dir / output_file
    display(HTML(str(full_path)))
    print(f"Saved interactive graph to {full_path}")
    return str(full_path)


# ─── 8. SAVE/LOAD ─────────────────────────────────────────────────────────────

def save_graph_parquet(g, name, project_root: Path, shuffle_partitions: int = 32):
    """
    Persist vertices & edges as Parquet under project_root/processed/.
     - vertices partitioned by year
     - edges repartitioned for downstream joins
    """
    out_verts = project_root/"processed"/"vertices"
    out_edges = project_root/"processed"/"edges_weighted"
    # vertices: keep id, title, year
    g.vertices \
     .select("id","title","year") \
     .write \
     .mode("overwrite") \
     .partitionBy("year") \
     .parquet(str(out_verts))
    # edges: shuffle into N files
    g.edges \
     .repartition(shuffle_partitions) \
     .write \
     .mode("overwrite") \
     .parquet(str(out_edges))
    print(f"Vertices → {out_verts}, edges → {out_edges}")
    return str(out_verts), str(out_edges)


# ─── 9. OPTIONAL UI FUNCTIONS ─────────────────────────────────────────────────

def search_papers_widget():
    """Display widget to search keywords (calls keyword_search)."""
    kw = widgets.Text(description="Keywords:")
    lim = widgets.IntSlider(description="Max Papers:", min=10, max=100, step=10, value=50)
    btn = widgets.Button(description="Search", button_style="primary")
    out = widgets.Output()

    def on_click(b):
        with out:
            out.clear_output()
            terms = [t.strip() for t in kw.value.split(",")]
            papers = keyword_search(terms, limit=lim.value)
            for i,p in enumerate(papers[:5],1):
                print(f"{i}. {p.get('title')} ({p.get('year')})\n   ID={p.get('paperId')}\n")
            if len(papers)>5:
                print(f"... and {len(papers)-5} more")
    btn.on_click(on_click)
    display(widgets.VBox([kw, lim, btn, out]))


def build_graph_widget(spark, sc):
    """Display widget to build & visualize a keyword-based graph."""
    kw = widgets.Text(description="Keywords:")
    lim = widgets.IntSlider(description="Max Papers:", min=10, max=100, step=10, value=50)
    # sim = widgets.FloatSlider(description="Min Sim:", min=0.1, max=0.9, step=0.1, value=0.3)
    cent = widgets.Text(description="Base Paper:")
    btn = widgets.Button(description="Build Graph", button_style="primary")
    out = widgets.Output()

    def on_click(b):
        with out:
            out.clear_output()
            terms = [t.strip() for t in kw.value.split(",")]
            g = build_graph_from_keywords(spark, sc, terms, limit=lim.value,
                                          # min_similarity=sim.value,
                                          central_paper=cent.value or None)
            if g:
                pr = compute_pagerank(g).orderBy(col("rank").desc()).limit(10)
                pr.show(truncate=False)
                generate_interactive_graph(g, "kw_graph.html", PROJECT_ROOT)
    btn.on_click(on_click)
    display(widgets.VBox([kw, lim, cent, btn, out]))#sim, cent, btn, out]))
