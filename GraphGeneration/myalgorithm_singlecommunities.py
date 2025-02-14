import os
import csv
import asyncio
import itertools
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Environment loading
from dotenv import load_dotenv
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# OpenAI client
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Elasticsearch
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# For splitting text (this is the key change)
from langchain.text_splitter import CharacterTextSplitter

# Neo4j + GraphRAG
from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists

# Clustering
from sklearn.neighbors import NearestNeighbors
import igraph
import leidenalg

# Disable HTTPS warnings if needed
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Override print to also write to a log file ---
import builtins
import atexit

_log_file = open("elastic-vs.graphrag-gpt.txt", "w", encoding="utf-8")
_original_print = builtins.print
def print(*args, **kwargs):
    _original_print(*args, **kwargs)
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    output = sep.join(str(arg) for arg in args) + end
    _log_file.write(output)
    _log_file.flush()
builtins.print = print
atexit.register(_log_file.close)
# --- End of print override ---

###########################
# GLOBAL SETTINGS (will be overridden by main variables)
###########################
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 0

###########################
# ASYNC HELPERS
###########################
async def _extract_text_from_pdf_async(pdf_path):
    """Asynchronously extract text from a PDF using PyPDF2 in a ThreadPool."""
    from PyPDF2 import PdfReader
    def _read_pdf(path):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        text = await loop.run_in_executor(pool, _read_pdf, pdf_path)
    return text

async def _async_invoke_llm(llm, prompt):
    loop = asyncio.get_running_loop()
    def _sync_invoke():
        return llm.invoke(prompt)
    return await loop.run_in_executor(None, _sync_invoke)

async def _generate_name_for_text_async(llm, text, prompt="Give a short descriptive name for this text:"):
    if not text.strip():
        return "NoContent"
    full_prompt = f"""{prompt}
Text:
{text}

Short name:"""
    response = await _async_invoke_llm(llm, full_prompt)
    return response.content.strip()

async def _generate_name_for_group_async(llm, texts, group_prompt="Name this community based on the following texts:"):
    if not texts:
        return "EmptyGroup"
    snippet = "\n\n".join(texts[:10])
    full_prompt = f"""{group_prompt}
Here are some representative texts of this group:
{snippet}

Short descriptive name for this group:"""
    response = await _async_invoke_llm(llm, full_prompt)
    return response.content.strip()

def _check_if_graph_exists(driver):
    with driver.session() as session:
        result = session.run("MATCH (c:Chunk) RETURN COUNT(c) AS cnt")
        record = result.single()
        return (record["cnt"] > 0)

def _clear_neo4j_chunks_and_index(driver, index_name):
    try:
        drop_index_if_exists(driver, index_name)
    except:
        pass
    with driver.session() as session:
        session.run("MATCH (c:Chunk) DETACH DELETE c")

###########################
# BUILD THE KNOWLEDGE GRAPH (FROM PDF)
###########################
async def _ingest_pdf_async(driver, pdf_path, llm, embedding_model, dims, index_name):
    print("📌 Building graph from PDF:", pdf_path)
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedding_model,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=FixedSizeSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    )
    pdf_text = await _extract_text_from_pdf_async(pdf_path)
    await kg_builder.run_async(text=pdf_text)
    create_vector_index(driver, index_name, label="Chunk", embedding_property="embedding", dimensions=dims, similarity_fn="cosine")
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL OR c.text IS NULL OR c.text = ''
            DETACH DELETE c
        """)
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.uuid IS NULL
            SET c.uuid = elementId(c)
        """)
    with driver.session() as session:
        results = session.run("""
            MATCH (c:Chunk)
            RETURN elementId(c) AS nodeId, c.embedding AS embedding, c.text AS text
        """)
        records = list(results)
    node_ids = []
    embeddings = []
    node_text_map = {}
    for r in records:
        emb = r["embedding"]
        txt = r["text"]
        if emb and txt:
            node_ids.append(r["nodeId"])
            embeddings.append(emb)
            node_text_map[r["nodeId"]] = txt
    X = np.array(embeddings)
    num_nodes = len(X)
    if num_nodes == 0:
        print("No valid chunks found after ingestion. Exiting build.")
        return
    k = min(2, num_nodes)
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    edges = []
    weights = []
    for i in range(num_nodes):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = max(0, 1 - dist)
            edges.append((i, j_idx))
            weights.append(sim)
    g = igraph.Graph(n=num_nodes, edges=edges, directed=False)
    g.es["weight"] = weights
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"], resolution_parameter=1.0)
    community_labels = partition.membership
    with driver.session() as session:
        session.run("MATCH ()-[r:SIMILAR]->() DELETE r")
        for i in range(num_nodes):
            idA = node_ids[i]
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                sim = 1 - dist
                idB = node_ids[j_idx]
                session.run("""
                    MERGE (a:Chunk { uuid: $idA })
                    MERGE (b:Chunk { uuid: $idB })
                    MERGE (a)-[r:SIMILAR]->(b)
                    SET r.score = $sim
                """, {"idA": idA, "idB": idB, "sim": sim})
        for i in range(num_nodes):
            c_comm = int(community_labels[i])
            c_id   = node_ids[i]
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.community = $community
            """, {"idVal": c_id, "community": c_comm})
    async def _name_chunk(i):
        txt = node_text_map[node_ids[i]]
        return await _generate_name_for_text_async(llm, txt, prompt="Give a short descriptive name for this chunk's content:")
    chunk_name_tasks = [_name_chunk(i) for i in range(num_nodes)]
    chunk_names = await asyncio.gather(*chunk_name_tasks)
    with driver.session() as session:
        for i in range(num_nodes):
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal, c.name = $chunkName
            """, {"idVal": node_ids[i], "chunkName": chunk_names[i]})
    community_to_nodes = defaultdict(list)
    for i, nid in enumerate(node_ids):
        c_label = community_labels[i]
        community_to_nodes[c_label].append(nid)
    async def _name_community(comm_id, nodelist):
        chunk_texts = [node_text_map[nid] for nid in nodelist]
        return await _generate_name_for_group_async(llm, chunk_texts, group_prompt="Name this first-level community based on its chunk texts:")
    comm_tasks = []
    for cid, nlist in community_to_nodes.items():
        comm_tasks.append((cid, _name_community(cid, nlist)))
    comm_results = await asyncio.gather(*[t[1] for t in comm_tasks])
    community_name_map = {}
    for i, (comm_id, _) in enumerate(comm_tasks):
        community_name_map[comm_id] = comm_results[i]
    with driver.session() as session:
        for comm_id, c_name in community_name_map.items():
            session.run("""
                MATCH (c:Chunk)
                WHERE c.community = $commId
                SET c.community_name = $cName
            """, {"commId": comm_id, "cName": c_name})
    community_centroids = {}
    for comm_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        community_centroids[comm_id] = np.mean(subset_emb, axis=0) if len(subset_emb) > 0 else None
    comm_ids = sorted(community_centroids.keys())
    comm_index = {c: idx for idx, c in enumerate(comm_ids)}
    num_comms = len(comm_ids)
    if num_comms > 1:
        edges2, weights2 = [], []
        for c1, c2 in itertools.combinations(comm_ids, 2):
            v1 = community_centroids[c1]
            v2 = community_centroids[c2]
            if v1 is None or v2 is None:
                sim = 0
            else:
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                sim = (np.dot(v1, v2) / denom) if denom > 1e-12 else 0
            sim = max(0, sim)
            edges2.append((comm_index[c1], comm_index[c2]))
            weights2.append(sim)
        g2 = igraph.Graph(n=num_comms, edges=edges2, directed=False)
        g2.es["weight"] = weights2
        partition2 = leidenalg.find_partition(g2, leidenalg.RBConfigurationVertexPartition, weights=g2.es["weight"], resolution_parameter=1.0)
        super_community_labels = partition2.membership
        community_to_super = {comm_ids[i]: super_community_labels[i] for i in range(num_comms)}
    else:
        community_to_super = {cid: 0 for cid in comm_ids}
    with driver.session() as session:
        for i in range(num_nodes):
            c_comm = community_labels[i]
            sc = community_to_super[c_comm]
            c_id = node_ids[i]
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.super_community = $sc
            """, {"idVal": c_id, "sc": sc})
    super_comm_to_first_comm = defaultdict(list)
    for fc in comm_ids:
        sc = community_to_super[fc]
        super_comm_to_first_comm[sc].append(fc)
    if len(super_comm_to_first_comm) > 1:
        async def _name_super_comm(sc, fc_list):
            fc_names = [community_name_map[fc] for fc in fc_list]
            return await _generate_name_for_group_async(llm, fc_names, group_prompt="Name this second-level community based on its first-level community names:")
        super_tasks = []
        for sc, fc_list in super_comm_to_first_comm.items():
            super_tasks.append((sc, _name_super_comm(sc, fc_list)))
        super_results = await asyncio.gather(*[t[1] for t in super_tasks])
        super_community_name_map = {}
        for i, (sc, _) in enumerate(super_tasks):
            super_community_name_map[sc] = super_results[i]
        with driver.session() as session:
            for i in range(num_nodes):
                c_comm = community_labels[i]
                sc = community_to_super[c_comm]
                sc_name = super_community_name_map.get(sc, "SingleSuperCommunity")
                c_id = node_ids[i]
                session.run("""
                    MATCH (c:Chunk { uuid: $idVal })
                    SET c.super_community_name = $scName
                """, {"idVal": c_id, "scName": sc_name})
    else:
        with driver.session() as session:
            session.run("""
                MATCH (c:Chunk)
                SET c.super_community_name = "SingleSuperCommunity"
            """)
    print("✅ Knowledge Graph built successfully from PDF.")

def _build_graph_async(driver, pdf_path, llm, embedding_model, dims, index_name):
    asyncio.run(_ingest_pdf_async(driver=driver, pdf_path=pdf_path, llm=llm, embedding_model=embedding_model, dims=dims, index_name=index_name))

###########################
# BUILD THE KNOWLEDGE GRAPH (FROM DATASET DOCUMENTS)
###########################
async def _ingest_docs_async(driver, docs, llm, embedding_model, dims, index_name):
    print("📌 Building graph from a documents list...")
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedding_model,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=FixedSizeSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    )
    dataset_text = "\n\n".join(docs)
    await kg_builder.run_async(text=dataset_text)
    create_vector_index(driver, index_name, label="Chunk", embedding_property="embedding", dimensions=dims, similarity_fn="cosine")
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL OR c.text IS NULL OR c.text = ''
            DETACH DELETE c
        """)
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.uuid IS NULL
            SET c.uuid = elementId(c)
        """)
    with driver.session() as session:
        results = session.run("""
            MATCH (c:Chunk)
            RETURN elementId(c) AS nodeId, c.embedding AS embedding, c.text AS text
        """)
        records = list(results)
    node_ids = []
    embeddings = []
    node_text_map = {}
    for r in records:
        emb = r["embedding"]
        txt = r["text"]
        if emb and txt:
            node_ids.append(r["nodeId"])
            embeddings.append(emb)
            node_text_map[r["nodeId"]] = txt
    X = np.array(embeddings)
    num_nodes = len(X)
    if num_nodes == 0:
        print("No valid chunks found after ingestion. Exiting build.")
        return
    k = min(2, num_nodes)
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    edges = []
    weights = []
    for i in range(num_nodes):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = max(0, 1 - dist)
            edges.append((i, j_idx))
            weights.append(sim)
    g = igraph.Graph(n=num_nodes, edges=edges, directed=False)
    g.es["weight"] = weights
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"], resolution_parameter=1.0)
    community_labels = partition.membership
    with driver.session() as session:
        session.run("MATCH ()-[r:SIMILAR]->() DELETE r")
        for i in range(num_nodes):
            idA = node_ids[i]
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                sim = 1 - dist
                idB = node_ids[j_idx]
                session.run("""
                    MERGE (a:Chunk { uuid: $idA })
                    MERGE (b:Chunk { uuid: $idB })
                    MERGE (a)-[r:SIMILAR]->(b)
                    SET r.score = $sim
                """, {"idA": idA, "idB": idB, "sim": sim})
        for i in range(num_nodes):
            c_comm = int(community_labels[i])
            c_id = node_ids[i]
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.community = $community
            """, {"idVal": c_id, "community": c_comm})
    async def _name_chunk(i):
        txt = node_text_map[node_ids[i]]
        return await _generate_name_for_text_async(llm, txt, prompt="Give a short descriptive name for this chunk's content:")
    chunk_name_tasks = [_name_chunk(i) for i in range(num_nodes)]
    chunk_names = await asyncio.gather(*chunk_name_tasks)
    with driver.session() as session:
        for i in range(num_nodes):
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal, c.name = $chunkName
            """, {"idVal": node_ids[i], "chunkName": chunk_names[i]})
    community_to_nodes = defaultdict(list)
    for i, nid in enumerate(node_ids):
        c_label = community_labels[i]
        community_to_nodes[c_label].append(nid)
    async def _name_community(comm_id, nodelist):
        chunk_texts = [node_text_map[nid] for nid in nodelist]
        return await _generate_name_for_group_async(llm, chunk_texts, group_prompt="Name this first-level community based on its chunk texts:")
    comm_tasks = []
    for cid, nlist in community_to_nodes.items():
        comm_tasks.append((cid, _name_community(cid, nlist)))
    comm_results = await asyncio.gather(*[t[1] for t in comm_tasks])
    community_name_map = {}
    for i, (comm_id, _) in enumerate(comm_tasks):
        community_name_map[comm_id] = comm_results[i]
    with driver.session() as session:
        for comm_id, c_name in community_name_map.items():
            session.run("""
                MATCH (c:Chunk)
                WHERE c.community = $commId
                SET c.community_name = $cName
            """, {"commId": comm_id, "cName": c_name})
    community_centroids = {}
    for comm_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        community_centroids[comm_id] = np.mean(subset_emb, axis=0) if len(subset_emb) > 0 else None
    comm_ids = sorted(community_centroids.keys())
    comm_index = {c: idx for idx, c in enumerate(comm_ids)}
    num_comms = len(comm_ids)
    if num_comms > 1:
        edges2, weights2 = [], []
        for c1, c2 in itertools.combinations(comm_ids, 2):
            v1 = community_centroids[c1]
            v2 = community_centroids[c2]
            if v1 is None or v2 is None:
                sim = 0
            else:
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                sim = (np.dot(v1, v2) / denom) if denom > 1e-12 else 0
            sim = max(0, sim)
            edges2.append((comm_index[c1], comm_index[c2]))
            weights2.append(sim)
        g2 = igraph.Graph(n=num_comms, edges=edges2, directed=False)
        g2.es["weight"] = weights2
        partition2 = leidenalg.find_partition(g2, leidenalg.RBConfigurationVertexPartition, weights=g2.es["weight"], resolution_parameter=1.0)
        super_community_labels = partition2.membership
        community_to_super = {comm_ids[i]: super_community_labels[i] for i in range(num_comms)}
    else:
        community_to_super = {cid: 0 for cid in comm_ids}
    with driver.session() as session:
        for i in range(num_nodes):
            c_comm = community_labels[i]
            sc = community_to_super[c_comm]
            c_id = node_ids[i]
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.super_community = $sc
            """, {"idVal": c_id, "sc": sc})
    super_comm_to_first_comm = defaultdict(list)
    for i, fc in enumerate(community_labels):
        sc = None
        # Determine super community label from previously computed mapping
        for cid in community_to_super:
            if cid == fc:
                sc = community_to_super[cid]
                break
        if sc is not None and fc not in super_comm_to_first_comm[sc]:
            super_comm_to_first_comm[sc].append(fc)
    if len(super_comm_to_first_comm) > 1:
        async def _name_super_comm(sc, fc_list):
            fc_names = [community_name_map[fc] for fc in fc_list]
            return await _generate_name_for_group_async(llm, fc_names, group_prompt="Name this second-level community based on its first-level community names:")
        super_tasks = []
        for sc, fc_list in super_comm_to_first_comm.items():
            super_tasks.append((sc, _name_super_comm(sc, fc_list)))
        super_results = await asyncio.gather(*[t[1] for t in super_tasks])
        super_community_name_map = {}
        for i, (sc, _) in enumerate(super_tasks):
            super_community_name_map[sc] = super_results[i]
        with driver.session() as session:
            for i in range(num_nodes):
                c_comm = community_labels[i]
                sc = community_to_super[c_comm]
                sc_name = super_community_name_map.get(sc, "SingleSuperCommunity")
                c_id = node_ids[i]
                session.run("""
                    MATCH (c:Chunk { uuid: $idVal })
                    SET c.super_community_name = $scName
                """, {"idVal": c_id, "scName": sc_name})
    else:
        with driver.session() as session:
            session.run("""
                MATCH (c:Chunk)
                SET c.super_community_name = "SingleSuperCommunity"
            """)
    print("✅ Knowledge Graph built successfully (from these documents).")

def _build_graph_async_from_docs(driver, docs, llm, embedding_model, dims, index_name):
    asyncio.run(_ingest_docs_async(driver=driver, docs=docs, llm=llm, embedding_model=embedding_model, dims=dims, index_name=index_name))

###########################
# ANSWER A QUESTION (GRAPH RAG)
###########################
def answer_question(question, driver, index_name, embedding_model, top_k):
    with driver.session() as session:
        records = list(session.run("""
            MATCH (c:Chunk)
            RETURN elementId(c) AS nodeId, c.embedding AS embedding, c.text AS text, c.community AS community, c.community_name AS communityName, c.super_community AS superComm, c.super_community_name AS superCommunityName
        """))
    if not records:
        print("No Chunks found in the DB, cannot answer question.")
        return "No data available."
    node_ids = []
    embeddings = []
    community_labels = []
    super_community_labels = []
    for r in records:
        node_ids.append(r["nodeId"])
        embeddings.append(r["embedding"])
        community_labels.append(r["community"])
        super_community_labels.append(r["superComm"])
    X = np.array(embeddings)
    community_to_nodes = defaultdict(list)
    for i, nid in enumerate(node_ids):
        c_label = community_labels[i]
        community_to_nodes[c_label].append(nid)
    community_centroids = {}
    for c_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        community_centroids[c_id] = np.mean(subset_emb, axis=0) if len(subset_emb) > 0 else None
    super_comm_to_first_comm = defaultdict(list)
    for i, fc in enumerate(community_labels):
        sc = super_community_labels[i]
        if fc not in super_comm_to_first_comm[sc]:
            super_comm_to_first_comm[sc].append(fc)
    super_community_centroids = {}
    sc_ids = sorted(list(super_comm_to_first_comm.keys()))
    question_embedding = np.array(embedding_model.embed_query(question))
    for sc_id in sc_ids:
        fc_ids = super_comm_to_first_comm[sc_id]
        sc_embs = []
        for fc_id in fc_ids:
            emb_vec = community_centroids.get(fc_id)
            if emb_vec is not None:
                sc_embs.append(emb_vec)
        if sc_embs:
            super_community_centroids[sc_id] = np.mean(sc_embs, axis=0)
        else:
            super_community_centroids[sc_id] = np.zeros_like(question_embedding)
    best_sc, best_sc_sim = None, -1
    for sc_id, sc_centroid in super_community_centroids.items():
        denom = np.linalg.norm(question_embedding) * np.linalg.norm(sc_centroid)
        sim = (np.dot(question_embedding, sc_centroid)/denom) if denom > 1e-12 else 0
        if sim > best_sc_sim:
            best_sc_sim = sim
            best_sc = sc_id
    best_fc, best_fc_sim = None, -1
    for fc_id in super_comm_to_first_comm[best_sc]:
        fc_centroid = community_centroids[fc_id]
        if fc_centroid is None:
            continue
        denom = np.linalg.norm(question_embedding) * np.linalg.norm(fc_centroid)
        sim = (np.dot(question_embedding, fc_centroid)/denom) if denom > 1e-12 else 0
        if sim > best_fc_sim:
            best_fc_sim = sim
            best_fc = fc_id
    print(f"Routing question to super_community={best_sc} and community={best_fc}")
    llm = OpenAILLM(model_name="gpt-4o", model_params={"max_tokens": 2000, "temperature": 0})
    retriever = VectorRetriever(driver, index_name, embedding_model)
    graph_rag = GraphRAG(retriever=retriever, llm=llm)
    if best_fc is None:
        response = graph_rag.search(query_text=question, retriever_config={"top_k": top_k}, return_context=True)
        return response.answer
    else:
        response = graph_rag.search(query_text=question, retriever_config={"top_k": top_k, "filters": {"community": best_fc}}, return_context=True)
        return response.answer

###########################
# HELPER: Evaluate correctness via GPT
###########################
def evaluate_correctness_via_gpt(candidate_answer, ground_truth):
    check_prompt = [
        {"role": "system", "content": ("Check if the candidate answer means the same as the ground truth. Print 1 if yes, else 0.\n" f"Ground Truth: {ground_truth}")},
        {"role": "user", "content": f"Answer: {candidate_answer}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=check_prompt,
            temperature=0,
            max_tokens=256
        )
        verdict_str = response.choices[0].message.content.strip()
        verdict = int(verdict_str)
    except Exception:
        verdict = 0
    return verdict

###########################
# ELASTICSEARCH SETUP
###########################
ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
es_client = Elasticsearch(ES_URL, basic_auth=(ES_USERNAME, ES_PASSWORD), verify_certs=False, request_timeout=3600)
es_embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

###########################
# CSV + PDF Approach
###########################
def run_csv_pdf_experiment(pdf_file_path, csv_path, use_es, top_k):
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    embedding_model = SentenceTransformerEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")
    dims = embedding_model.model.get_sentence_embedding_dimension()
    index_name = "pdf_vector_index"
    es_index_name = "rag_pdf_index_temp"
    llm = OpenAILLM(model_name="gpt-4o", model_params={"max_tokens": 2000, "temperature": 0})
    pdf_text = asyncio.run(_extract_text_from_pdf_async(pdf_file_path))
    _clear_neo4j_chunks_and_index(driver, index_name)
    try:
        es_client.indices.delete(index=es_index_name)
    except:
        pass
    _build_graph_async(driver, pdf_file_path, llm, embedding_model, dims, index_name)
    # --- Change: split the PDF text into smaller chunks ---
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separator='')
    pdf_chunks_for_es = splitter.split_text(pdf_text)
    vector_store = ElasticsearchStore.from_texts(
        texts=pdf_chunks_for_es,
        embedding=es_embedding_model,
        es_connection=es_client,
        index_name=es_index_name
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    graph_rag_score = 0
    es_rag_score = 0
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row["question"]
            ground_truth = row["reference"]
            print("\n=============================================================")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            candidate_answer = answer_question(question, driver, index_name, embedding_model, top_k)
            print(f"GraphRAG Candidate Answer: {candidate_answer}")
            if use_es:
                retrieved_docs = retriever.invoke(question)
                es_context = "\n".join([doc.page_content[:1000] for doc in retrieved_docs])
                es_prompt = f"""
Based on the provided context, answer the question.
Keep numeric or factual details verbatim.

Context:
{es_context}

Question: {question}
"""
                system_msg_es = "You are a query-focused summarizer. Generate an answer based solely on the provided context."
                es_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_msg_es},
                        {"role": "user", "content": es_prompt},
                    ],
                    temperature=0,
                    max_tokens=512
                )
                es_candidate_answer = es_response.choices[0].message.content
                print(f"Elasticsearch RAG Candidate Answer: {es_candidate_answer}")
                es_verdict = evaluate_correctness_via_gpt(es_candidate_answer, ground_truth)
            else:
                print("Elasticsearch assessment disabled.")
                es_candidate_answer = "Not evaluated"
                es_verdict = 0
            graph_verdict = evaluate_correctness_via_gpt(candidate_answer, ground_truth)
            graph_rag_score += graph_verdict
            es_rag_score += es_verdict
            print(f"GraphRAG Correctness (1/0): {graph_verdict}")
            print(f"Elasticsearch RAG Correctness (1/0): {es_verdict}")
            print(f"Cumulative GraphRAG Score: {graph_rag_score}")
            print(f"Cumulative Elasticsearch RAG Score: {es_rag_score}")
            print("=============================================================")
    try:
        es_client.indices.delete(index=es_index_name)
    except:
        pass
    _clear_neo4j_chunks_and_index(driver, index_name)
    driver.close()
    print(f"\nFinal GraphRAG Score across CSV: {graph_rag_score}")
    print(f"Final Elasticsearch RAG Score across CSV: {es_rag_score}")

###########################
# Dataset Approach
###########################
def run_dataset_experiment(dataset_name, use_es, top_k):
    from datasets import load_dataset
    ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    embedding_model = SentenceTransformerEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")
    dims = embedding_model.model.get_sentence_embedding_dimension()
    index_name = "dataset_vector_index"
    llm = OpenAILLM(model_name="gpt-4o", model_params={"max_tokens": 2000, "temperature": 0})
    es_index_name = "rag_dataset_index_temp"
    es_rag_score = 0
    graph_rag_score = 0
    for x in ragbench_dataset:
        docs = x["documents"]
        question = x["question"]
        ground_truth = x["response"]
        print("\n=================================")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        _clear_neo4j_chunks_and_index(driver, index_name)
        _build_graph_async_from_docs(driver, docs, llm, embedding_model, dims, index_name)
        try:
            es_client.indices.delete(index=es_index_name)
        except:
            pass
        vector_store = ElasticsearchStore.from_texts(
            texts=docs,
            embedding=es_embedding_model,
            es_connection=es_client,
            index_name=es_index_name
        )
        es_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        if use_es:
            retrieved_docs = es_retriever.invoke(question)
            es_context = "\n".join([doc.page_content[:1000] for doc in retrieved_docs])
            es_prompt = f"""
You are a query-focused summarizer. Use the context below to answer the question. 
Keep numeric or factual details verbatim.

Context:
{es_context}

Question: {question}
"""
            es_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a query-focused summarizer."},
                    {"role": "user", "content": es_prompt},
                ],
                temperature=0,
                max_tokens=512
            )
            es_candidate_answer = es_response.choices[0].message.content
            print("Elasticsearch RAG Answer:", es_candidate_answer)
            es_verdict = evaluate_correctness_via_gpt(es_candidate_answer, ground_truth)
        else:
            print("Elasticsearch assessment disabled.")
            es_candidate_answer = "Not evaluated"
            es_verdict = 0
        candidate_answer = answer_question(question, driver, index_name, embedding_model, top_k)
        print("GraphRAG Answer:", candidate_answer)
        graph_verdict = evaluate_correctness_via_gpt(candidate_answer, ground_truth)
        es_rag_score += es_verdict
        graph_rag_score += graph_verdict
        print(f"ES RAG Verdict: {es_verdict}")
        print(f"GraphRAG Verdict: {graph_verdict}")
        _clear_neo4j_chunks_and_index(driver, index_name)
        try:
            es_client.indices.delete(index=es_index_name)
        except:
            pass
        print("=================================")
    driver.close()
    print("\nFinal Scores across the dataset:")
    print(f"Total Elasticsearch RAG Score: {es_rag_score}")
    print(f"Total GraphRAG Score: {graph_rag_score}")

###########################
# ENTRY POINT WITH BOOLEAN VARIABLES
###########################
if __name__ == "__main__":
    use_datasets = False   # Change to True for dataset-based approach
    use_es = False          # Set to False to disable Elasticsearch assessment
    top_k = 8              # Number of chunks to be retrieved for both Elasticsearch and GraphRAG
    # New variables to control chunking behavior:
    chunk_size = 2000      # Desired chunk size for splitting text
    chunk_overlap = 0     # Desired overlap between chunks
    # Override global chunking settings
    CHUNK_SIZE = chunk_size
    CHUNK_OVERLAP = chunk_overlap

    pdf_file_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/merged.pdf"
    csv_file_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/QnA_combined_20241103.csv"
    dataset_name = "covidqa"
    if use_datasets:
        run_dataset_experiment(dataset_name, use_es, top_k)
    else:
        run_csv_pdf_experiment(pdf_file_path, csv_file_path, use_es, top_k)
