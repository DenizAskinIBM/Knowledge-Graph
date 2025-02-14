import sys
import atexit
import os
import asyncio
import itertools
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import openai
from openai import OpenAI
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Neo4j + GraphRAG
from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever  # Query-based retrieval from Neo4j
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists

# Clustering
from sklearn.neighbors import NearestNeighbors
import igraph
import leidenalg

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

##############################
# Additional LLM Imports
##############################
from langchain_openai import ChatOpenAI  # For IBM-compatible code usage
from langchain_ibm import ChatWatsonx

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Provide fallback values for environment variables to avoid Pydantic validation issues
url = os.getenv("WATSONX_URL") or "https://default-watsonx-url"
apikey = os.getenv("API_KEY") or "your-api-key"
project_id = os.getenv("PROJECT_ID") or "your-project-id"
openai_apikey = os.getenv("OPENAI_API_KEY") or "your-openai-api-key"

model_id_llama   = "meta-llama/llama-3-405b-instruct"
model_id_mistral = "mistralai/mixtral-8x7b-instruct-v01"
model_id_code    = "ibm/granite-34b-code-instruct"
model_id_granite = "ibm/granite-3-8b-instruct"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42
}

llm_code = ChatWatsonx(
    model_id=model_id_code,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)
llm_mistral = ChatWatsonx(
    model_id=model_id_mistral,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)
llm_granite = ChatWatsonx(
    model_id=model_id_granite,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)
llm_llama = ChatWatsonx(
    model_id=model_id_llama,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)
llm_chat_gpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_apikey
)

from typing import Optional, Any
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
import json
from langchain.schema import HumanMessage

# --- Tee Class for Logging Printed Output ---
class Tee:
    def __init__(self, file, stream):
        self.file = file      # The file to write to.
        self.stream = stream  # The original stdout (terminal).

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()

output_file = open("es-vs.graphrag-granite-BMO.txt", "w")
original_stdout = sys.stdout
sys.stdout = Tee(output_file, original_stdout)
atexit.register(lambda: output_file.close())
# --- End Tee Setup ---

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CustomLLM(LLMInterface):
    """
    Custom implementation of LLMInterface for integration with SimpleKGPipeline.
    Wraps a ChatWatsonx or similar non-OpenAI LLM.
    """

    def __init__(
        self,
        watsonx_llm,  # A ChatWatsonx instance
        model_name: str = "ibm/granite-3-8b-instruct",
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, model_params, **kwargs)
        self.watsonx_llm = watsonx_llm

    def invoke(self, input: str) -> LLMResponse:
        try:
            result = self.watsonx_llm.invoke([HumanMessage(content=input)])
            response_text = result.content
            response_dict = {"response": response_text}
            response_content = json.dumps(response_dict)
            return LLMResponse(content=response_content)
        except Exception as e:
            raise LLMGenerationError(f"Failed to generate response: {e}")

    async def ainvoke(self, input: str) -> LLMResponse:
        try:
            loop = asyncio.get_running_loop()
            def run_sync():
                res = self.watsonx_llm.invoke([HumanMessage(content=input)])
                return res.content
            response_text = await loop.run_in_executor(None, run_sync)
            response_dict = {"response": response_text}
            response_content = json.dumps(response_dict)
            return LLMResponse(content=response_content)
        except Exception as e:
            raise LLMGenerationError(f"Failed to generate response asynchronously: {e}")

###########################
# GLOBAL SETTINGS
###########################
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
LEIDEN_RESOLUTION = 3.0
NEIGHBORHOOD_K = 4

###########################
# ASYNC HELPERS
###########################
async def _extract_text_from_pdf_async(pdf_path):
    from PyPDF2 import PdfReader

    def _read_pdf(path):
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        text = await loop.run_in_executor(pool, _read_pdf, pdf_path)
    return text

async def _async_invoke_llm(llm, prompt):
    loop = asyncio.get_running_loop()
    def _sync_invoke():
        return llm.invoke(prompt)
    return await loop.run_in_executor(None, _sync_invoke)

async def _generate_name_for_text_async(llm, text, prompt="Provide a short factual summary of up to 50 words for the chunk's content. Focus on key details:"):
    if not text.strip():
        return "NoContent"
    full_prompt = f"""{prompt}
Text:
{text}

Summary:"""
    response = await _async_invoke_llm(llm, full_prompt)
    return response.content.strip()

async def _generate_name_for_group_async(llm, texts, group_prompt="Name this community based on the following chunks' content:"):
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
    print(f"Cleared existing Chunks and dropped index '{index_name}' from Neo4j.")

###########################
# BUILD THE KNOWLEDGE GRAPH (FROM PDF)
# Using 'RELATED' relationships
###########################
async def _ingest_pdf_async(
    driver,
    pdf_path,
    llm,
    embedding_model,
    dims,
    index_name
):
    print("------------------------------------------------------")
    print("📌 Starting ingestion from PDF:", pdf_path)
    print("------------------------------------------------------")

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedding_model,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=FixedSizeSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    )

    print("[1/8] Extracting text from PDF...")
    pdf_text = await _extract_text_from_pdf_async(pdf_path)

    print("[2/8] Splitting and storing chunks in Neo4j...")
    await kg_builder.run_async(text=pdf_text)

    print("[3/8] Creating Neo4j vector index...")
    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=dims,
        similarity_fn="cosine",
    )

    # Delete empty embeddings or text
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL OR c.text IS NULL OR c.text = ''
            DETACH DELETE c
        """)

    # Ensure each chunk has a uuid
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.uuid IS NULL
            SET c.uuid = elementId(c)
        """)

    # Retrieve chunk data
    with driver.session() as session:
        results = session.run("""
            MATCH (c:Chunk)
            RETURN
                elementId(c) AS nodeId,
                c.embedding AS embedding,
                c.text AS text
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
        print("❗ No valid chunks found after ingestion. Exiting build.")
        return

    print(f"Total valid chunks in Neo4j: {num_nodes}")

    print("[4/8] Running nearest neighbors for similarity graph...")
    nbrs = NearestNeighbors(n_neighbors=NEIGHBORHOOD_K, metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    edges = []
    weights = []
    for i in range(num_nodes):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = max(0, 1 - dist)  # Convert distance -> similarity
            edges.append((i, j_idx))
            weights.append(sim)

    print("[5/8] Running Leiden community detection (first-level)...")
    g = igraph.Graph(n=num_nodes, edges=edges, directed=False)
    g.es["weight"] = weights
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=LEIDEN_RESOLUTION
    )
    community_labels = partition.membership

    print("Deleting existing RELATED relationships, then recreating them in bulk...")
    with driver.session() as session:
        session.run("MATCH ()-[r:RELATED]->() DELETE r")

    edge_data = []
    for i in range(num_nodes):
        idA = node_ids[i]
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = 1 - dist
            idB = node_ids[j_idx]
            edge_data.append({"idA": idA, "idB": idB, "sim": sim})

    # Bulk creation of RELATED edges
    with driver.session() as session:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (a:Chunk { uuid: pair.idA })
            MATCH (b:Chunk { uuid: pair.idB })
            MERGE (a)-[r:RELATED]->(b)
            SET r.score = pair.sim
            """,
            {"pairs": edge_data}
        )

    # Bulk set community
    community_data = []
    for i in range(num_nodes):
        c_comm = int(community_labels[i])
        c_id = node_ids[i]
        community_data.append({"idVal": c_id, "community": c_comm})

    with driver.session() as session:
        session.run(
            """
            UNWIND $communityData AS c
            MATCH (chunk:Chunk { uuid: c.idVal })
            SET chunk.community = c.community
            """,
            {"communityData": community_data}
        )

    print("Assigned 'community' labels to chunks.")

    print("[6/8] Generating short factual summaries for each chunk...")
    async def _name_chunk(i):
        txt = node_text_map[node_ids[i]]
        return await _generate_name_for_text_async(
            llm,
            txt,
            prompt=(
                "Provide a short factual summary of up to 50 words for the chunk's content. "
                "Focus on key details and context for improved retrieval-based QA."
            )
        )

    chunk_name_tasks = [_name_chunk(i) for i in range(num_nodes)]
    chunk_names = await asyncio.gather(*chunk_name_tasks)

    name_data = []
    for i in range(num_nodes):
        name_data.append({
            "idVal": node_ids[i],
            "summaryVal": chunk_names[i]
        })

    # Append the short summary to the chunk's text property for better retrieval
    with driver.session() as session:
        for nd in name_data:
            session.run(
                """
                MATCH (chunk:Chunk { uuid: $idVal })
                SET chunk.name = $summaryVal,
                    chunk.text = chunk.text + '\\n\\nShortSummary: ' + $summaryVal
                """,
                {"idVal": nd["idVal"], "summaryVal": nd["summaryVal"]}
            )

    print("Chunk summarization complete.")

    print("[7/8] Generating first-level community names and centroids...")
    community_labels = partition.membership  # reuse from above
    community_to_nodes = defaultdict(list)
    for i, nid in enumerate(node_ids):
        c_label = community_labels[i]
        community_to_nodes[c_label].append(nid)

    async def _name_community(comm_id, nodelist):
        chunk_texts = [node_text_map[nid] for nid in nodelist]
        return await _generate_name_for_group_async(
            llm, chunk_texts,
            group_prompt="Name this first-level community based on its chunk texts:"
        )

    comm_tasks = []
    for cid, nlist in community_to_nodes.items():
        comm_tasks.append((cid, _name_community(cid, nlist)))

    comm_results = await asyncio.gather(*[t[1] for t in comm_tasks])
    community_name_map = {}
    for i, (comm_id, _) in enumerate(comm_tasks):
        community_name_map[comm_id] = comm_results[i]

    for cid, c_name in community_name_map.items():
        with driver.session() as session:
            session.run(
                """
                MATCH (c:Chunk)
                WHERE c.community = $commId
                SET c.community_name = $cName
                """,
                {"commId": cid, "cName": c_name}
            )

    # Compute centroids
    community_centroids = {}
    for comm_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        community_centroids[comm_id] = np.mean(subset_emb, axis=0)

    print("First-level communities named.")

    print("[8/8] Creating second-level super_communities + naming them...")
    comm_ids = sorted(community_centroids.keys())
    comm_index = {c: idx for idx, c in enumerate(comm_ids)}
    num_comms = len(comm_ids)

    edges2, weights2 = [], []
    for c1, c2 in itertools.combinations(comm_ids, 2):
        v1 = community_centroids[c1]
        v2 = community_centroids[c2]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        sim = (np.dot(v1, v2) / denom) if denom > 1e-12 else 0
        sim = max(0, sim)
        edges2.append((comm_index[c1], comm_index[c2]))
        weights2.append(sim)

    g2 = igraph.Graph(n=num_comms, edges=edges2, directed=False)
    g2.es["weight"] = weights2
    partition2 = leidenalg.find_partition(
        g2,
        leidenalg.RBConfigurationVertexPartition,
        weights=g2.es["weight"],
        resolution_parameter=LEIDEN_RESOLUTION
    )
    super_community_labels = partition2.membership
    community_to_super = {comm_ids[i]: super_community_labels[i] for i in range(num_comms)}

    sc_data = []
    for i in range(num_nodes):
        c_comm = community_labels[i]
        sc = community_to_super[c_comm]
        c_id = node_ids[i]
        sc_data.append({"idVal": c_id, "sc": sc})

    with driver.session() as session:
        session.run(
            """
            UNWIND $scData AS x
            MATCH (c:Chunk { uuid: x.idVal })
            SET c.super_community = x.sc
            """,
            {"scData": sc_data}
        )

    super_comm_to_first_comm = defaultdict(list)
    for fc in comm_ids:
        sc = community_to_super[fc]
        super_comm_to_first_comm[sc].append(fc)

    async def _name_super_comm(sc, fc_list):
        fc_names = [community_name_map[fc] for fc in fc_list]
        return await _generate_name_for_group_async(
            llm, fc_names,
            group_prompt="Name this second-level community based on its first-level community names:"
        )

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
            sc_name = super_community_name_map[sc]
            c_id = node_ids[i]
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.super_community_name = $scName
            """, {"idVal": c_id, "scName": sc_name})

    print("✅ Knowledge Graph built successfully from PDF.")
    print("------------------------------------------------------\n")

def _build_graph_async(driver, pdf_path, llm, embedding_model, dims, index_name):
    asyncio.run(
        _ingest_pdf_async(
            driver=driver,
            pdf_path=pdf_path,
            llm=llm,
            embedding_model=embedding_model,
            dims=dims,
            index_name=index_name
        )
    )

###########################
# BUILD THE KNOWLEDGE GRAPH (FROM DATASET DOCUMENTS)
# Using 'RELATED' relationships
###########################
async def _ingest_docs_async(
    driver,
    docs,
    llm,
    embedding_model,
    dims,
    index_name
):
    print("------------------------------------------------------")
    print("📌 Starting ingestion from a documents list...")
    print("------------------------------------------------------")

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedding_model,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=FixedSizeSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
    )

    print("[1/8] Combining documents and ingesting into Neo4j chunks...")
    dataset_text = "\n\n".join(docs)
    await kg_builder.run_async(text=dataset_text)

    print("[2/8] Creating Neo4j vector index...")
    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=dims,
        similarity_fn="cosine",
    )

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
            RETURN
                elementId(c) AS nodeId,
                c.embedding AS embedding,
                c.text AS text
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
        print("❗ No valid chunks found. Exiting build.")
        return

    print(f"Total valid chunks in Neo4j: {num_nodes}")

    print("[3/8] Running nearest neighbors for similarity graph...")
    nbrs = NearestNeighbors(n_neighbors=NEIGHBORHOOD_K, metric="cosine")
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

    print("[4/8] Running Leiden community detection (first-level)...")
    g = igraph.Graph(n=num_nodes, edges=edges, directed=False)
    g.es["weight"] = weights
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=LEIDEN_RESOLUTION
    )
    community_labels = partition.membership

    print("Deleting existing RELATED relationships, then recreating them in bulk...")
    with driver.session() as session:
        session.run("MATCH ()-[r:RELATED]->() DELETE r")

    edge_data = []
    for i in range(num_nodes):
        idA = node_ids[i]
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = 1 - dist
            idB = node_ids[j_idx]
            edge_data.append({"idA": idA, "idB": idB, "sim": sim})

    with driver.session() as session:
        session.run(
            """
            UNWIND $pairs AS pair
            MATCH (a:Chunk { uuid: pair.idA })
            MATCH (b:Chunk { uuid: pair.idB })
            MERGE (a)-[r:RELATED]->(b)
            SET r.score = pair.sim
            """,
            {"pairs": edge_data}
        )

    community_data = []
    for i in range(num_nodes):
        c_comm = int(community_labels[i])
        c_id = node_ids[i]
        community_data.append({"idVal": c_id, "community": c_comm})

    with driver.session() as session:
        session.run(
            """
            UNWIND $communityData AS c
            MATCH (chunk:Chunk { uuid: c.idVal })
            SET chunk.community = c.community
            """,
            {"communityData": community_data}
        )

    print("Assigned 'community' labels to chunks.")

    print("[5/8] Generating short factual summaries for each chunk...")
    async def _name_chunk(i):
        txt = node_text_map[node_ids[i]]
        return await _generate_name_for_text_async(
            llm,
            txt,
            prompt=(
                "Provide a short factual summary of up to 50 words for the chunk's content. "
                "Focus on key details and context for improved retrieval-based QA."
            )
        )

    chunk_name_tasks = [_name_chunk(i) for i in range(num_nodes)]
    chunk_names = await asyncio.gather(*chunk_name_tasks)

    for i in range(num_nodes):
        with driver.session() as session:
            session.run(
                """
                MATCH (chunk:Chunk { uuid: $idVal })
                SET chunk.name = $summaryVal,
                    chunk.text = chunk.text + '\\n\\nShortSummary: ' + $summaryVal
                """,
                {"idVal": node_ids[i], "summaryVal": chunk_names[i]}
            )

    print("Chunk summarization complete.")

    print("[6/8] Generating first-level community names + centroids...")
    community_labels = partition.membership
    community_to_nodes = defaultdict(list)
    for i, nid in enumerate(node_ids):
        c_label = community_labels[i]
        community_to_nodes[c_label].append(nid)

    async def _name_community(comm_id, nodelist):
        chunk_texts = [node_text_map[nid] for nid in nodelist]
        return await _generate_name_for_group_async(
            llm, chunk_texts,
            group_prompt="Name this first-level community based on its chunk texts:"
        )

    comm_tasks = []
    for cid, nlist in community_to_nodes.items():
        comm_tasks.append((cid, _name_community(cid, nlist)))

    comm_results = await asyncio.gather(*[t[1] for t in comm_tasks])
    community_name_map = {}
    for i, (comm_id, _) in enumerate(comm_tasks):
        community_name_map[comm_id] = comm_results[i]

    for cid, c_name in community_name_map.items():
        with driver.session() as session:
            session.run(
                """
                MATCH (c:Chunk)
                WHERE c.community = $commId
                SET c.community_name = $cName
                """,
                {"commId": cid, "cName": c_name}
            )

    community_centroids = {}
    for comm_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        community_centroids[comm_id] = np.mean(subset_emb, axis=0)

    print("First-level communities named.")

    print("[7/8] Creating second-level 'super_communities' with Leiden + naming them...")
    comm_ids = sorted(community_centroids.keys())
    comm_index = {c: idx for idx, c in enumerate(comm_ids)}
    num_comms = len(comm_ids)

    edges2, weights2 = [], []
    for c1, c2 in itertools.combinations(comm_ids, 2):
        v1 = community_centroids[c1]
        v2 = community_centroids[c2]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        sim = (np.dot(v1, v2) / denom) if denom > 1e-12 else 0
        sim = max(0, sim)
        edges2.append((comm_index[c1], comm_index[c2]))
        weights2.append(sim)

    g2 = igraph.Graph(n=num_comms, edges=edges2, directed=False)
    g2.es["weight"] = weights2
    partition2 = leidenalg.find_partition(
        g2,
        leidenalg.RBConfigurationVertexPartition,
        weights=g2.es["weight"],
        resolution_parameter=LEIDEN_RESOLUTION
    )
    super_community_labels = partition2.membership
    community_to_super = {
        comm_ids[i]: super_community_labels[i] for i in range(num_comms)
    }

    sc_data = []
    for i in range(num_nodes):
        c_comm = community_labels[i]
        sc = community_to_super[c_comm]
        c_id = node_ids[i]
        sc_data.append({"idVal": c_id, "sc": sc})

    with driver.session() as session:
        session.run(
            """
            UNWIND $scData AS x
            MATCH (c:Chunk { uuid: x.idVal })
            SET c.super_community = x.sc
            """,
            {"scData": sc_data}
        )

    super_comm_to_first_comm = defaultdict(list)
    for fc in comm_ids:
        sc = community_to_super[fc]
        super_comm_to_first_comm[sc].append(fc)

    async def _name_super_comm(sc, fc_list):
        fc_names = [community_name_map[fc] for fc in fc_list]
        return await _generate_name_for_group_async(
            llm, fc_names,
            group_prompt="Name this second-level community based on its first-level community names:"
        )

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
            sc_name = super_community_name_map[sc]
            c_id = node_ids[i]
            session.run("""
                MATCH (c:Chunk { uuid: $idVal })
                SET c.super_community_name = $scName
            """, {"idVal": c_id, "scName": sc_name})

    print("✅ Knowledge Graph built successfully (from given documents).")
    print("------------------------------------------------------\n")

def _build_graph_async_from_docs(driver, docs, llm, embedding_model, dims, index_name):
    asyncio.run(
        _ingest_docs_async(
            driver=driver,
            docs=docs,
            llm=llm,
            embedding_model=embedding_model,
            dims=dims,
            index_name=index_name
        )
    )

############################################################
# Helper functions for retrieving chunk neighbors
############################################################
def find_chunk_uuid_by_text(driver, chunk_text):
    """
    Returns the UUID of the chunk that exactly matches 'chunk_text'.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.text = $chunk_text
            RETURN c.uuid AS uuid
            LIMIT 1
        """, {"chunk_text": chunk_text})
        record = result.single()
        if record is None:
            return None
        return record["uuid"]

def get_top_neighbor_chunk(driver, chunk_uuid):
    """
    Finds the single highest-related neighbor chunk based on 'RELATED' relationship score.
    Returns a dict containing {"text": ..., "score": ...} or None if none found.
    """
    with driver.session() as session:
        rec = session.run("""
            MATCH (c:Chunk {uuid: $chunkUuid})-[r:RELATED]->(n:Chunk)
            RETURN n.text AS textVal, r.score AS sim
            ORDER BY sim DESC
            LIMIT 1
        """, {"chunkUuid": chunk_uuid}).single()

    if rec is None:
        return None
    return {"text": rec["textVal"], "score": rec["sim"]}

###########################
# ANSWER A QUESTION (GRAPH RAG) + neighbor-based retrieval expansion
###########################
def answer_question(question, driver, index_name, embedding_model, retrieval_k, llm=None):
    """
    Retrieves chunks from the knowledge graph using a multi-step retrieval process.
    The final number of chunks returned will be `retrieval_k`.
    """
    # Gather chunk metadata
    with driver.session() as session:
        records = list(session.run("""
            MATCH (c:Chunk)
            RETURN
                elementId(c) AS nodeId,
                c.embedding AS embedding,
                c.text AS text,
                c.community AS community,
                c.community_name AS communityName,
                c.super_community AS superComm,
                c.super_community_name AS superCommunityName
        """))

    if not records:
        print("No Chunks found in the DB, cannot answer question.")
        return "No data available."

    node_ids = []
    embeddings = []
    community_labels = []
    super_community_labels = []
    text_map = {}

    for r in records:
        node_ids.append(r["nodeId"])
        embeddings.append(r["embedding"])
        community_labels.append(r["community"])
        super_community_labels.append(r["superComm"])
        text_map[r["nodeId"]] = r["text"]

    X = np.array(embeddings)

    # Build mapping from community to node_ids
    community_to_nodes = defaultdict(list)
    for i, nid in enumerate(node_ids):
        c_label = community_labels[i]
        community_to_nodes[c_label].append(nid)

    # Compute first-level community centroids
    community_centroids = {}
    for c_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        community_centroids[c_id] = np.mean(subset_emb, axis=0)

    # Group first-level communities by second-level super_community
    super_comm_to_first_comm = defaultdict(list)
    for i, fc in enumerate(community_labels):
        sc = super_community_labels[i]
        if fc not in super_comm_to_first_comm[sc]:
            super_comm_to_first_comm[sc].append(fc)

    # 1) Compute question embedding
    question_embedding = np.array(embedding_model.embed_query(question))

    # 2) Pick top super communities.
    # Here we set the number to a fraction of retrieval_k (e.g., retrieval_k//4)
    top_k_sc = max(1, retrieval_k // 4)
    sc_sims = []
    # Compute centroid of each super community
    super_community_centroids = {}
    sc_ids = sorted(list(super_comm_to_first_comm.keys()))
    for sc_id in sc_ids:
        fc_ids = super_comm_to_first_comm[sc_id]
        sc_embs = []
        for fc_id in fc_ids:
            idxs = [node_ids.index(nid) for nid in community_to_nodes[fc_id]]
            for idx in idxs:
                sc_embs.append(X[idx])
        if sc_embs:
            super_community_centroids[sc_id] = np.mean(sc_embs, axis=0)
        else:
            super_community_centroids[sc_id] = np.zeros_like(X[0])

    for sc_id, sc_centroid in super_community_centroids.items():
        denom = np.linalg.norm(question_embedding) * np.linalg.norm(sc_centroid)
        sim = (np.dot(question_embedding, sc_centroid) / denom) if denom > 1e-12 else 0
        sc_sims.append((sc_id, sim))

    sc_sims.sort(key=lambda x: x[1], reverse=True)
    chosen_super_comms = sc_sims[:top_k_sc]

    # 3) For each chosen super community, pick top first-level communities.
    top_k_fc = max(1, retrieval_k // 4)
    candidate_communities = set()
    for (sc_id, _) in chosen_super_comms:
        fc_sims = []
        for fc_id in super_comm_to_first_comm[sc_id]:
            fc_centroid = community_centroids[fc_id]
            denom = np.linalg.norm(question_embedding) * np.linalg.norm(fc_centroid)
            sim = (np.dot(question_embedding, fc_centroid)/denom) if denom > 1e-12 else 0
            fc_sims.append((fc_id, sim))
        fc_sims.sort(key=lambda x: x[1], reverse=True)
        top_fc_sims = fc_sims[:top_k_fc]
        for (fc_id, _) in top_fc_sims:
            candidate_communities.add(fc_id)

    print(f"Routing question to super communities: {[x[0] for x in chosen_super_comms]}")
    print(f"Chosen first-level communities: {candidate_communities}")

    # 4) For each candidate community, retrieve chunks.
    retriever = VectorRetriever(driver, index_name, embedding_model)
    all_results = []

    # We use retrieval_k//2 for the per-community retrieval (adjust as desired)
    per_comm_k = max(1, retrieval_k // 2)
    for fc_id in candidate_communities:
        fc_results_generator = retriever.get_search_results(
            query_text=question,
            top_k=per_comm_k,
            filters={"community": fc_id}
        )
        fc_results_list = []
        for result in fc_results_generator:
            # The result can be a tuple or a dict; convert accordingly.
            if isinstance(result, tuple):
                text_val, score_val = result
            elif isinstance(result, dict):
                text_val = result.get("text", "")
                score_val = result.get("score", 0)
            else:
                text_val, score_val = str(result), 0
            fc_results_list.append({"text": text_val, "score": score_val})

        # For each retrieved chunk, also add its top neighbor
        fc_expanded_results = []
        for item in fc_results_list:
            fc_expanded_results.append(item)  # the chunk itself
            chunk_uuid = find_chunk_uuid_by_text(driver, item["text"])
            if chunk_uuid:
                neighbor = get_top_neighbor_chunk(driver, chunk_uuid)
                if neighbor:
                    # Ensure neighbor is in dict format with a valid score.
                    if isinstance(neighbor, tuple):
                        n_text, n_score = neighbor
                        neighbor = {"text": n_text, "score": n_score if isinstance(n_score, (int, float)) else 0}
                    elif isinstance(neighbor, dict):
                        n_score = neighbor.get("score", 0)
                        neighbor["score"] = n_score if isinstance(n_score, (int, float)) else 0
                    else:
                        neighbor = {"text": str(neighbor), "score": 0}
                    fc_expanded_results.append(neighbor)
        all_results.extend(fc_expanded_results)
    
    # Robust conversion of all_results to ensure each item is a dict with a numeric "score"
    def safe_convert(result):
        if isinstance(result, tuple):
            text_val = result[0] if isinstance(result[0], str) else str(result[0])
            score_val = result[1] if isinstance(result[1], (int, float)) else 0
            return {"text": text_val, "score": score_val}
        elif isinstance(result, dict):
            text_val = result.get("text", "")
            score_val = result.get("score", 0)
            if not isinstance(score_val, (int, float)):
                score_val = 0
            return {"text": text_val, "score": score_val}
        else:
            return {"text": str(result), "score": 0}
    all_results = [safe_convert(r) for r in all_results]

    # Sort final results by similarity
    all_results.sort(key=lambda x: x["score"], reverse=True)
    final_contexts = all_results[:retrieval_k]

    # Build final context prompt
    context_text = "\n".join([res["text"] for res in final_contexts])
    prompt = (
        "Use only the following context to answer the question as accurately as possible.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    if llm is None:
        print("No LLM provided! Returning raw context only.")
        return context_text
    else:
        response = llm.invoke(prompt)
        return response.content.strip()

###########################
# HELPER: Evaluate correctness via LLM judge
###########################
def evaluate_correctness_via_gpt(candidate_answer, ground_truth, llm_judge=None):
    check_prompt = (
        "Check if the candidate answer means the same as the ground truth. "
        "Print 1 if yes, else 0.\n"
        f"Ground Truth: {ground_truth}\n"
        f"Answer: {candidate_answer}"
    )
    if llm_judge is not None:
        try:
            response = llm_judge.invoke(check_prompt)
            verdict_str = response.content.strip()
            verdict = int(verdict_str)
        except Exception as e:
            verdict = 0
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    "Check if the candidate answer means the same as the ground truth. "
                    "Print 1 if yes, else 0."
                )
            },
            {
                "role": "user",
                "content": f"Ground Truth: {ground_truth}\nAnswer: {candidate_answer}"
            }
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0,
                max_tokens=256
            )
            verdict_str = response.choices[0].message.content.strip()
            verdict = int(verdict_str)
        except Exception:
            verdict = 0
    return verdict

###########################
# EXPERIMENT FUNCTIONS (Modified to accept llm_judge and retrieval_k)
###########################
def run_csv_pdf_experiment(pdf_file_path, csv_path, llm, llm_judge, retrieval_k):
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    embedding_model = SentenceTransformerEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")
    dims = embedding_model.model.get_sentence_embedding_dimension()
    index_name = "pdf_vector_index"
    es_index_name = "rag_pdf_index_temp"

    print("\nExtracting PDF text before ingestion into ES or Graph...")
    pdf_text = asyncio.run(_extract_text_from_pdf_async(pdf_file_path))

    # Build Graph RAG
    _clear_neo4j_chunks_and_index(driver, index_name)
    _build_graph_async(driver, pdf_file_path, llm, embedding_model, dims, index_name)

    # Prepare Elasticsearch indexing
    ES_URL = os.getenv("ES_URL")
    ES_USERNAME = os.getenv("ES_USERNAME")
    ES_PASSWORD = os.getenv("ES_PASSWORD")
    es_client = Elasticsearch(
        ES_URL,
        basic_auth=(ES_USERNAME, ES_PASSWORD),
        verify_certs=False,
        request_timeout=3600,
    )

    try:
        es_client.indices.delete(index=es_index_name)
    except:
        pass

    from langchain.text_splitter import CharacterTextSplitter
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separator=''
    )
    pdf_chunks_for_es = splitter.split_text(pdf_text)

    es_embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    vector_store = ElasticsearchStore.from_texts(
        texts=pdf_chunks_for_es,
        embedding=es_embedding_model,
        es_connection=es_client,
        index_name=es_index_name
    )
    # Use the same retrieval_k for Elasticsearch
    retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})

    import csv
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

            # Graph RAG answer using the global retrieval_k
            candidate_answer = answer_question(
                question,
                driver,
                index_name,
                embedding_model,
                retrieval_k,
                llm=llm
            )

            # Elasticsearch-based retrieval
            retrieved_docs = retriever.get_relevant_documents(question)
            es_context = "\n".join([doc.page_content for doc in retrieved_docs])
            es_prompt = f"""
Based on the provided context, answer the question. Keep numeric or factual details verbatim.

Context:
{es_context}

Question: {question}
"""
            system_msg_es = "You are a query answerer. Generate an answer based solely on the provided context."
            try:
                invoked_es_answer = llm.invoke(system_msg_es + "\n\n" + es_prompt)
                es_candidate_answer = invoked_es_answer.content
            except:
                es_candidate_answer = "(No ES answer)"

            try:
                print(f"GraphRAG Candidate Answer: {candidate_answer["response"]}")
            except:
                print(f"GraphRAG Candidate Answer: {candidate_answer}")
            print(f"Elasticsearch RAG Candidate Answer: {es_candidate_answer}")

            graph_verdict = evaluate_correctness_via_gpt(candidate_answer, ground_truth, llm_judge)
            es_verdict = evaluate_correctness_via_gpt(es_candidate_answer, ground_truth, llm_judge)

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

def run_dataset_experiment(dataset_name, llm, llm_judge, retrieval_k):
    from datasets import load_dataset
    ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    embedding_model = SentenceTransformerEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1")
    dims = embedding_model.model.get_sentence_embedding_dimension()
    index_name = "dataset_vector_index"
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

        ES_URL = os.getenv("ES_URL")
        ES_USERNAME = os.getenv("ES_USERNAME")
        ES_PASSWORD = os.getenv("ES_PASSWORD")
        es_client = Elasticsearch(
            ES_URL,
            basic_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=False,
            request_timeout=3600,
        )

        try:
            es_client.indices.delete(index=es_index_name)
        except:
            pass

        vector_store = ElasticsearchStore.from_texts(
            texts=docs,
            embedding=HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1"),
            es_connection=es_client,
            index_name=es_index_name
        )
        es_retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})

        retrieved_docs = es_retriever.get_relevant_documents(question)
        es_context = "\n".join([doc.page_content for doc in retrieved_docs])
        es_prompt = f"""
You are a query answerer. Use the context below to answer the question. 
Keep numeric or factual details verbatim.

Context:
{es_context}

Question: {question}
"""
        try:
            invoked_es_answer = llm.invoke(es_prompt)
            es_candidate_answer = invoked_es_answer.content
        except:
            es_candidate_answer = "(No ES answer)"

        print("Elasticsearch RAG Answer:", es_candidate_answer)

        candidate_answer = answer_question(question, driver, index_name, embedding_model, retrieval_k, llm=llm)
        try:
            print("GraphRAG Answer:", candidate_answer["response"])
        except:
            print("GraphRAG Answer:", candidate_answer)
        es_verdict = evaluate_correctness_via_gpt(es_candidate_answer, ground_truth, llm_judge)
        graph_verdict = evaluate_correctness_via_gpt(candidate_answer, ground_truth, llm_judge)

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

###############################
# MAIN ENTRY POINT
###############################
if __name__ == "__main__":
    use_datasets = False
    use_gpt = False
    chosen_model = "granite"

    # Set the global retrieval_k value once.
    retrieval_k = 3  # Change this value as needed

    # Choose the answer-generation LLM
    if use_gpt:
        llm = OpenAILLM(model_name="gpt-4o", model_params={"max_tokens": 2000, "temperature": 0})
    else:
        if chosen_model == "mistral":
            ibm_llm = llm_mistral
            model_name = model_id_mistral
        elif chosen_model == "code":
            ibm_llm = llm_code
            model_name = model_id_code
        elif chosen_model == "granite":
            ibm_llm = llm_granite
            model_name = model_id_granite
        elif chosen_model == "llama":
            ibm_llm = llm_llama
            model_name = model_id_llama
        else:
            ibm_llm = llm_granite
            model_name = model_id_granite

        llm = CustomLLM(
            watsonx_llm=ibm_llm,
            model_name=model_name,
            model_params=parameters
        )

    # Decide which LLM to use for grading (llm_judge)
    llm_judge_choice = "gpt-4o"  # change to "custom" to use the same as answer generation
    if llm_judge_choice == "gpt-4o":
        llm_judge = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=256,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        llm_judge = llm

    pdf_file_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/merged.pdf"
    csv_file_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/QnA_combined_20241103.csv"
    dataset_name = "covidqa"

    if use_datasets:
        run_dataset_experiment(dataset_name, llm, llm_judge, retrieval_k)
    else:
        run_csv_pdf_experiment(pdf_file_path, csv_file_path, llm, llm_judge, retrieval_k)
