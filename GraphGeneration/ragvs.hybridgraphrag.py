# ========================
# BEGIN SINGLE CODE BLOCK
# ========================

import warnings
from urllib3.exceptions import InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

from datasets import load_dataset
import os
import json
import numpy as np
import random
import hashlib
import asyncio

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from langchain import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

# --------------------------
# Load environment variables
# --------------------------
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not openai.api_key:
    raise ValueError("Please set OPENAI_API_KEY in your environment or .env file.")

# --------------------------
# ElasticSearch for RAG
# --------------------------
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")

es_client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    request_timeout=3600,
)

# Same embedding everywhere
embedding_function = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# --------------------------------------------------------------------
# OPTIONAL: Simple Disk Caching to speed up repeated calls
# --------------------------------------------------------------------
import glob
os.makedirs("cache", exist_ok=True)

def get_cache(key):
    cache_file = os.path.join("cache", f"{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return None
    return None

def set_cache(key, data):
    cache_file = os.path.join("cache", f"{key}.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# --------------------------------------------------------------------
# Minimal Async LLM Wrappers
# --------------------------------------------------------------------
class AsyncOpenAIWrapper:
    """
    Minimal asynchronous wrapper around OpenAI ChatCompletion.
    Example usage:
        llm = AsyncOpenAIWrapper("gpt-4", temperature=0)
        response = await llm.invoke("Hello?")
    """
    def __init__(self, model_name="gpt-4", temperature=0):
        self.model_name = model_name
        self.temperature = temperature

    async def invoke(self, prompt: str) -> str:
        response = await aclient.chat.completions.create(model=self.model_name,
        messages=[{"role": "system", "content": prompt}],
        temperature=self.temperature)
        return response["choices"][0]["message"]["content"].strip()

# We will use GPT-3.5 for extraction (faster) and GPT-4 for summarization, Q&A
extract_llm = AsyncOpenAIWrapper(model_name="gpt-3.5-turbo", temperature=0)
summ_llm = AsyncOpenAIWrapper(model_name="gpt-4", temperature=0)
qa_llm   = AsyncOpenAIWrapper(model_name="gpt-4", temperature=0)

# --------------------------
# CHUNKING CONFIG
# --------------------------
MAX_CTX_TOKENS = 8000
SEED = 42

# 1) Adjust chunk size
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 0

# 2) Gleaning
MAX_GLEANINGS = 1

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into one or more chunks."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start += (chunk_size - overlap)
        if start >= len(tokens):
            break
    return chunks

# --------------------------
# Extraction: Entities + Relationships (with caching & GPT-3.5)
# --------------------------
async def async_entity_relationship_extraction_prompt(text_chunk):
    # Use caching
    chunk_hash = hashlib.md5(text_chunk.encode("utf-8")).hexdigest()
    cache_key = f"extract_{chunk_hash}_v1"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached  # return the stored JSON text

    prompt = f"""
You are an information extraction assistant. Given the following text:
\"\"\"{text_chunk}\"\"\"

1) Extract all entities (name, type, description).
2) Extract relationships (source name, target name, relationship description).
Return them as JSON with "entities" and "relationships" keys.

Preserve any numeric or factual details verbatim.
"""
    resp = await aclient.chat.completions.create(model="gpt-3.5-turbo",  # use GPT-3.5 for extraction
    messages=[{"role": "system", "content": prompt}],
    temperature=0)
    content = resp["choices"][0]["message"]["content"]
    set_cache(cache_key, content)
    return content

async def async_glean_missing_entities(text_chunk, known_entities):
    # We can also do a simpler caching if needed
    glean_key_base = text_chunk + "|||".join(known_entities)
    glean_hash = hashlib.md5(glean_key_base.encode("utf-8")).hexdigest()
    glean_cache_key = f"glean_{glean_hash}_v1"
    cached = get_cache(glean_cache_key)
    if cached is not None:
        return cached

    prompt = f"""
In the text below, were there ANY entities missed in the previous extraction?
Answer ONLY with "YES" or "NO".

Text:
\"\"\"{text_chunk}\"\"\"
Already found: {known_entities}
"""
    resp = await aclient.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role":"system","content":prompt}],
    temperature=0,
    max_tokens=30)
    decision = resp["choices"][0]["message"]["content"].strip()

    if decision.upper() == "YES":
        glean_prompt = f"""
Some entities or relationships were missed!
Extract them from the following text:
\"\"\"{text_chunk}\"\"\"
Return them as JSON with "entities" and "relationships" keys,
again preserving numeric/factual data verbatim.
"""
        glean_resp = await aclient.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role":"system","content":glean_prompt}],
        temperature=0)
        glean_content = glean_resp["choices"][0]["message"]["content"]
        set_cache(glean_cache_key, glean_content)
        return glean_content
    else:
        set_cache(glean_cache_key, None)
        return None

async def async_extract_element_instances(text_chunks, gleanings=MAX_GLEANINGS):
    tasks = [async_entity_relationship_extraction_prompt(ch) for ch in text_chunks]
    results = await asyncio.gather(*tasks)

    all_entities = []
    all_relationships = []

    for i, extraction_json in enumerate(results):
        text_chunk = text_chunks[i]
        try:
            extracted = json.loads(extraction_json)
            if isinstance(extracted, dict):
                chunk_ents = extracted.get("entities", [])
                chunk_rels = extracted.get("relationships", [])
                if not (isinstance(chunk_ents, list) and all(isinstance(e, dict) for e in chunk_ents)):
                    chunk_ents = []
                if not (isinstance(chunk_rels, list) and all(isinstance(r, dict) for r in chunk_rels)):
                    chunk_rels = []
            else:
                chunk_ents, chunk_rels = [], []
        except:
            chunk_ents, chunk_rels = [], []

        for _ in range(gleanings):
            known_names = [e.get("name","") for e in chunk_ents]
            gleaned_json = await async_glean_missing_entities(text_chunk, known_names)
            if gleaned_json:
                try:
                    glean_data = json.loads(gleaned_json)
                    if isinstance(glean_data, dict):
                        glean_ents = glean_data.get("entities", [])
                        glean_rels = glean_data.get("relationships", [])
                        if not (isinstance(glean_ents, list) and all(isinstance(e, dict) for e in glean_ents)):
                            glean_ents = []
                        if not (isinstance(glean_rels, list) and all(isinstance(r, dict) for r in glean_rels)):
                            glean_rels = []
                        chunk_ents.extend(glean_ents)
                        chunk_rels.extend(glean_rels)
                except:
                    pass
            else:
                break

        all_entities.extend(chunk_ents)
        all_relationships.extend(chunk_rels)

    return all_entities, all_relationships

# --------------------------
# Summaries: Entities + Relationships (use GPT-4 + caching)
# --------------------------
async def async_summarize_entity(name, entity_type, descriptions):
    joined = "\n".join(descriptions)
    hash_input = name + entity_type + joined
    ent_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
    cache_key = f"ent_sum_{ent_hash}_v1"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    prompt = f"""
Create a concise summary for this entity, preserving numeric/factual details verbatim.

Entity Name: {name}
Entity Type: {entity_type}
Partial Descriptions:
{joined}

Return 2-3 sentences as a single string.
"""
    resp = await aclient.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": prompt}],
    temperature=0)
    result = resp["choices"][0]["message"]["content"].strip()
    set_cache(cache_key, result)
    return result

async def async_summarize_relationship(relationship_description):
    rel_hash = hashlib.md5(relationship_description.encode("utf-8")).hexdigest()
    cache_key = f"rel_sum_{rel_hash}_v1"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    prompt = f"""
Summarize this relationship in 1-2 sentences, preserving any numeric/factual details:
\"\"\"{relationship_description}\"\"\"
"""
    resp = await aclient.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": prompt}],
    temperature=0)
    result = resp["choices"][0]["message"]["content"].strip()
    set_cache(cache_key, result)
    return result

async def async_consolidate_element_summaries(entities, relationships):
    entity_map = {}
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        name = ent.get("name","")
        etype = ent.get("type","")
        desc = ent.get("description","")
        key_text = f"{name}::{etype}"
        key = hashlib.md5(key_text.encode("utf-8")).hexdigest()

        if key not in entity_map:
            entity_map[key] = {
                "name": name,
                "type": etype,
                "raw_snippets": [],
                "final_summary": None
            }
        entity_map[key]["raw_snippets"].append(desc)

    edges = []
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        s_txt = rel.get("source","")
        t_txt = rel.get("target","")
        r_desc = rel.get("description","")
        s_key = hashlib.md5((s_txt + "::SRC").encode("utf-8")).hexdigest()
        t_key = hashlib.md5((t_txt + "::TGT").encode("utf-8")).hexdigest()
        edges.append({
            "source_key": s_key,
            "target_key": t_key,
            "raw_relationship_text": r_desc,
            "final_summary": None
        })

    # Summarize entities in parallel
    summarize_tasks = []
    for k, val in entity_map.items():
        summarize_tasks.append(
            (k, async_summarize_entity(val["name"], val["type"], val["raw_snippets"]))
        )
    sum_results = await asyncio.gather(*[t[1] for t in summarize_tasks])
    for (k, _), result_txt in zip(summarize_tasks, sum_results):
        entity_map[k]["final_summary"] = result_txt

    # Summarize relationships in parallel
    rel_tasks = [async_summarize_relationship(e["raw_relationship_text"]) for e in edges]
    rel_sums = await asyncio.gather(*rel_tasks)
    for e, rs in zip(edges, rel_sums):
        e["final_summary"] = rs

    return entity_map, edges

# ========================================================================
# Graph + Community Detection Code
# ========================================================================
import igraph as ig
import leidenalg as la

def build_multilevel_communities(entity_map, edges, resolution_params=[1.0, 0.8, 0.6, 0.4]):
    """
    Build a graph from entity_map, edges, then run Leiden community detection
    across multiple resolution parameters.
    """
    g = ig.Graph()
    node_keys = list(entity_map.keys())
    idx_of = {k: i for i, k in enumerate(node_keys)}
    g.add_vertices(len(node_keys))

    edge_count_map = {}
    for e in edges:
        s = e["source_key"]
        t = e["target_key"]
        if s in idx_of and t in idx_of:
            pair = tuple(sorted([idx_of[s], idx_of[t]]))
            edge_count_map[pair] = edge_count_map.get(pair, 0) + 1

    all_edge_tuples = []
    all_weights = []
    for (u,v), w in edge_count_map.items():
        all_edge_tuples.append((u,v))
        all_weights.append(float(w))

    if all_edge_tuples:
        g.add_edges(all_edge_tuples)
        g.es["weight"] = all_weights

    partition_levels = {}
    for i, res in enumerate(resolution_params):
        partition = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights=g.es["weight"] if g.ecount() > 0 else None,
            resolution_parameter=res
        )
        partition_levels[i] = partition

    return g, node_keys, partition_levels

def get_community_dict(partition):
    """
    Convert a Leiden partition object into a dict: { comm_id: [list_of_node_indices], ... }
    """
    comm_dict = {}
    for node_idx, comm_id in enumerate(partition.membership):
        comm_dict.setdefault(comm_id, []).append(node_idx)
    return comm_dict

async def async_hierarchical_community_summary(entity_map, edges, node_keys, comm_members, level_idx, partition_levels):
    """
    Summarize a community by checking if we can fit all text in one GPT call.
    If it's too large, we go to the next partition level or chunk.
    """
    entity_texts = []
    edge_texts = []
    node_set = set(comm_members)

    for i in comm_members:
        k = node_keys[i]
        ent = entity_map[k]
        if ent["final_summary"]:
            entity_texts.append(ent["final_summary"])
        for rs in ent["raw_snippets"]:
            entity_texts.append(rs)

    for e in edges:
        try:
            s_idx = node_keys.index(e["source_key"])
            t_idx = node_keys.index(e["target_key"])
        except ValueError:
            continue
        if (s_idx in node_set) and (t_idx in node_set):
            if e["final_summary"]:
                edge_texts.append(e["final_summary"])
            if e["raw_relationship_text"]:
                edge_texts.append(e["raw_relationship_text"])

    combined_text = "\n".join(entity_texts + edge_texts)

    # If it fits, do a single summary:
    if len(combined_text.split()) < (MAX_CTX_TOKENS // 2):
        prompt = f"""
You are summarizing an entire community of entities and relationships from a dataset.

Raw text:
\"\"\"{combined_text}\"\"\"

Please create a cohesive summary preserving numeric/factual details.
"""
        resp = await aclient.chat.completions.create(model="gpt-4",
        messages=[{"role":"system","content":prompt}],
        temperature=0)
        short_summary = resp["choices"][0]["message"]["content"].strip()
        return short_summary
    else:
        # fallback to next level partition or chunk
        next_level = level_idx + 1
        partition_down = partition_levels.get(next_level, None)
        if partition_down is None:
            # chunk approach
            words = combined_text.split()
            step = MAX_CTX_TOKENS // 4
            chunked_text = []
            for i in range(0, len(words), step):
                sub_chunk = " ".join(words[i:i+step])
                chunked_text.append(sub_chunk)
            chunk_summaries = []
            for ch in chunked_text:
                prompt = f"""
You are summarizing text from a large community.
Chunk:
\"\"\"{ch}\"\"\"

Please produce a concise partial summary.
"""
                part_resp = await aclient.chat.completions.create(model="gpt-4",
                messages=[{"role":"system","content":prompt}],
                temperature=0)
                chunk_summaries.append(part_resp["choices"][0]["message"]["content"].strip())

            merged = "\n".join(chunk_summaries)
            prompt_merge = f"""
We have these partial summaries of a large community:
\"\"\"{merged}\"\"\"

Please merge them into one final, coherent summary, preserving factual details.
"""
            merge_resp = await aclient.chat.completions.create(model="gpt-4",
            messages=[{"role":"system","content":prompt_merge}],
            temperature=0)
            final_merged = merge_resp["choices"][0]["message"]["content"].strip()
            return final_merged
        else:
            # go deeper in the partition
            subcomm_dict = get_community_dict(partition_down)
            relevant_subcomms = []
            for sc_id, sc_members in subcomm_dict.items():
                overlap = set(sc_members).intersection(node_set)
                if overlap:
                    relevant_subcomms.append(overlap)

            sub_summaries = []
            for sub_nodes in relevant_subcomms:
                sub_list = sorted(list(sub_nodes))
                sum_sub = await async_hierarchical_community_summary(
                    entity_map, edges, node_keys, sub_list,
                    next_level, partition_levels
                )
                sub_summaries.append(sum_sub)

            partial_merged = "\n".join(sub_summaries)
            prompt_merge = f"""
We have these sub-community summaries for a parent community:
\"\"\"{partial_merged}\"\"\"

Please merge them into one final summary for the parent community, preserving factual details.
"""
            merge_resp = await aclient.chat.completions.create(model="gpt-4",
            messages=[{"role":"system","content":prompt_merge}],
            temperature=0)
            final_merged = merge_resp["choices"][0]["message"]["content"].strip()
            return final_merged

async def async_partial_answer(query, summary_text, comm_id):
    prompt = f"""
QUESTION:
{query}

COMMUNITY (ID={comm_id}) TEXT:
\"\"\"{summary_text}\"\"\"

1) Provide a partial answer to the question using details from this text.
2) Provide a helpfulness score from 0 to 100, rating how useful this partial answer is.

Return JSON with "partial_answer" and "helpfulness_score".
"""
    resp = await aclient.chat.completions.create(model="gpt-4",
    messages=[{"role":"system","content":prompt}],
    temperature=0)
    content = resp["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        pa = data.get("partial_answer","")
        score = float(data.get("helpfulness_score",0))
    except:
        pa = ""
        score = 0
    return pa, score

async def async_reduce_answers(query, partial_answers):
    valid_partial = [(pa, sc) for (pa, sc) in partial_answers if sc > 0]
    valid_partial.sort(key=lambda x: x[1], reverse=True)

    merged_text = ""
    for pa, sc in valid_partial:
        snippet = f"[Score={sc}] PartialAnswer:\n{pa}\n\n"
        if len(merged_text.split()) + len(snippet.split()) < (MAX_CTX_TOKENS // 2):
            merged_text += snippet
        else:
            break

    final_prompt = f"""
You have these partial answers for the question:
\"\"\"{query}\"\"\"

PARTIAL ANSWERS (sorted by helpfulness):
\"\"\"{merged_text}\"\"\"

Please merge them into one final, coherent response, preserving factual details.
Return a concise but accurate answer.
"""
    resp = await aclient.chat.completions.create(model="gpt-4",
    messages=[{"role":"system","content":final_prompt}],
    temperature=0)
    return resp["choices"][0]["message"]["content"].strip()

# ========================================================================
# >>>>> Revised "GraphRAG" - Simpler Hybrid Approach <<<<<
# ========================================================================
async def async_graph_rag_pipeline(docs, query, es_client, chosen_level=2, top_k=2):
    """
    1. Retrieve top-k docs via Elasticsearch (vector store).
    2. Build a graph from only those top-k docs.
    3. Summarize or do entity-level Q&A on that subgraph.
    4. Combine partial answers with the original top-k context as a fallback.
    """

    # --------------------------
    # STEP 1: Retrieve top_k docs
    # --------------------------
    index_name = "graph_rag_temp"
    try:
        es_client.indices.delete(index=index_name)
    except:
        pass

    # Index all docs for ephemeral retrieval
    vector_store = ElasticsearchStore.from_texts(
        texts=docs,
        embedding=embedding_function,
        es_connection=es_client,
        index_name=index_name
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)
    # Let's keep the plain text of top-k
    top_k_context = "\n".join([r.page_content for r in retrieved_docs])

    # Clean up ephemeral index
    try:
        es_client.indices.delete(index=index_name)
    except:
        pass

    # Now we have "retrieved_docs" from ES. We'll build a graph out of these top-k docs only.
    # This ensures the graph approach is focusing on relevant text.

    # --------------------------
    # STEP 2: Build Graph from top-k docs
    # --------------------------
    # Re-chunk the top-k docs
    sub_docs = [r.page_content for r in retrieved_docs]
    text_chunks = []
    for doc_text in sub_docs:
        text_chunks.extend(chunk_text(doc_text, CHUNK_SIZE, CHUNK_OVERLAP))

    all_entities, all_relationships = await async_extract_element_instances(text_chunks, gleanings=MAX_GLEANINGS)
    entity_map, edges = await async_consolidate_element_summaries(all_entities, all_relationships)

    # --------------------------
    # STEP 3: Summarize subgraph + partial Q&A
    # --------------------------
    g, node_keys, partition_levels = build_multilevel_communities(entity_map, edges)
    partition = partition_levels.get(chosen_level, None)
    if partition is None:
        # fallback to the last partition if chosen_level doesn't exist
        partition = list(partition_levels.values())[-1]

    comm_dict = get_community_dict(partition)

    partial_answers = []
    for comm_id, members in comm_dict.items():
        summary_text = await async_hierarchical_community_summary(
            entity_map, edges, node_keys, members, chosen_level, partition_levels
        )
        pa, sc = await async_partial_answer(query, summary_text, comm_id)
        partial_answers.append((pa, sc))

    # Merge partial answers
    partial_graph_answer = await async_reduce_answers(query, partial_answers)

    # --------------------------
    # STEP 4: Combine partial answers with top-k context
    # --------------------------
    final_prompt = f"""
We have a partial answer from a knowledge-graph approach:
\"\"\"{partial_graph_answer}\"\"\"

We also have the original top-k context from Elasticsearch retrieval:
\"\"\"{top_k_context}\"\"\"

Please combine them into a single, coherent answer to the question below,
preserving numeric/factual details.

Question: {query}
"""
    resp = await aclient.chat.completions.create(model="gpt-4",
    messages=[{"role":"system","content":final_prompt}],
    temperature=0)
    final_answer = resp["choices"][0]["message"]["content"].strip()

    return final_answer

# --------------------------------------------------------------------
# Minimal reduce step for final answer (if needed by other code)
# --------------------------------------------------------------------
async def async_llm_reduce_final_answer(query, partial_answers_text, level_idx):
    prompt = f"""
You have these partial answers, each with a helpfulness score, for the query:
\"\"\"{query}\"\"\"

PARTIAL ANSWERS:
\"\"\"{partial_answers_text}\"\"\"

Please merge them into one final, coherent response, preserving any factual details.
Return a concise but accurate answer.
"""
    resp = await aclient.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": prompt}],
    temperature=0)
    return resp["choices"][0]["message"]["content"].strip()

def _cosine_sim(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12))

# --------------------------------------------------------------------
# MAIN: Compare ES-based RAG vs. Hybrid GraphRAG
# --------------------------------------------------------------------
async def main():
    # We'll keep top_k=2 for both ES and Graph approaches
    k=2

    dataset_name = "delucionqa"  # example name
    ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")

    out_name = f"results-{dataset_name}-hybrid-graphrag.txt"
    f = open(out_name, "w")

    es_score_number = 0
    graph_score_number = 0

    cnt=0
    for x in ragbench_dataset:
        question = x["question"]
        docs = x["documents"]
        ground_truth = x["response"]

        print("Question:", question)
        f.write("========================================\n")
        f.write(f"Question: {question}\n")

        # ------------------------------------------------
        # (1) ES-based RAG
        # ------------------------------------------------
        index_name = "rag_index_temp"
        try:
            es_client.indices.delete(index=index_name)
        except:
            pass

        vector_store = ElasticsearchStore.from_texts(
            texts=docs,
            embedding=embedding_function,
            es_connection=es_client,
            index_name=index_name
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        retrieved = retriever.invoke(question)
        es_context = "\n".join([d.page_content for d in retrieved])

        rag_prompt = f"""
Based on the provided context, answer the question.
Keep numeric or factual details verbatim.

Context:
{es_context}

Question: {question}
"""
        rag_resp = await aclient.chat.completions.create(model="gpt-4",
        messages=[{"role":"system","content":rag_prompt}],
        temperature=0)
        es_answer = rag_resp["choices"][0]["message"]["content"].strip()

        try:
            es_client.indices.delete(index=index_name)
        except:
            pass

        f.write("--- ES-based RAG ---\n")
        f.write("Context:\n"+es_context+"\n")
        f.write("Answer:\n"+es_answer+"\n")

        # ------------------------------------------------
        # (2) Hybrid GraphRAG (Simpler)
        # ------------------------------------------------
        graph_answer = await async_graph_rag_pipeline(docs, question, es_client, chosen_level=2, top_k=k)
        f.write("--- Hybrid GraphRAG (Simpler) ---\n")
        f.write("Answer:\n"+graph_answer+"\n")

        # ------------------------------------------------
        # Evaluate correctness (simple 0/1 scoring)
        # ------------------------------------------------
        s_prompt_1 = f"Your task is to check if the candidate answer means the same as ground truth. Print 1 if yes, else 0. Ground Truth: {ground_truth} Candidate: {es_answer}"
        s_prompt_2 = f"Your task is to check if the candidate answer means the same as ground truth. Print 1 if yes, else 0. Ground Truth: {ground_truth} Candidate: {graph_answer}"

        stasks = [
            aclient.chat.completions.create(model="gpt-4", messages=[{"role":"system","content":s_prompt_1}], temperature=0),
            aclient.chat.completions.create(model="gpt-4", messages=[{"role":"system","content":s_prompt_2}], temperature=0)
        ]
        sres = await asyncio.gather(*stasks)
        es_scr = sres[0]["choices"][0]["message"]["content"].strip()
        graph_scr = sres[1]["choices"][0]["message"]["content"].strip()

        try: es_score_number += int(es_scr)
        except: pass
        try: graph_score_number += int(graph_scr)
        except: pass
        print("====================================================")
        print("ES Score so far:", es_score_number)
        print("Hybrid GraphRAG Score so far:", graph_score_number)

        f.write(f"ES-based RAG so far: {es_score_number}\n")
        f.write(f"Hybrid GraphRAG so far: {graph_score_number}\n")

        cnt += 1

    f.write("========================================\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Total examples: {len(ragbench_dataset)}\n")
    f.write(f"ES-based RAG final score: {es_score_number}\n")
    f.write(f"Hybrid GraphRAG final score: {graph_score_number}\n")
    f.close()

    print("Done. Results in:", out_name)

if __name__ == "__main__":
    asyncio.run(main())

# =======================
# END SINGLE CODE BLOCK
# =======================
