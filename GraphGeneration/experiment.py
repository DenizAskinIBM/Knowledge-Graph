import os
import random
import math
import json
import asyncio
from typing import List, Dict, Any, Tuple
import networkx as nx
from openai import AsyncOpenAI

aclient = AsyncOpenAI()
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
# ---------------------------
# Elasticsearch Setup
# ---------------------------
ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")

es_client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,  # In production, set this to True and provide CA certificates
    request_timeout=3600,
)

# Embedding function for vector similarity search
embedding_function = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# ---------------------------
# Community Detection Dependencies
# ---------------------------
try:
    import igraph
    import leidenalg
    HAVE_LEIDEN = True
except ImportError:
    HAVE_LEIDEN = False
    print("Warning: 'igraph' or 'leidenalg' not installed. Community detection will be skipped.")

# ---------------------------
# OpenAI Setup
# ---------------------------
MODEL_NAME = "gpt-4"

async def call_chat_completion_async(system_prompt: str, user_prompt: str) -> str:
    """
    Convenience asynchronous function to call GPT-4/GPT4o.
    Ensure your OPENAI_API_KEY is set.
    """
    try:
        response = await aclient.chat.completions.create(model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0)
        text = response["choices"][0]["message"]["content"]
        return text.strip()
    except Exception as e:
        print("Error calling GPT:", e)
        return ""

# ---------------------------
# GraphRAG Class Definition
# ---------------------------
class GraphRAG:
    """
    GraphRAG processes documents by first chunking them, extracting entities and relationships,
    summarizing them to build a graph, detecting communities via Leiden (if available),
    and then summarizing each community. At query time, the community summaries are
    concatenated to serve as context.
    """
    def __init__(self,
                 chunk_size: int = 600,
                 overlap_size: int = 100,
                 max_gleanings: int = 1,
                 llm_context_size: int = 8000):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_gleanings = max_gleanings
        self.llm_context_size = llm_context_size

        self.all_chunks = []         # List of text chunks
        self.element_instances = []  # Raw extracted instances (entities and relationships)
        self.element_summaries = {}  # Summaries keyed by element ID
        self.G = None                # The NetworkX graph
        self.hierarchy_levels = []   # Community memberships (one per resolution)
        self.community_summaries = {}  # {level: {community_id: summary}}

    def chunk_documents(self, documents: List[str]) -> None:
        """
        Split documents into overlapping chunks.
        """
        for doc in documents:
            words = doc.split()
            i = 0
            while i < len(words):
                end = i + self.chunk_size
                chunk_text = " ".join(words[i:end])
                self.all_chunks.append(chunk_text)
                i += (self.chunk_size - self.overlap_size)

    async def _extract_entities_and_relationships_once(self, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        system_prompt = (
            "You are an information extraction assistant. "
            "Extract entities and relationships from the following text and return JSON with keys 'entities' and 'relationships'."
        )
        user_prompt = f"Text:\n{chunk}\nExtract entities and relationships as JSON."
        response = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        entities, relationships = [], []
        try:
            j = json.loads(response)
            entities = j.get("entities", [])
            relationships = j.get("relationships", [])
        except Exception:
            pass
        return entities, relationships

    async def _check_for_missed_entities(self, chunk: str, iteration: int) -> bool:
        system_prompt = "You are a helpful assistant. Respond only with 'Yes' or 'No'."
        user_prompt = f"Did you miss any entities or relationships in the following text?\n\n{chunk}"
        ans = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        return ans.strip().lower().startswith("yes")

    async def _extract_additional_entities(self, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        system_prompt = (
            "You are an information extraction assistant. Some entities may have been missed. "
            "Please extract any additional entities and relationships from the following text and return JSON."
        )
        user_prompt = f"Text:\n{chunk}\nExtract additional entities and relationships as JSON."
        response = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        entities, relationships = [], []
        try:
            j = json.loads(response)
            entities = j.get("entities", [])
            relationships = j.get("relationships", [])
        except Exception:
            pass
        return entities, relationships

    async def _process_chunk(self, chunk: str) -> Dict:
        entities, relationships = await self._extract_entities_and_relationships_once(chunk)
        combined_entities = list(entities)
        combined_relationships = list(relationships)
        for i in range(self.max_gleanings):
            if not await self._check_for_missed_entities(chunk, i):
                break
            new_entities, new_relationships = await self._extract_additional_entities(chunk)
            combined_entities.extend(new_entities)
            combined_relationships.extend(new_relationships)
        return {"chunk": chunk, "entities": combined_entities, "relationships": combined_relationships}

    async def extract_element_instances(self) -> None:
        tasks = [self._process_chunk(chunk) for chunk in self.all_chunks]
        self.element_instances = await asyncio.gather(*tasks)

    async def _summarize_entity(self, etype: str, ename: str, descriptions: List[str], eid: int) -> Tuple[str, Dict]:
        system_prompt = (
            "You are a summarizer. Below are descriptions for an entity. "
            "Produce a concise summary capturing its essence."
        )
        user_prompt = "\n".join(descriptions)
        summary = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        key = f"entity_{eid}"
        return key, {"type": etype, "name": ename, "summary": summary}

    async def _summarize_relationship(self, src: str, tgt: str, rel_descs: List[str], eid: int) -> Tuple[str, Dict]:
        system_prompt = (
            "You are a summarizer. Below are descriptions for a relationship. "
            "Produce a concise summary capturing its key details."
        )
        user_prompt = "\n".join(rel_descs)
        summary = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        key = f"rel_{eid}"
        return key, {"source": src, "target": tgt, "summary": summary}

    async def create_element_summaries(self) -> None:
        from collections import defaultdict
        entity_map = defaultdict(list)
        relationship_map = defaultdict(list)
        for instance in self.element_instances:
            for ent in instance["entities"]:
                # Handle if ent is a dict or string
                if isinstance(ent, dict):
                    key = (ent.get("type", ""), ent.get("name", ""))
                    description = ent.get("description", "")
                else:
                    try:
                        ent_obj = json.loads(ent)
                        key = (ent_obj.get("type", ""), ent_obj.get("name", ""))
                        description = ent_obj.get("description", "")
                    except Exception:
                        continue
                entity_map[key].append(description)
            for rel in instance["relationships"]:
                if isinstance(rel, dict):
                    key = (rel.get("source", ""), rel.get("target", ""))
                    description = rel.get("description", "")
                else:
                    try:
                        rel_obj = json.loads(rel)
                        key = (rel_obj.get("source", ""), rel_obj.get("target", ""))
                        description = rel_obj.get("description", "")
                    except Exception:
                        continue
                relationship_map[key].append(description)
        self.element_summaries = {}
        entity_items = list(entity_map.items())
        entity_tasks = [self._summarize_entity(etype, ename, descs, i)
                        for i, ((etype, ename), descs) in enumerate(entity_items)]
        for key, summary in await asyncio.gather(*entity_tasks):
            self.element_summaries[key] = summary
        relationship_items = list(relationship_map.items())
        start_id = len(entity_items)
        relationship_tasks = [self._summarize_relationship(src, tgt, rels, i)
                              for i, ((src, tgt), rels) in enumerate(relationship_items, start=start_id)]
        for key, summary in await asyncio.gather(*relationship_tasks):
            self.element_summaries[key] = summary

    def build_graph_and_detect_communities(self) -> None:
        # Build a graph where nodes are entities (with their summaries) and edges are relationships.
        G = nx.Graph()
        for el_id, data in self.element_summaries.items():
            if el_id.startswith("entity_"):
                node_label = f"{data['type']}::{data['name']}"
                G.add_node(node_label, summary=data["summary"])
        for el_id, data in self.element_summaries.items():
            if el_id.startswith("rel_"):
                src = data["source"]
                tgt = data["target"]
                src_node, tgt_node = None, None
                for node in G.nodes:
                    if src in node:
                        src_node = node
                    if tgt in node:
                        tgt_node = node
                if src_node and tgt_node:
                    if G.has_edge(src_node, tgt_node):
                        G[src_node][tgt_node]["weight"] += 1
                    else:
                        G.add_edge(src_node, tgt_node, weight=1)
        self.G = G
        if not HAVE_LEIDEN:
            print("Leiden not available; skipping community detection.")
            return
        nx_nodes = list(G.nodes)
        ig = igraph.Graph()
        ig.add_vertices(len(nx_nodes))
        node_index_map = {n: i for i, n in enumerate(nx_nodes)}
        edges = []
        weights = []
        for u, v, d in G.edges(data=True):
            edges.append((node_index_map[u], node_index_map[v]))
            weights.append(d.get("weight", 1))
        ig.add_edges(edges)
        ig.es["weight"] = weights
        resolutions = [1.0, 1.5, 2.0, 3.0]
        self.hierarchy_levels = []
        for r in resolutions:
            part = leidenalg.find_partition(
                ig,
                leidenalg.RBConfigurationVertexPartition,
                weights=ig.es["weight"],
                resolution_parameter=r
            )
            self.hierarchy_levels.append(part.membership)

    async def _summarize_community(self, level_idx: int, cid: int, context: str) -> Tuple[int, str]:
        system_prompt = (
            "You are a domain-tailored summarizer. "
            "Given the following texts from a community, produce a concise summary."
        )
        summary = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=context)
        return cid, summary

    async def summarize_communities(self) -> None:
        """
        For simplicity, we use the first (lowest resolution) community partition.
        Each community’s node summaries are concatenated and summarized.
        """
        if not self.hierarchy_levels:
            return
        node_summaries = {n: data.get("summary", "") for n, data in self.G.nodes(data=True)}
        nx_nodes = list(self.G.nodes)
        # Use the first resolution level (index 0)
        membership = self.hierarchy_levels[0]
        level_comms = {}
        for idx, comm_id in enumerate(membership):
            text = node_summaries.get(nx_nodes[idx], "")
            level_comms.setdefault(comm_id, []).append(text)
        self.community_summaries = {}
        self.community_summaries[0] = {}
        tasks = []
        for cid, texts in level_comms.items():
            context = " ".join(texts)
            tasks.append(self._summarize_community(0, cid, context))
        for cid, summary in await asyncio.gather(*tasks):
            self.community_summaries[0][cid] = summary

    async def answer_query(self, query: str) -> str:
        """
        (Optional) GraphRAG can itself generate an answer using its community summaries.
        """
        if not self.community_summaries:
            return "No community summaries available."
        comm_contexts = list(self.community_summaries[0].values())
        context = " ".join(comm_contexts)
        system_prompt = "You are a query-focused summarizer. Use the provided context to answer the query."
        user_prompt = f"Question: {query}\nContext: {context}"
        return await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)

# ---------------------------
# Elasticsearch RAG Function
# ---------------------------
async def es_rag_answer(query: str, docs: List[str], k: int = 5) -> str:
    """
    Given a query and documents, index the texts into Elasticsearch,
    retrieve the top-k similar documents, and then use GPT-4 to generate an answer.
    """
    index_name = "rag_index_temp"
    try:
        es_client.indices.delete(index=index_name)
    except Exception:
        pass
    vector_store = ElasticsearchStore.from_texts(
        texts=docs,
        embedding=embedding_function,
        es_connection=es_client,
        index_name=index_name
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved = retriever.invoke(query)
    es_context = "\n".join([d.page_content for d in retrieved])
    system_prompt = "You are a query-focused summarizer. Generate an answer based solely on the provided context."
    user_prompt = f"Question: {query}\nContext: {es_context}"
    return await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)

# ---------------------------
# Main Function: Running Both ES RAG Pipelines and Comparing Results
# ---------------------------
async def main():
    # For cumulative scoring (simple 0/1 evaluation) if desired
    es_standard_score = 0
    es_graph_score = 0
    k = 5  # Number of documents to retrieve

    from datasets import load_dataset
    dataset_name = "delucionqa"
    ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")

    for x in ragbench_dataset:
        question = x["question"]
        docs = x["documents"]
        ground_truth = x["response"]

        # ---------------------------
        # Run GraphRAG Pipeline
        # ---------------------------
        graphrag = GraphRAG(chunk_size=600, overlap_size=100, max_gleanings=1, llm_context_size=8000)
        graphrag.chunk_documents(docs)
        await graphrag.extract_element_instances()
        await graphrag.create_element_summaries()
        graphrag.build_graph_and_detect_communities()
        await graphrag.summarize_communities()

        # Use the community summaries (from level 0) and concatenate them to form a query.
        if graphrag.community_summaries and 0 in graphrag.community_summaries:
            community_query = " ".join(graphrag.community_summaries[0].values())
        else:
            community_query = ""

        # ---------------------------
        # ES RAG Standard: Use Original Question as Query
        # ---------------------------
        es_answer_standard = await es_rag_answer(question, docs, k=k)

        # ---------------------------
        # ES RAG Graph-based: Use Community Summaries as Query
        # ---------------------------
        if community_query:
            es_answer_graph = await es_rag_answer(community_query, docs, k=k)
        else:
            es_answer_graph = "No community query available."

        # ---------------------------
        # Print the Answers
        # ---------------------------
        print("===========================================")
        print("Question:", question)
        print("Ground Truth:", ground_truth)
        print("\nElasticsearch RAG (Standard) Answer:\n", es_answer_standard)
        print("\nElasticsearch RAG (Graph-based query) Answer:\n", es_answer_graph)
        print("===========================================\n")

        # ---------------------------
        # Evaluation (0/1 scoring via GPT-4 evaluator)
        # ---------------------------
        evaluator_sys = "You are an evaluator. Answer with 1 if the candidate answer matches the ground truth in meaning, otherwise 0."
        eval_prompt_standard = f"Ground Truth: {ground_truth}\nCandidate (Standard): {es_answer_standard}"
        eval_prompt_graph = f"Ground Truth: {ground_truth}\nCandidate (Graph-based): {es_answer_graph}"
        score_str_standard = await call_chat_completion_async(system_prompt=evaluator_sys, user_prompt=eval_prompt_standard)
        score_str_graph = await call_chat_completion_async(system_prompt=evaluator_sys, user_prompt=eval_prompt_graph)
        try:
            es_standard_score += int(score_str_standard.strip())
        except Exception:
            pass
        try:
            es_graph_score += int(score_str_graph.strip())
        except Exception:
            pass

        print("Scores for this example:")
        print("ES RAG Standard:", score_str_standard.strip())
        print("ES RAG Graph-based:", score_str_graph.strip())
        print("Cumulative Scores - Standard:", es_standard_score, "Graph-based:", es_graph_score)
        print("===========================================\n")

if __name__ == "__main__":
    asyncio.run(main())