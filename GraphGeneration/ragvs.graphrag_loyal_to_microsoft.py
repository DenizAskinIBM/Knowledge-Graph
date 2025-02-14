"""
Below is a revised Python implementation of the GraphRAG algorithm described in the paper
“From Local to Global: A Graph RAG Approach to Query-Focused Summarization,” with
the original LLM CALL placeholders replaced by direct calls to GPT-4 (or your GPT4o endpoint).

IMPORTANT:
  - You must supply your OpenAI/GPT4o API key and handle usage/billing.
  - Adjust the `openai.ChatCompletion.acreate` calls as appropriate for your environment.
  - The prompts provided here are sample prompts; you may want to refine them further
    for best results in your domain.

INSTALL:
  pip install openai

If you also want Leiden hierarchical community detection, install:
  pip install igraph leidenalg

Usage:
  1) Provide a list of documents to the constructor (chunking them, etc.).
  2) After index-time steps, call `answer_query(your_query, level_idx)` to get an answer.
"""

import os
import random
import math
import json
import asyncio
from typing import List, Dict, Any, Tuple
import networkx as nx
from openai import AsyncOpenAI

aclient = AsyncOpenAI()

# For community detection with Leiden:
try:
    import igraph
    import leidenalg
    HAVE_LEIDEN = True
except ImportError:
    HAVE_LEIDEN = False
    print("Warning: 'igraph' or 'leidenalg' not installed. "
          "Community detection placeholder will be used.")

# ------------------------------------------------------------------
# Adjust the model name here to match your GPT-4 or GPT4o endpoint
# ------------------------------------------------------------------
MODEL_NAME = "gpt-4"

async def call_chat_completion_async(system_prompt: str, user_prompt: str) -> str:
    """
    Asynchronous convenience function to call GPT-4 / GPT4o.
    You need OPENAI_API_KEY set in your environment (or adapt as needed).
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

class GraphRAG:
    """
    A reference implementation of the Graph RAG pipeline, with GPT-4 calls
    replacing the placeholders.

    INDEX TIME:
      1) Chunk documents
      2) Extract entity/relationship instances (possibly with gleanings)
      3) Create consolidated element summaries
      4) Build an undirected weighted graph from these elements
      5) Detect hierarchical communities (C0, C1, C2, etc.) using Leiden
      6) Summarize communities in a hierarchical manner

    QUERY TIME:
      - For a user query, pick a particular level of community hierarchy
        (C0, C1, C2, or C3 in the paper).
      - For each community at that level, chunk its summary, use the LLM to produce
        a partial answer & a helpfulness score.
      - Filter out partial answers with score = 0, then run a final "reduce" step
        to produce the final global answer.
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

        # Data structures populated at index time:
        self.all_chunks = []              # List of text chunks
        self.element_instances = []       # Raw extracted nodes & edges from text
        self.element_summaries = {}       # Summaries keyed by element ID
        self.G = None                     # NetworkX graph
        self.hierarchy_levels = []        # List of membership arrays for each hierarchical level
        self.community_summaries = {}     # {level_idx: {community_id: summary_text}}

    # --------------------------------------------------
    # INDEX TIME STEPS
    # --------------------------------------------------

    def chunk_documents(self, documents: List[str]) -> None:
        """
        1) SOURCE DOCUMENTS → TEXT CHUNKS
        """
        for doc in documents:
            words = doc.split()
            i = 0
            while i < len(words):
                end = i + self.chunk_size
                chunk_words = words[i:end]
                chunk_text = " ".join(chunk_words)
                self.all_chunks.append(chunk_text)
                i += (self.chunk_size - self.overlap_size)

    async def _extract_entities_and_relationships_once(self, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """
        A single pass: identifies all entities and relationships.
        """
        system_prompt = (
            "You are an information extraction assistant. "
            "You will identify all entities and the relationships between them in a text. "
            "Return JSON with keys 'entities' (list) and 'relationships' (list). "
            "Entities: {name, type, description}. "
            "Relationships: {source, target, description}."
        )
        user_prompt = f"Text:\n{chunk}\nExtract the entities and relationships as JSON."
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
        """
        Checks if the LLM identifies that entities were missed in the prior extraction.
        Expect 'Yes' or 'No' answer.
        """
        system_prompt = (
            "You are a helpful assistant. "
            "Respond ONLY with 'Yes' or 'No'."
        )
        user_prompt = (
            f"I have extracted entities from the text below in a previous pass. "
            f"Did I miss ANY entities or relationships?\n\nText:\n{chunk}"
        )
        ans = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        ans_clean = ans.strip().lower()
        if ans_clean.startswith("yes"):
            return True
        return False

    async def _extract_additional_entities(self, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """
        We suspect we missed some. Encourage GPT to glean missing ones.
        """
        system_prompt = (
            "You are an information extraction assistant. "
            "We know some entities were missed. Please glean them now. "
            "Return JSON with keys 'entities' (list) and 'relationships' (list)."
        )
        user_prompt = f"MANY entities or relationships were missed in this text:\n{chunk}"
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
        """
        Process one text chunk by extracting entities/relationships (with possible gleanings).
        """
        entities, relationships = await self._extract_entities_and_relationships_once(chunk)
        combined_entities = list(entities)
        combined_relationships = list(relationships)

        for glean_iter in range(self.max_gleanings):
            missed = await self._check_for_missed_entities(chunk, glean_iter)
            if not missed:
                break
            new_ents, new_rels = await self._extract_additional_entities(chunk)
            combined_entities.extend(new_ents)
            combined_relationships.extend(new_rels)

        return {
            "chunk": chunk,
            "entities": combined_entities,
            "relationships": combined_relationships
        }

    async def extract_element_instances(self) -> None:
        """
        2) TEXT CHUNKS → ELEMENT INSTANCES
        """
        tasks = [self._process_chunk(chunk) for chunk in self.all_chunks]
        self.element_instances = await asyncio.gather(*tasks)

    async def _summarize_entity(self, etype: str, ename: str, descriptions: List[str], element_id: int) -> Tuple[str, Dict]:
        system_prompt = (
            "You are a summarizer. "
            "Below are multiple descriptive texts for the same entity. "
            "Produce a concise summary capturing its essence."
        )
        user_prompt = "\n".join(descriptions)
        summary = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        key = f"entity_{element_id}"
        return key, {"type": etype, "name": ename, "summary": summary}

    async def _summarize_relationship(self, src: str, tgt: str, rel_descs: List[str], element_id: int) -> Tuple[str, Dict]:
        system_prompt = (
            "You are a summarizer. "
            "Below are multiple descriptive texts for the same relationship. "
            "Produce a concise summary capturing the key relationship details."
        )
        user_prompt = "\n".join(rel_descs)
        summary = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        key = f"rel_{element_id}"
        return key, {"source": src, "target": tgt, "summary": summary}

    async def create_element_summaries(self) -> None:
        """
        3) ELEMENT INSTANCES → ELEMENT SUMMARIES
        """
        from collections import defaultdict
        entity_map = defaultdict(list)
        relationship_map = defaultdict(list)

        for instance_batch in self.element_instances:
            # Process entities safely:
            for ent in instance_batch["entities"]:
                # If ent is a dict, use it; otherwise try to parse it as JSON.
                if isinstance(ent, dict):
                    key = (ent.get("type", ""), ent.get("name", ""))
                    entity_map[key].append(ent.get("description", ""))
                else:
                    try:
                        ent_obj = json.loads(ent)
                        key = (ent_obj.get("type", ""), ent_obj.get("name", ""))
                        entity_map[key].append(ent_obj.get("description", ""))
                    except Exception:
                        continue
            # Process relationships safely:
            for rel in instance_batch["relationships"]:
                if isinstance(rel, dict):
                    key = (rel.get("source", ""), rel.get("target", ""))
                    relationship_map[key].append(rel.get("description", ""))
                else:
                    try:
                        rel_obj = json.loads(rel)
                        key = (rel_obj.get("source", ""), rel_obj.get("target", ""))
                        relationship_map[key].append(rel_obj.get("description", ""))
                    except Exception:
                        continue

        self.element_summaries = {}
        # Process entities concurrently.
        entity_items = list(entity_map.items())
        entity_tasks = []
        for i, ((etype, ename), descriptions) in enumerate(entity_items):
            entity_tasks.append(self._summarize_entity(etype, ename, descriptions, i))
        entity_results = await asyncio.gather(*entity_tasks)
        for key, summary_data in entity_results:
            self.element_summaries[key] = summary_data

        # Process relationships concurrently.
        relationship_items = list(relationship_map.items())
        relationship_tasks = []
        start_id = len(entity_items)
        for i, ((src, tgt), rel_descs) in enumerate(relationship_items, start=start_id):
            relationship_tasks.append(self._summarize_relationship(src, tgt, rel_descs, i))
        relationship_results = await asyncio.gather(*relationship_tasks)
        for key, summary_data in relationship_results:
            self.element_summaries[key] = summary_data

    def build_graph_and_detect_communities(self) -> None:
        """
        4) ELEMENT SUMMARIES → GRAPH COMMUNITIES (initial build)
        5) Use Leiden for hierarchical partition if available
        """
        G = nx.Graph()

        # Map entity_ to node
        entity_id_map = {}
        for el_id, el_data in self.element_summaries.items():
            if el_id.startswith("entity_"):
                node_label = f"{el_data['type']}::{el_data['name']}"
                G.add_node(node_label, summary=el_data["summary"])
                entity_id_map[el_id] = node_label

        # Add edges from 'rel_...' items
        for el_id, el_data in self.element_summaries.items():
            if el_id.startswith("rel_"):
                src = el_data["source"]
                tgt = el_data["target"]
                # Naive approach: find nodes by matching text.
                src_node = None
                tgt_node = None
                for node_lbl in G.nodes:
                    if src in node_lbl and tgt != src:
                        src_node = node_lbl
                    if tgt in node_lbl and tgt != src:
                        tgt_node = node_lbl
                if src_node and tgt_node:
                    if G.has_edge(src_node, tgt_node):
                        G[src_node][tgt_node]['weight'] += 1
                    else:
                        G.add_edge(src_node, tgt_node, weight=1)

        self.G = G

        if not HAVE_LEIDEN:
            print("Leiden not available; skipping hierarchical detection.")
            return

        # Convert to igraph for Leiden
        nx_nodes = list(G.nodes)
        ig = igraph.Graph()
        ig.add_vertices(len(nx_nodes))
        node_index_map = {n: i for i, n in enumerate(nx_nodes)}

        edges_list = []
        weights = []
        for u, v, d in G.edges(data=True):
            edges_list.append((node_index_map[u], node_index_map[v]))
            weights.append(d.get('weight', 1))
        ig.add_edges(edges_list)
        ig.es['weight'] = weights

        # Try multiple resolutions for a hierarchy
        resolutions = [1.0, 1.5, 2.0, 3.0]
        partitions = []
        for r in resolutions:
            part = leidenalg.find_partition(
                ig,
                leidenalg.RBConfigurationVertexPartition,
                weights=ig.es['weight'],
                resolution_parameter=r
            )
            partitions.append(part.membership)
        self.hierarchy_levels = partitions

    async def _summarize_community(self, level_idx: int, cid: int, user_prompt: str) -> Tuple[int, str]:
        system_prompt = (
            "You are a domain-tailored summarizer. "
            "Here are descriptions of nodes in one community. "
            "Please produce a coherent, concise summary that integrates their content."
        )
        summary = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        return cid, summary

    async def summarize_communities(self) -> None:
        """
        6) Summarize each community at each level in the hierarchy.
        """
        if not self.hierarchy_levels:
            return

        node_summary_map = {n: data.get('summary', '') for n, data in self.G.nodes(data=True)}
        nx_nodes = list(self.G.nodes)
        level_to_comms = {}

        for level_idx, membership in enumerate(self.hierarchy_levels):
            comm_map = {}
            for node_index, comm_id in enumerate(membership):
                node_label = nx_nodes[node_index]
                comm_map.setdefault(comm_id, []).append(node_label)
            level_to_comms[level_idx] = comm_map

        self.community_summaries = {}
        for level_idx, comm_map in level_to_comms.items():
            self.community_summaries[level_idx] = {}
            tasks = []
            for cid, node_labels in comm_map.items():
                user_prompt = "\n\n".join([node_summary_map.get(n, '') for n in node_labels])
                if not user_prompt.strip():
                    user_prompt = "No data"
                tasks.append(self._summarize_community(level_idx, cid, user_prompt))
            results = await asyncio.gather(*tasks)
            for cid, summary in results:
                self.community_summaries[level_idx][cid] = summary

    async def _generate_partial_answer(self, query: str, context: str) -> Tuple[str, float]:
        """
        Produces a partial answer plus a helpfulness score (0..100).
        """
        system_prompt = (
            "You are a query-focused summarizer. "
            "You are given some context, and a user query. "
            "Generate a partial answer that addresses the query as best as possible "
            "using the context. Also provide a helpfulness score from 0 to 100, integer. "
            "Answer must be valid JSON with keys 'partial_answer' and 'score'."
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Context: {context}\n"
            f"Return JSON like: {{'partial_answer': '...', 'score': 50}}"
        )
        raw_response = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        partial_answer = "[No partial answer]"
        score = 0.0
        try:
            j = json.loads(raw_response)
            partial_answer = j.get("partial_answer", "[No partial answer in JSON]")
            score = float(j.get("score", 0))
        except Exception:
            pass
        return partial_answer, score

    async def _final_reduce(self, query: str, partial_answers: List[Tuple[str, float]]) -> str:
        """
        Summarize partial answers into one final answer.
        """
        combined_answers = "\n".join([a for (a, s) in partial_answers])
        system_prompt = (
            "You are a query-focused summarizer. "
            "Combine the partial answers below into a single, coherent final answer to the query."
        )
        user_prompt = f"QUERY: {query}\n\nPARTIAL ANSWERS:\n{combined_answers}"
        final_answer = await call_chat_completion_async(system_prompt=system_prompt, user_prompt=user_prompt)
        return final_answer

    async def answer_query(self, query: str, community_level: int) -> str:
        """
        6) COMMUNITY SUMMARIES → COMMUNITY ANSWERS
        7) COMMUNITY ANSWERS → GLOBAL ANSWER
        """
        if community_level not in self.community_summaries:
            return f"No community summaries for level {community_level}."

        comm_summaries = list(self.community_summaries[community_level].items())
        random.shuffle(comm_summaries)
        partial_answers = []
        chunk_size = 2000  # naive chunk size in words

        tasks = []
        # For each community summary, split it into segments and schedule partial answer generation.
        for cid, summary_text in comm_summaries:
            words = summary_text.split()
            for i in range(0, len(words), chunk_size):
                segment = " ".join(words[i:i+chunk_size])
                tasks.append(self._generate_partial_answer(query, segment))
        results = await asyncio.gather(*tasks)
        # Filter out partial answers with a score of 0.
        for partial_answer, score in results:
            if score > 0:
                partial_answers.append((partial_answer, score))
        partial_answers.sort(key=lambda x: x[1], reverse=True)
        final_answer = await self._final_reduce(query, partial_answers)
        return final_answer

# ------------------------------------------------------------------
# DEMO USAGE
# ------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        # Provide your OpenAI API key or GPT4o equivalent:
        # os.environ["OPENAI_API_KEY"] = "sk-..."
        dataset_name = "delucionqa"
        from datasets import load_dataset
        ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")

        for x in ragbench_dataset:
            question = x["question"]
            docs = x["documents"]
            ground_truth = x["response"]
            graphrag = GraphRAG(chunk_size=600, overlap_size=100, max_gleanings=1, llm_context_size=8000)
            graphrag.chunk_documents(docs)
            await graphrag.extract_element_instances()
            await graphrag.create_element_summaries()
            graphrag.build_graph_and_detect_communities()
            await graphrag.summarize_communities()
            final_answer = await graphrag.answer_query(question, community_level=0)
            print("\n=== FINAL ANSWER ===\n", final_answer)

    asyncio.run(main())