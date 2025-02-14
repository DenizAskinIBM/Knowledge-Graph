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

# For Elasticsearch-based RAG:
from elasticsearch import Elasticsearch

# ------------------------------------------------------------------
# Adjust the model name here to match your GPT-4 or GPT4o endpoint
# ------------------------------------------------------------------
MODEL_NAME = "gpt-4"

async def call_chat_completion_async(system_msg: str, user_msg: str) -> str:
    """
    Asynchronous convenience function to call GPT-4 / GPT4o.
    You need OPENAI_API_KEY set in your environment (or adapt as needed).
    """
    try:
        response = await aclient.chat.completions.create(model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
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
        system_msg = (
            "You are an information extraction assistant. "
            "You will identify all entities and the relationships between them in a text. "
            "Return JSON with keys 'entities' (list) and 'relationships' (list). "
            "Entities: {name, type, description}. "
            "Relationships: {source, target, description}."
        )
        user_msg = f"Text:\n{chunk}\nExtract the entities and relationships as JSON."
        response = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)

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
        system_msg = (
            "You are a helpful assistant. "
            "Respond ONLY with 'Yes' or 'No'."
        )
        user_msg = (
            f"I have extracted entities from the text below in a previous pass. "
            f"Did I miss ANY entities or relationships?\n\nText:\n{chunk}"
        )
        ans = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
        ans_clean = ans.strip().lower()
        if ans_clean.startswith("yes"):
            return True
        return False

    async def _extract_additional_entities(self, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        """
        We suspect we missed some. Encourage GPT to glean missing ones.
        """
        system_msg = (
            "You are an information extraction assistant. "
            "We know some entities were missed. Please glean them now. "
            "Return JSON with keys 'entities' (list) and 'relationships' (list)."
        )
        user_msg = f"MANY entities or relationships were missed in this text:\n{chunk}"
        response = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
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
        system_msg = (
            "You are a summarizer. "
            "Below are multiple descriptive texts for the same entity. "
            "Produce a concise summary capturing its essence."
        )
        user_msg = "\n".join(descriptions)
        summary = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
        key = f"entity_{element_id}"
        return key, {"type": etype, "name": ename, "summary": summary}

    async def _summarize_relationship(self, src: str, tgt: str, rel_descs: List[str], element_id: int) -> Tuple[str, Dict]:
        system_msg = (
            "You are a summarizer. "
            "Below are multiple descriptive texts for the same relationship. "
            "Produce a concise summary capturing the key relationship details."
        )
        user_msg = "\n".join(rel_descs)
        summary = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
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
            for ent in instance_batch["entities"]:
                key = (ent.get("type", ""), ent.get("name", ""))
                entity_map[key].append(ent.get("description", ""))
            for rel in instance_batch["relationships"]:
                key = (rel.get("source", ""), rel.get("target", ""))
                relationship_map[key].append(rel.get("description", ""))

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

    async def _summarize_community(self, level_idx: int, cid: int, user_msg: str) -> Tuple[int, str]:
        system_msg = (
            "You are a domain-tailored summarizer. "
            "Here are descriptions of nodes in one community. "
            "Please produce a coherent, concise summary that integrates their content."
        )
        summary = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
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
                user_msg = "\n\n".join([node_summary_map.get(n, '') for n in node_labels])
                if not user_msg.strip():
                    user_msg = "No data"
                tasks.append(self._summarize_community(level_idx, cid, user_msg))
            results = await asyncio.gather(*tasks)
            for cid, summary in results:
                self.community_summaries[level_idx][cid] = summary

    async def _generate_partial_answer(self, query: str, context: str) -> Tuple[str, float]:
        """
        Produces a partial answer plus a helpfulness score (0..100).
        """
        system_msg = (
            "You are a query-focused summarizer. "
            "You are given some context, and a user query. "
            "Generate a partial answer that addresses the query as best as possible "
            "using the context. Also provide a helpfulness score from 0 to 100, integer. "
            "Answer must be valid JSON with keys 'partial_answer' and 'score'."
        )
        user_msg = (
            f"Query: {query}\n"
            f"Context: {context}\n"
            f"Return JSON like: {{'partial_answer': '...', 'score': 50}}"
        )
        raw_response = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
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
        system_msg = (
            "You are a query-focused summarizer. "
            "Combine the partial answers below into a single, coherent final answer to the query."
        )
        user_msg = f"QUERY: {query}\n\nPARTIAL ANSWERS:\n{combined_answers}"
        final_answer = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
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

class ElasticsearchRAG:
    """
    A simple Elasticsearch-based Retrieval-Augmented Generation (RAG) implementation.
    """
    def __init__(self, index_name: str = "rag_documents", top_k: int = 5):
        self.index_name = index_name
        self.top_k = top_k
        self.es = Elasticsearch()

    def index_documents(self, documents: List[str]) -> None:
        """
        Index a list of documents into Elasticsearch.
        """
        # Create the index if it doesn't exist.
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, ignore=400)
        for i, doc in enumerate(documents):
            self.es.index(index=self.index_name, id=i, body={"content": doc})
        # Refresh the index to make documents searchable.
        self.es.indices.refresh(index=self.index_name)

    async def answer_query(self, query: str) -> str:
        """
        Search for relevant documents and generate an answer based on them.
        """
        search_body = {
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": self.top_k
        }
        res = self.es.search(index=self.index_name, body=search_body)
        hits = res['hits']['hits']
        retrieved_context = " ".join(hit['_source']['content'] for hit in hits)
        system_msg = (
            "You are a query-focused summarizer. "
            "Generate an answer based solely on the following retrieved documents."
        )
        user_msg = f"Query: {query}\n\nRetrieved Documents:\n{retrieved_context}"
        answer = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
        return answer

async def grade_answers(query: str, graphrag_answer: str, es_answer: str) -> Dict[str, float]:
    """
    Compare the two answers by asking GPT to grade them on a scale of 0 to 100.
    Returns a dict with keys 'graphrag_grade' and 'elasticsearch_grade'.
    """
    system_msg = (
        "You are an expert evaluator. "
        "Given a user query and two answers from different systems, assign a grade from 0 to 100 to each answer based on correctness, relevance, and clarity. "
        "Return a JSON object with keys 'graphrag_grade' and 'elasticsearch_grade'."
    )
    user_msg = (
        f"User Query: {query}\n\n"
        f"GraphRAG Answer:\n{graphrag_answer}\n\n"
        f"Elasticsearch RAG Answer:\n{es_answer}\n\n"
        "Please provide your evaluation as JSON."
    )
    response = await call_chat_completion_async(system_msg=system_msg, user_msg=user_msg)
    grades = {"graphrag_grade": 0.0, "elasticsearch_grade": 0.0}
    try:
        j = json.loads(response)
        grades["graphrag_grade"] = float(j.get("graphrag_grade", 0))
        grades["elasticsearch_grade"] = float(j.get("elasticsearch_grade", 0))
    except Exception:
        pass
    return grades

# ------------------------------------------------------------------
# DEMO USAGE
# ------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        # Provide your OpenAI API key or GPT4o equivalent if needed:
        # os.environ["OPENAI_API_KEY"] = "sk-..."
        dataset_name = "delucionqa"
        from datasets import load_dataset
        ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")

        # Initialize Elasticsearch RAG and index documents (across all dataset examples)
        es_rag = ElasticsearchRAG(index_name="rag_documents", top_k=5)
        all_docs = []
        for x in ragbench_dataset:
            docs = x["documents"]
            all_docs.extend(docs)
        # Remove duplicates and index
        all_docs = list(set(all_docs))
        es_rag.index_documents(all_docs)

        # Process each example in the dataset.
        for x in ragbench_dataset:
            question = x["question"]
            docs = x["documents"]
            ground_truth = x["response"]

            # --- GraphRAG Processing ---
            graphrag = GraphRAG(chunk_size=600, overlap_size=100, max_gleanings=1, llm_context_size=8000)
            graphrag.chunk_documents(docs)
            await graphrag.extract_element_instances()
            await graphrag.create_element_summaries()
            graphrag.build_graph_and_detect_communities()
            await graphrag.summarize_communities()
            graphrag_answer = await graphrag.answer_query(question, community_level=0)

            # --- Elasticsearch RAG Processing ---
            es_answer = await es_rag.answer_query(question)

            # --- Grade Comparison ---
            grades = await grade_answers(question, graphrag_answer, es_answer)

            print("\n=== QUESTION ===\n", question)
            print("\n--- GraphRAG Answer ---\n", graphrag_answer)
            print("\n--- Elasticsearch RAG Answer ---\n", es_answer)
            print("\n--- Grades ---\n", grades)

    asyncio.run(main())