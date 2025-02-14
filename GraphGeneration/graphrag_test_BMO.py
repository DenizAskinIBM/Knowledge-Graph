import os
import csv
import random
import math
import json
import asyncio
from typing import List, Dict, Any, Tuple
import networkx as nx
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# For reading PDF
import PyPDF2

# Attempt to import igraph and leidenalg for community detection
try:
    import igraph
    import leidenalg
    HAVE_LEIDEN = True
except ImportError:
    HAVE_LEIDEN = False
    print("Warning: 'igraph' or 'leidenalg' not installed. "
          "Community detection placeholder will be used.")

# Environment variables for Elasticsearch connection
ES_URL = os.getenv("ES_URL")
ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")

es_client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False,
    request_timeout=3600,
)

# Define the embedding function
embedding_function = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# ------------------------------------------------------------------
# Adjust the model name here to match your GPT-4 or GPT-3.5 or GPT4o endpoint
# ------------------------------------------------------------------
MODEL_NAME = "gpt-4-turbo"

async def call_chat_completion_async(system_prompt: str, user_prompt: str) -> str:
    """
    Convenience asynchronous function to call GPT-4/GPT-3.5, etc.
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

class GraphRAG:
    """
    A reference implementation of the Graph RAG pipeline, with GPT calls
    replacing the placeholders.
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
        self.hierarchy_levels = []        # Membership arrays for each hierarchical level
        self.community_summaries = {}     # {level_idx: {community_id: summary_text}}

    # --------------------------------------------------
    # INDEX TIME STEPS
    # --------------------------------------------------
    def chunk_documents(self, documents: List[str]) -> None:
        """
        Split each document into overlapping text chunks.
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
        system_msg = (
            "You are an information extraction assistant. "
            "Identify all entities and relationships in the text and return JSON with keys 'entities' and 'relationships'."
        )
        user_msg = f"Text:\n{chunk}\nExtract the entities and relationships as JSON."
        response = await call_chat_completion_async(system_prompt=system_msg, user_prompt=user_msg)
        entities, relationships = [], []
        try:
            j = json.loads(response)
            entities = j.get("entities", [])
            relationships = j.get("relationships", [])
        except Exception:
            pass
        return entities, relationships

    async def _check_for_missed_entities(self, chunk: str, iteration: int) -> bool:
        system_msg = "You are a helpful assistant. Respond ONLY with 'Yes' or 'No'."
        user_msg = f"Did you miss any entities or relationships in this text?\n\n{chunk}"
        ans = await call_chat_completion_async(system_prompt=system_msg, user_prompt=user_msg)
        return ans.strip().lower().startswith("yes")

    async def _extract_additional_entities(self, chunk: str) -> Tuple[List[Dict], List[Dict]]:
        system_msg = (
            "You are an information extraction assistant. Some entities may have been missed. "
            "Please extract any additional entities and relationships, and return JSON."
        )
        user_msg = f"Additional extraction for text:\n{chunk}"
        response = await call_chat_completion_async(system_prompt=system_msg, user_prompt=user_msg)
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
        system_msg = (
            "You are a summarizer. "
            "Below are multiple descriptions for an entity. Produce a concise summary of its essence."
        )
        user_msg = "\n".join(descriptions)
        summary = await call_chat_completion_async(system_prompt=system_msg, user_prompt=user_msg)
        key = f"entity_{eid}"
        return key, {"type": etype, "name": ename, "summary": summary}

    async def _summarize_relationship(self, src: str, tgt: str, rel_descs: List[str], eid: int) -> Tuple[str, Dict]:
        system_msg = (
            "You are a summarizer. "
            "Below are descriptions for a relationship. Produce a concise summary capturing its key details."
        )
        user_msg = "\n".join(rel_descs)
        summary = await call_chat_completion_async(system_prompt=system_msg, user_prompt=user_msg)
        key = f"rel_{eid}"
        return key, {"source": src, "target": tgt, "summary": summary}

    async def create_element_summaries(self) -> None:
        from collections import defaultdict
        entity_map = defaultdict(list)
        relationship_map = defaultdict(list)

        for instance_batch in self.element_instances:
            # Process entities
            for ent in instance_batch["entities"]:
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

            # Process relationships
            for rel in instance_batch["relationships"]:
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
        G = nx.Graph()
        # Add entity nodes
        for el_id, data in self.element_summaries.items():
            if el_id.startswith("entity_"):
                node_label = f"{data['type']}::{data['name']}"
                G.add_node(node_label, summary=data["summary"])
        # Add relationship edges
        for el_id, data in self.element_summaries.items():
            if el_id.startswith("rel_"):
                src = data["source"]
                tgt = data["target"]
                src_node, tgt_node = None, None
                # Try to find a node that contains the src or tgt name
                for node in G.nodes:
                    if src in node:
                        src_node = node
                    if tgt in node:
                        tgt_node = node
                if src_node and tgt_node:
                    if G.has_edge(src_node, tgt_node):
                        G[src_node][tgt_node]['weight'] += 1
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
        ig.es['weight'] = weights
        resolutions = [1.0, 1.5, 2.0, 3.0]
        self.hierarchy_levels = [
            leidenalg.find_partition(
                ig, leidenalg.RBConfigurationVertexPartition,
                weights=ig.es['weight'],
                resolution_parameter=r
            ).membership
            for r in resolutions
        ]

    async def _summarize_community(self, level_idx: int, cid: int, context: str) -> Tuple[int, str]:
        system_msg = (
            "You are a domain-tailored summarizer. "
            "Given the following node descriptions from a community, produce a coherent and concise summary."
        )
        summary = await call_chat_completion_async(system_prompt=system_msg, user_prompt=context)
        return cid, summary

    async def summarize_communities(self) -> None:
        if not self.hierarchy_levels:
            return
        node_summary = {n: data.get("summary", "") for n, data in self.G.nodes(data=True)}
        nx_nodes = list(self.G.nodes)
        level_comms = {}
        for level_idx, membership in enumerate(self.hierarchy_levels):
            comm_map = {}
            for node_idx, comm_id in enumerate(membership):
                node_label = nx_nodes[node_idx]
                comm_map.setdefault(comm_id, []).append(node_summary.get(node_label, ""))
            level_comms[level_idx] = comm_map
        self.community_summaries = {}
        for level_idx, comm_map in level_comms.items():
            self.community_summaries[level_idx] = {}
            tasks = []
            for cid, texts in comm_map.items():
                context = "\n\n".join(texts) if texts else "No data"
                tasks.append(self._summarize_community(level_idx, cid, context))
            for cid, summary in await asyncio.gather(*tasks):
                self.community_summaries[level_idx][cid] = summary

    # --------------------------------------------------
    # QUERY TIME
    # --------------------------------------------------
    async def _generate_partial_answer(self, query: str, context: str) -> Tuple[str, float]:
        system_msg = (
            "You are a query-focused summarizer. "
            "Given the query and some context, generate a partial answer along with a helpfulness score (0 to 100) in JSON."
        )
        user_msg = f"Query: {query}\nContext: {context}\nReturn JSON like: {{'partial_answer': '...', 'score': 50}}"
        raw = await call_chat_completion_async(system_prompt=system_msg, user_prompt=user_msg)
        partial_answer, score = "[No answer]", 0.0
        try:
            j = json.loads(raw)
            partial_answer = j.get("partial_answer", partial_answer)
            score = float(j.get("score", score))
        except Exception:
            pass
        return partial_answer, score

    async def _final_reduce(self, query: str, partials: List[Tuple[str, float]]) -> str:
        combined = "\n".join([a for a, _ in partials])
        system_msg = (
            "You are a query-focused summarizer. "
            "Combine the following partial answers into one final, coherent answer."
        )
        user_msg = f"QUERY: {query}\nPARTIAL ANSWERS:\n{combined}"
        return await call_chat_completion_async(system_prompt=system_msg, user_prompt=user_msg)

    async def answer_query(self, query: str, community_level: int) -> str:
        if community_level not in self.community_summaries:
            return f"No community summaries for level {community_level}."
        comm_items = list(self.community_summaries[community_level].items())
        random.shuffle(comm_items)
        partials = []
        tasks = []
        # Use self.chunk_size for splitting the community summary text
        for cid, summary_text in comm_items:
            words = summary_text.split()
            for i in range(0, len(words), self.chunk_size):
                seg = " ".join(words[i:i+self.chunk_size])
                tasks.append(self._generate_partial_answer(query, seg))
        results = await asyncio.gather(*tasks)
        for pa, score in results:
            if score > 0:
                partials.append((pa, score))
        partials.sort(key=lambda x: x[1], reverse=True)
        return await self._final_reduce(query, partials)

f = open("/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/results-covidqa.txt", "w")
async def main():
    # Path to your PDF file
    pdf_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/merged.pdf"
    # 1) Extract text from PDF
    pdf_text = []
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text.append(text)
    # Combine into a single string
    merged_content = "\n".join(pdf_text)
    # 2) Initialize GraphRAG with your desired chunk size/overlap.
    # The chunk and overlap sizes will be used for both GraphRAG and ES RAG.
    graphrag = GraphRAG(chunk_size=1000, overlap_size=150, max_gleanings=3, llm_context_size=8000)
    # 3) Chunk the single large PDF document
    graphrag.chunk_documents([merged_content])
    # 4) Build ES index from chunked content
    #    (We use the same chunks so both GraphRAG & ES RAG have the same textual segments).
    chunked_docs = graphrag.all_chunks  # list of text chunks
    index_name = "rag_pdf_index_temp"
    try:
        es_client.indices.delete(index=index_name)
    except Exception:
        pass
    try:
        drop_index_if_exists(driver, index_name)
    except Exception:
        pass
    vector_store = ElasticsearchStore.from_texts(
        texts=chunked_docs,
        embedding=embedding_function,
        es_connection=es_client,
        index_name=index_name
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # 5) Create the entity/relationship graph from the chunked text
    await graphrag.extract_element_instances()
    await graphrag.create_element_summaries()
    graphrag.build_graph_and_detect_communities()
    await graphrag.summarize_communities()
    # 6) Read CSV questions and references
    csv_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/QnA_combined_20241103.csv"
    es_cumulative_score = 0
    graph_cumulative_score = 0
    total_examples = 0
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row["question"]
            ground_truth = row["reference"]
            total_examples += 1
            # --- GraphRAG Query ---
            graph_answer = await graphrag.answer_query(question, community_level=1)
            # --- Elasticsearch RAG Query ---
            retrieved = retriever.invoke(question)
            es_context = "\n".join([d.page_content for d in retrieved])
            rag_prompt = f"""
Based on the provided context, answer the question.
Keep numeric or factual details verbatim.

Context:
{es_context}

Question: {question}
"""
            system_msg_es = "You are a query-focused summarizer. Generate an answer based solely on the provided context."
            es_answer = await call_chat_completion_async(system_prompt=system_msg_es, user_prompt=rag_prompt)
            # --- Scoring ---
            s_prompt_es = (
                f"Check if the candidate answer means the same as the ground truth. "
                f"Print 1 if yes, else 0.\n\nGround Truth: {ground_truth}\nCandidate (Elasticsearch): {es_answer}"
            )
            s_prompt_graph = (
                f"Check if the candidate answer means the same as the ground truth. "
                f"Print 1 if yes, else 0.\n\nGround Truth: {ground_truth}\nCandidate (GraphRAG): {graph_answer}"
            )
            evaluator_sys = "You are an evaluator."
            es_score_str = await call_chat_completion_async(system_prompt=evaluator_sys, user_prompt=s_prompt_es)
            graph_score_str = await call_chat_completion_async(system_prompt=evaluator_sys, user_prompt=s_prompt_graph)
            try:
                es_score = int(es_score_str.strip())
            except Exception:
                es_score = 0
            try:
                graph_score = int(graph_score_str.strip())
            except Exception:
                graph_score = 0
            es_cumulative_score += es_score
            graph_cumulative_score += graph_score
            print("===========================================")
            f.write(f"Question: {question}\n")
            print(f"Question: {question}")
            f.write("GraphRAG Answer: " + graph_answer + "\n")
            print("GraphRAG Answer:\n", graph_answer)
            print("Elasticsearch RAG Answer:\n", es_answer)
            f.write("Elasticsearch RAG Answer: " + es_answer + "\n")
            print("Reference:\n", ground_truth)
            f.write("Ground Truth: " + ground_truth + "\n")
            print()
            print("Scores for this example - Elasticsearch:", es_score, "GraphRAG:", graph_score)
            f.write("Scores for this example - Elasticsearch: " + str(es_score) + " GraphRAG: " + str(graph_score) + "\n")
            print("Cumulative Scores - Elasticsearch:", es_cumulative_score, "GraphRAG:", graph_cumulative_score)
            f.write("Cumulative Scores - Elasticsearch: " + str(es_cumulative_score) + " GraphRAG: " + str(graph_cumulative_score) + "\n")
            print("===========================================\n")
            f.write("===========================================\n")
    print(f"Total Examples: {total_examples}")
    f.write(f"Total Examples: {total_examples}\n")
    print(f"Final Elasticsearch Score: {es_cumulative_score}/{total_examples}")
    f.write(f"Final Elasticsearch Score: {es_cumulative_score}/{total_examples}\n")
    print(f"Final GraphRAG Score: {graph_cumulative_score}/{total_examples}")
    f.write(f"Final GraphRAG Score: {graph_cumulative_score}/{total_examples}\n")
    f.close()

if __name__ == "__main__":
    asyncio.run(main())