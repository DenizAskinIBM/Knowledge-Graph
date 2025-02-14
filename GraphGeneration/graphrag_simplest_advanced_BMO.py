import os
from openai import OpenAI, AsyncOpenAI

client = OpenAI()
aclient = AsyncOpenAI()
import networkx as nx
import json
import asyncio
import urllib3
import re
import csv
import random
import PyPDF2  # For reading PDF
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

# Attempt to import igraph and leidenalg
try:
    import igraph
    import leidenalg
    HAVE_LEIDEN = True
except ImportError:
    HAVE_LEIDEN = False
    print("Warning: 'igraph' or 'leidenalg' is not installed. Community detection will be skipped.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Asynchronous helper to call GPT
# ---------------------------
async def call_chat_completion_async(system_prompt: str, user_prompt: str) -> str:
    try:
        response = await aclient.chat.completions.create(model="gpt-4o",  # Keeping the same model usage from original Code #1
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error in call_chat_completion_async:", e)
        return ""

# ---------------------------
# Helper to robustly parse evaluator scores
# ---------------------------
def parse_score(text: str) -> int:
    text = text.strip()
    if text.startswith("1"):
        return 1
    elif text.startswith("0"):
        return 0
    match = re.search(r'\d', text)
    if match:
        return int(match.group(0))
    return 0

# ---------------------------
# GraphRAG Class
# ---------------------------
class GraphRAG:
    def __init__(self, chunk_size=1000, overlap_size=150, max_gleanings=3, llm_context_size=8000):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.max_gleanings = max_gleanings
        self.llm_context_size = llm_context_size

        self.all_chunks = []         # List of text chunks
        self.element_instances = []  # List of extracted entity/relationship instances per chunk
        self.element_summaries = {}  # Summaries for entities and relationships
        self.G = None                # The constructed knowledge graph (NetworkX)
        self.hierarchy_levels = []   # Community membership arrays (one per resolution)
        self.community_summaries = {}  # {level: {community_id: summary}}

    # 1. Document Chunking
    def chunk_documents(self, documents: list):
        for doc in documents:
            words = doc.split()
            i = 0
            while i < len(words):
                end = i + self.chunk_size
                chunk = " ".join(words[i:end])
                self.all_chunks.append(chunk)
                i += (self.chunk_size - self.overlap_size)

    # 2. Entity and Relationship Extraction
    async def _extract_entities_and_relationships_once(self, chunk: str):
        system_prompt = (
            "You are an expert information extraction assistant. "
            "Extract all significant entities and relationships from the text. "
            "Return JSON with keys 'entities' and 'relationships'. "
            "Each entity should have keys 'entity' (or 'name'), 'type', and 'description'; "
            "each relationship should have keys 'source', 'target', and 'description'."
        )
        user_prompt = f"Text:\n{chunk}\nExtract entities and relationships as JSON."
        response = await call_chat_completion_async(system_prompt, user_prompt)
        entities, relationships = [], []
        try:
            data = json.loads(response)
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
        except Exception:
            pass
        return entities, relationships

    async def _check_for_missed_entities(self, chunk: str, iteration: int) -> bool:
        system_prompt = "You are a vigilant assistant. Answer only with 'Yes' or 'No'."
        user_prompt = f"Review the following text. Did you miss any significant entities or relationships?\n\n{chunk}"
        answer = await call_chat_completion_async(system_prompt, user_prompt)
        return answer.strip().lower().startswith("yes")

    async def _extract_additional_entities(self, chunk: str):
        system_prompt = (
            "You are an information extraction assistant. Some important details may have been missed. "
            "Please extract any additional significant entities and relationships from the text. Return JSON."
        )
        user_prompt = f"Text:\n{chunk}\nExtract additional entities and relationships as JSON."
        response = await call_chat_completion_async(system_prompt, user_prompt)
        entities, relationships = [], []
        try:
            data = json.loads(response)
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
        except Exception:
            pass
        return entities, relationships

    async def _process_chunk(self, chunk: str):
        entities, relationships = await self._extract_entities_and_relationships_once(chunk)
        all_entities = list(entities)
        all_relationships = list(relationships)
        for i in range(self.max_gleanings):
            if not await self._check_for_missed_entities(chunk, i):
                break
            new_entities, new_relationships = await self._extract_additional_entities(chunk)
            all_entities.extend(new_entities)
            all_relationships.extend(new_relationships)
        return {"chunk": chunk, "entities": all_entities, "relationships": all_relationships}

    async def extract_element_instances(self):
        tasks = [self._process_chunk(chunk) for chunk in self.all_chunks]
        self.element_instances = await asyncio.gather(*tasks)

    # 3. Summarization of Entities and Relationships
    async def _summarize_entity(self, etype: str, ename: str, descriptions: list, eid: int):
        system_prompt = "You are a summarizer. Given multiple descriptions of an entity, produce a concise summary of its essence."
        user_prompt = "\n".join(descriptions)
        summary = await call_chat_completion_async(system_prompt, user_prompt)
        key = f"entity_{eid}"
        return key, {"type": etype, "name": ename, "summary": summary}

    async def _summarize_relationship(self, src: str, tgt: str, descriptions: list, eid: int):
        system_prompt = "You are a summarizer. Given several descriptions of a relationship, produce a concise summary highlighting its key details."
        user_prompt = "\n".join(descriptions)
        summary = await call_chat_completion_async(system_prompt, user_prompt)
        key = f"rel_{eid}"
        return key, {"source": src, "target": tgt, "summary": summary}

    async def create_element_summaries(self):
        from collections import defaultdict
        entity_map = defaultdict(list)
        relationship_map = defaultdict(list)
        for instance in self.element_instances:
            for ent in instance["entities"]:
                if isinstance(ent, dict):
                    # Some JSON fields might use "entity" or "name" interchangeably
                    name_value = ent.get("entity", "") if "entity" in ent else ent.get("name", "")
                    key = (ent.get("type", ""), name_value)
                    desc = ent.get("description", "")
                else:
                    try:
                        ent_obj = json.loads(ent)
                        name_value = ent_obj.get("entity", "") if "entity" in ent_obj else ent_obj.get("name", "")
                        key = (ent_obj.get("type", ""), name_value)
                        desc = ent_obj.get("description", "")
                    except Exception:
                        continue
                entity_map[key].append(desc)
            for rel in instance["relationships"]:
                if isinstance(rel, dict):
                    src = rel.get("source", "")
                    tgt = rel.get("target", "")
                    key = (src, tgt)
                    desc = rel.get("description", "")
                else:
                    try:
                        rel_obj = json.loads(rel)
                        src = rel_obj.get("source", "")
                        tgt = rel_obj.get("target", "")
                        key = (src, tgt)
                        desc = rel_obj.get("description", "")
                    except Exception:
                        continue
                relationship_map[key].append(desc)
        self.element_summaries = {}

        # Summarize entities
        entity_items = list(entity_map.items())
        entity_tasks = [self._summarize_entity(etype, ename, descs, i)
                        for i, ((etype, ename), descs) in enumerate(entity_items)]
        results = await asyncio.gather(*entity_tasks)
        for key, summary in results:
            self.element_summaries[key] = summary

        # Summarize relationships
        relationship_items = list(relationship_map.items())
        start_id = len(entity_items)
        relationship_tasks = [self._summarize_relationship(src, tgt, descs, i)
                              for i, ((src, tgt), descs) in enumerate(relationship_items, start=start_id)]
        results = await asyncio.gather(*relationship_tasks)
        for key, summary in results:
            self.element_summaries[key] = summary

    # 4. Graph Construction and Community Detection
    def build_graph_and_detect_communities(self):
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

        if HAVE_LEIDEN:
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
                partition = leidenalg.find_partition(
                    ig,
                    leidenalg.RBConfigurationVertexPartition,
                    weights=ig.es["weight"],
                    resolution_parameter=r
                )
                self.hierarchy_levels.append(partition.membership)
        else:
            print("Leiden algorithm not available; skipping community detection.")

    # Summarize communities
    async def _summarize_community(self, level_idx: int, cid: int, context: str):
        system_prompt = "You are a domain-specific summarizer. Given the following texts from a community, produce a concise, insightful summary."
        summary = await call_chat_completion_async(system_prompt, context)
        return cid, summary

    async def summarize_communities(self):
        if not self.hierarchy_levels:
            return
        node_summaries = {n: data.get("summary", "") for n, data in self.G.nodes(data=True)}
        nx_nodes = list(self.G.nodes)

        # We'll use the first (coarse) resolution level for final grouping
        membership = self.hierarchy_levels[0]
        level_comms = {}
        for idx, comm_id in enumerate(membership):
            text = node_summaries.get(nx_nodes[idx], "")
            level_comms.setdefault(comm_id, []).append(text)

        self.community_summaries = {0: {}}
        tasks = []
        for cid, texts in level_comms.items():
            context = " ".join(texts)
            tasks.append(self._summarize_community(0, cid, context))
        results = await asyncio.gather(*tasks)
        for cid, summary in results:
            self.community_summaries[0][cid] = summary

    # Query answer
    async def answer_query(self, query: str) -> str:
        if not self.community_summaries or 0 not in self.community_summaries:
            return "No community summaries available."
        comm_context = " ".join(self.community_summaries[0].values())
        system_prompt = "You are a query-focused assistant. Use the following context to answer the question."
        user_prompt = f"Question: {query}\nContext: {comm_context}"
        return await call_chat_completion_async(system_prompt, user_prompt)

# ---------------------------
# Elasticsearch RAG Functions
# ---------------------------
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
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)

# Helper: Compute sentence embeddings
def embed_sentences(embedding_func, sentences):
    return embedding_func.embed_documents(sentences)

# We keep this function (although we'll use the simpler retrieval approach).
def retrieve_relevant_sentences(graph, query, top_k=5, embedding_func=None):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    query_entities = [ent.text for ent in doc.ents]
    candidate_sentences = set()

    for node in graph.nodes:
        if node in query_entities and graph.nodes[node].get("label") == "entity":
            for edge in graph.out_edges(node):
                sent_node = edge[1]
                if graph.nodes[sent_node].get("label") == "sentence":
                    candidate_sentences.add(sent_node)

    if not candidate_sentences:
        candidate_sentences = {n for n, data in graph.nodes(data=True) if data.get("label") == "sentence"}

    candidate_sentences_list = list(candidate_sentences)
    if embedding_func:
        query_embedding = embedding_func.embed_query(query)
        sent_embeddings = embed_sentences(embedding_func, candidate_sentences_list)
        import numpy as np
        query_embedding_np = np.array(query_embedding)
        sent_embeddings_np = np.array(sent_embeddings)
        similarities = sent_embeddings_np @ query_embedding_np
        sorted_indices = np.argsort(-similarities)
        ranked_sentences = [candidate_sentences_list[i] for i in sorted_indices]
        return ranked_sentences[:top_k]
    else:
        return candidate_sentences_list[:top_k]

def es_retrieve(question, documents, k=5):
    """
    Build a temporary vector store from the given documents and retrieve top-k matches.
    (Kept from the original code but not used directly below, since we do once-per-run indexing.)
    """
    index_name = "rag_index_temp"
    vector_store = ElasticsearchStore.from_texts(
        texts=documents,
        embedding=embedding_function,
        es_connection=es_client,
        index_name=index_name
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    embedding_results = [d.page_content for d in retriever.invoke(question)]
    return embedding_results

def generate_answer_with_context(query, context):
    context_str = "\n".join(context)
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context_str}

Question: {query}

Answer:
    """
    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200,
    temperature=0)
    return response.choices[0].message.content.strip()

# ---------------------------
# Main Execution
# ---------------------------
async def main():
    # 1) Read and merge PDF file
    pdf_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/merged.pdf"
    pdf_texts = []
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_texts.append(text)
    merged_content = "\n".join(pdf_texts)

    # 2) Initialize GraphRAG with chunk_size=1000, overlap_size=150, gleanings=3
    graphrag = GraphRAG(chunk_size=1000, overlap_size=150, max_gleanings=3, llm_context_size=8000)

    # 3) Chunk the PDF and build ES index from the chunks
    graphrag.chunk_documents([merged_content])
    chunked_docs = graphrag.all_chunks
    index_name = "rag_pdf_index_temp"
    try:
        es_client.indices.delete(index=index_name)
    except Exception:
        pass
    vector_store = ElasticsearchStore.from_texts(
        texts=chunked_docs,
        embedding=embedding_function,
        es_connection=es_client,
        index_name=index_name
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 4) Create the knowledge graph
    await graphrag.extract_element_instances()
    await graphrag.create_element_summaries()
    graphrag.build_graph_and_detect_communities()
    await graphrag.summarize_communities()

    # 5) Prepare to iterate over CSV question/answer reference data
    csv_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/QnA_combined_20241103.csv"
    f = open("/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/results-graph_simplest_advanced_BMO.txt", "w")

    es_rag_total_score = 0
    graph_rag_total_score = 0
    total_examples = 0

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row["question"]
            ground_truth = row["reference"]
            total_examples += 1

            # --- Elasticsearch RAG Answer ---
            retrieved_docs = retriever.invoke(question)
            es_context = "\n".join(d.page_content for d in retrieved_docs)

            es_system_prompt = "You are a query-focused summarizer. Generate an answer based solely on the provided context."
            es_user_prompt = f"Context:\n{es_context}\nQuestion: {question}"
            es_answer = await call_chat_completion_async(es_system_prompt, es_user_prompt)

            # --- GraphRAG Answer ---
            graph_answer = await graphrag.answer_query(question)

            # --- Evaluate with GPT (1 for correct, 0 for not) ---
            evaluator_system_prompt = "You are an evaluator."

            # Evaluate ES
            evaluator_prompt_es = (
                f"Ground Truth: {ground_truth}\nCandidate (Elasticsearch RAG): {es_answer}\n"
                f"Does the candidate answer mean the same as the ground truth? Answer with 1 for yes or 0 for no."
            )
            es_eval_response = await call_chat_completion_async(evaluator_system_prompt, evaluator_prompt_es)
            es_score = parse_score(es_eval_response)

            # Evaluate Graph
            evaluator_prompt_graph = (
                f"Ground Truth: {ground_truth}\nCandidate (GraphRAG): {graph_answer}\n"
                f"Does the candidate answer mean the same as the ground truth? Answer with 1 for yes or 0 for no."
            )
            graph_eval_response = await call_chat_completion_async(evaluator_system_prompt, evaluator_prompt_graph)
            graph_score = parse_score(graph_eval_response)

            es_rag_total_score += es_score
            graph_rag_total_score += graph_score

            # --- Print & Write results ---
            print("===========================================")
            f.write("===========================================\n")
            print(f"Question: {question}")
            f.write(f"Question: {question}\n")
            print("Elasticsearch RAG Answer:", es_answer)
            f.write("Elasticsearch RAG Answer: " + es_answer + "\n")
            print("GraphRAG Answer:", graph_answer)
            f.write("GraphRAG Answer: " + graph_answer + "\n")
            print("Ground Truth:", ground_truth)
            f.write("Ground Truth: " + ground_truth + "\n")

            print("\nScores for this example:")
            print(f"  ES Score: {es_score}   GraphRAG Score: {graph_score}")
            f.write(f"\nScores for this example:\n  ES Score: {es_score}   GraphRAG Score: {graph_score}\n")

            print("\nCumulative Scores so far:")
            print(f"  ES RAG: {es_rag_total_score}   GraphRAG: {graph_rag_total_score}")
            f.write(f"Cumulative Scores so far:\n  ES RAG: {es_rag_total_score}   GraphRAG: {graph_rag_total_score}\n")
            print("===========================================\n")
            f.write("===========================================\n\n")

    # Final results
    print(f"Total Examples: {total_examples}")
    f.write(f"Total Examples: {total_examples}\n")
    print(f"Final ES RAG Score: {es_rag_total_score}/{total_examples}")
    f.write(f"Final ES RAG Score: {es_rag_total_score}/{total_examples}\n")
    print(f"Final GraphRAG Score: {graph_rag_total_score}/{total_examples}")
    f.write(f"Final GraphRAG Score: {graph_rag_total_score}/{total_examples}\n")
    f.close()

if __name__ == "__main__":
    asyncio.run(main())