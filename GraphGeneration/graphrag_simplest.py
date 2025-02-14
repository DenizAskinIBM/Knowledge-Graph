import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import spacy
import networkx as nx

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
from urllib3.exceptions import InsecureRequestWarning
from dotenv import load_dotenv
warnings.simplefilter("ignore", InsecureRequestWarning)

# Load environment variables
load_dotenv()

# -----------------------
# 1. Setup & Credentials
# -----------------------
# (A) OpenAI

# (B) ElasticSearch (Optional if you still want to compare with ES)
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

# (C) spaCy for NER
nlp = spacy.load("en_core_web_sm")

# -------------------------
# 2. Text Preprocessing
# -------------------------
def chunk_paragraphs_into_sentences(text):
    """
    Splits a single document into sentence chunks using spaCy.
    Returns a list of sentence strings.
    """
    doc = nlp(text)
    sentence_chunks = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentence_chunks

# -------------------------
# 3. Entity Extraction
# -------------------------
def extract_entities_and_relations(sentence_list):
    """
    Extracts named entities from a list of sentences and creates relationships (entity -> MENTIONED_IN -> sentence).
    """
    entities = set()  
    relations = []

    for sentence in sentence_list:
        doc = nlp(sentence)
        sent_text = sentence.strip()
        sent_entities = [(ent.text, ent.label_) for ent in doc.ents]

        for ent_text, _ in sent_entities:
            relations.append((ent_text, "MENTIONED_IN", sent_text))
            entities.add(ent_text)
    return list(entities), relations

# -------------------------
# 4. Build Knowledge Graph
# -------------------------
def build_knowledge_graph(entities, relations):
    """
    Builds a directed graph of entity and sentence nodes using NetworkX.
    """
    G = nx.DiGraph()
    # Add entity nodes
    for ent in entities:
        G.add_node(ent, label="entity")

    # Add sentence nodes & edges
    for (entity, rel, sentence) in relations:
        if not G.has_node(sentence):
            G.add_node(sentence, label="sentence")
        G.add_edge(entity, sentence, relation=rel)
    return G

# -------------------------
# 5. Embedding-based Re-Ranking
# -------------------------
def embed_sentences(embedding_func, sentences):
    """
    Returns a list of sentence embeddings using the given huggingface embedding function.
    """
    return embedding_func.embed_documents(sentences)  # shape: (len(sentences), embedding_dim)

def retrieve_relevant_sentences(graph, query, top_k=3, embedding_func=None):
    """
    1) Find candidate sentences by entity match in the knowledge graph
    2) If no direct entity match, fallback to all sentences
    3) Re-rank candidate sentences using embedding similarity with the query
    """
    doc = nlp(query)
    query_entities = [ent.text for ent in doc.ents]
    candidate_sentences = set()

    # 1) Collect sentences from matched entities
    for node in graph.nodes:
        if node in query_entities and graph.nodes[node].get("label") == "entity":
            # Get outgoing edges to "sentence" nodes
            for edge in graph.out_edges(node):
                sent_node = edge[1]
                if graph.nodes[sent_node].get("label") == "sentence":
                    candidate_sentences.add(sent_node)

    # Fallback: no matches → take all sentence nodes
    if not candidate_sentences:
        candidate_sentences = {
            n for n, data in graph.nodes(data=True) if data.get("label") == "sentence"
        }

    # 2) Convert to a list so we can reorder them
    candidate_sentences_list = list(candidate_sentences)

    # 3) Re-rank with embeddings if embedding_func is provided
    if embedding_func:
        query_embedding = embedding_func.embed_query(query)  # shape: (embedding_dim,)
        sent_embeddings = embed_sentences(embedding_func, candidate_sentences_list)  # shape: (N, embedding_dim)

        # Compute dot-product or cosine similarity
        # HuggingFaceEmbeddings returns normal L2 vectors, so let's do a dot product
        # or we can do a naive approach with something like:
        import numpy as np
        query_embedding_np = np.array(query_embedding)
        sent_embeddings_np = np.array(sent_embeddings)

        # Dot product similarity
        similarities = sent_embeddings_np @ query_embedding_np

        # Sort descending by similarity
        sorted_indices = np.argsort(-similarities)
        ranked_sentences = [candidate_sentences_list[i] for i in sorted_indices]
        return ranked_sentences[:top_k]
    else:
        # If no embedding_func, return the first k
        return candidate_sentences_list[:top_k]

# -------------------------
# 6. Generate Answer
# -------------------------
def generate_answer_with_context(query, context):
    """
    Uses OpenAI chat completion to generate an answer based on the retrieved context.
    """
    context_str = "\n".join(context)
    prompt = f"""
You are a helpful assistant. Use the information in the context below to answer the question.

Context:
{context_str}

Question: {query}

Answer:
    """
    response = client.chat.completions.create(model="gpt-4",  # or "gpt-4o" if that is your internal alias
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300,
    temperature=0)
    return response.choices[0].message.content.strip()

# -------------------------
# 7. Main
# -------------------------
if __name__ == "__main__":
    from datasets import load_dataset
    dataset_name = "delucionqa"
    ragbench_dataset = load_dataset("rungalileo/ragbench", dataset_name, split="test")

    rag_score_number = 0
    graph_score_number = 0

    # For demonstration, use top_k=3
    k = 2

    for x in ragbench_dataset:
        question = x["question"]
        docs = x["documents"]
        ground_truth = x.response

        print("\n=================================")
        print(f"Q: {question}")

        # ---------------
        # (A) ES RAG
        # ---------------
        index_name = "rag_index_temp"
        try:
            es_client.indices.delete(index=index_name)
        except:
            pass

        # Insert documents as-is for ES search
        vector_store = ElasticsearchStore.from_texts(
            texts=docs,
            embedding=embedding_function,
            es_connection=es_client,
            index_name=index_name
        )
        es_retriever = vector_store.as_retriever(search_kwargs={"k": k})
        es_docs = es_retriever.invoke(question)
        es_context = [d.page_content for d in es_docs]
        rag_answer = generate_answer_with_context(question, es_context)

        # ---------------
        # (B) Graph RAG
        # ---------------
        # (1) Break docs into sentences
        all_sentences = []
        for doc_text in docs:
            all_sentences.extend(chunk_paragraphs_into_sentences(doc_text))

        # (2) Build knowledge graph
        entities, relations = extract_entities_and_relations(all_sentences)
        graph = build_knowledge_graph(entities, relations)

        # (3) Retrieve top-K sentences (entity + embedding re-rank)
        graph_rag_retrieved = retrieve_relevant_sentences(
            graph,
            question,
            top_k=k,
            embedding_func=embedding_function
        )
        graph_rag_answer = generate_answer_with_context(question, graph_rag_retrieved)

        print(f"RAG answer: {rag_answer}")
        print(f"GraphRAG answer: {graph_rag_answer}")
        print(f"Ground Truth: {ground_truth}")

        # (Optional) Evaluate correctness with GPT-4
        # (same approach as your original code)
        rag_prompt = f"Check if candidate answer means the same as ground truth. Print 1 if yes, else 0.\nGround Truth: {ground_truth}\nCandidate: {rag_answer}"
        rag_eval = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "system", "content": rag_prompt}],
        temperature=0)
        rag_score = rag_eval.choices[0].message.content.strip()

        graph_rag_prompt = f"Check if candidate answer means the same as ground truth. Print 1 if yes, else 0.\nGround Truth: {ground_truth}\nCandidate: {graph_rag_answer}"
        graph_rag_eval = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "system", "content": graph_rag_prompt}],
        temperature=0)
        graph_rag_score = graph_rag_eval.choices[0].message.content.strip()

        # Keep running totals
        try:
            rag_score_number += int(rag_score)
            graph_score_number += int(graph_rag_score)
        except:
            pass

        print("ES RAG Score (for now):", rag_score_number)
        print("GraphRAG Score (for now):", graph_score_number)
    print()
    print("Total ES RAG Score (for now):", rag_score_number)
    print("TotalGraphRAG Score (for now):", graph_score_number)