import spacy
from sentence_transformers import SentenceTransformer
import networkx as nx
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import os
import numpy as np
from dotenv import load_dotenv
import PyPDF2
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Initialize environment and models
load_dotenv()
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
graph = nx.Graph()

def process_text(text):
    """Create overlapping text chunks with entity awareness"""
    doc = nlp(text)
    chunks = []
    current_chunk = []

    for sent in doc.sents:
        entities = [ent.text for ent in sent.ents]
        current_chunk.append(sent.text)

        # Split chunk if it contains multiple entities
        if len(entities) > 1 and len(current_chunk) > 1:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent.text]  # Overlap

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def build_knowledge_graph(chunks):
    """Build graph with entity relationships and global embeddings"""
    graph.clear()
    global_doc_embedding = embedder.encode(" ".join(chunks))

    for chunk in chunks:
        doc = nlp(chunk)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Add nodes with global context
        for entity, label in entities:
            if not graph.has_node(entity):
                local_embed = embedder.encode(entity)
                combined_embed = np.concatenate([local_embed, global_doc_embedding])
                graph.add_node(entity, 
                             embedding=combined_embed,
                             type=label,
                             global_score=0.0)

        # Create relationships with weights
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                e1, e2 = entities[i][0], entities[j][0]
                if graph.has_edge(e1, e2):
                    graph[e1][e2]["weight"] += 1
                else:
                    graph.add_edge(e1, e2, weight=1)

def global_search(question_embed, top_k=3):
    """Search across entire graph using semantic similarity"""
    similarities = []
    question_embed = question_embed / np.linalg.norm(question_embed)

    for node in graph.nodes():
        node_embed = graph.nodes[node]['embedding']
        node_embed = node_embed / np.linalg.norm(node_embed)
        sim = np.dot(question_embed, node_embed)
        similarities.append((node, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [n[0] for n in similarities[:top_k]]

def local_search(seed_nodes, depth=2):
    """Expand from seed nodes using graph connections"""
    local_context = set()

    for node in seed_nodes:
        # Get multi-hop neighbors
        neighbors = nx.single_source_shortest_path_length(graph, node, cutoff=depth)
        local_context.update(neighbors.keys())

    return list(local_context)

def rag_query(question, top_k=2):
    """Combine global and local search strategies"""
    # Encode question with same dimensions
    chunk_embed = embedder.encode(" ".join(graph.nodes()))
    question_embed = np.concatenate([embedder.encode(question), chunk_embed])

    # Global phase
    global_nodes = global_search(question_embed, top_k=top_k)

    # Local phase
    local_nodes = local_search(global_nodes, depth=2)

    # Combine and rank results
    all_nodes = list(set(global_nodes + local_nodes))

    # Create context map
    context = []
    for node in all_nodes:
        connections = []
        for neighbor in graph.neighbors(node):
            weight = graph[node][neighbor]["weight"]
            connections.append(f"{neighbor} ({weight} links)")
        context.append(f"{node} connects to: {', '.join(connections[:3])}")

    # Generate response
    response = client.chat.completions.create(model="gpt-4",
    messages=[{
        "role": "system",
        "content": f"Global-Local Context:\n{chr(10).join(context)}\n\nAnswer concisely:"
    }, {
        "role": "user",
        "content": question
    }],
    temperature=0.5,
    max_tokens=256)
    return response.choices[0].message.content

# Usage example
if __name__ == "__main__":
    # Load and process document
    pdf_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/merged.pdf"
    pdf_text = []
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            if text := page.extract_text():
                pdf_text.append(text)

    chunks = process_text("\n".join(pdf_text))
    build_knowledge_graph(chunks)
    import csv
    csv_path = "/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/QnA_combined_20241103.csv"
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        graph_rag_score=0
        for row in reader:
            question = row["question"]
            ground_truth = row["reference"]
            # Generate response
            response = client.chat.completions.create(model="gpt-4",
            messages=[{
                "role": "system",
                "content": f"Check if the candidate answer means the same as the ground truth. Print 1 if yes, else 0. Ground Truth: {ground_truth}"
            }, {
                "role": "user",
                "content": "Answer: "+str(rag_query(question))
            }],
            temperature=0,
            max_tokens=256)
            print("Question:", question)
            graph_rag_score+=int(response.choices[0].message.content)
            print("GraphRAG Score:",graph_rag_score)
            print("=============================================================")
            print()