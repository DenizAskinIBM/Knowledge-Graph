#@ Copyright Deniz Askin - Edited by ChatGPT o1-pro

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import asyncio
import nest_asyncio
import numpy as np
import igraph
import leidenalg

from collections import defaultdict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.neighbors import NearestNeighbors

# ---------------------------------
# Tavily & Neo4j-GraphRAG Imports
# ---------------------------------
from tavily import TavilyClient
from codebase import delete_all_indexes
from codebase import display_graph
from codebase import context_retriever

from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index

nest_asyncio.apply()


def main(question, web_scraping_questions, index_name):
    # --------------------------
    # 1. Load Environment
    # --------------------------
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # --------------------------
    # 2. Tavily Setup & Queries
    # --------------------------
    tavily_client = TavilyClient()
    queries = web_scraping_questions

    # Collect responses from Tavily
    responses = []
    for q in queries:
        responses.append(tavily_client.search(q))

    # Extract the first search result's content from each response
    documents = []
    for resp in responses:
        documents.append(resp["results"][0]["content"])

    # --------------------------
    # 3. Neo4j Driver & LLM
    # --------------------------
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 2000,
            "temperature": 0,
        },
    )

    # --------------------------
    # 4. Embeddings & Splitter
    # --------------------------
    embedding_model = SentenceTransformerEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    transformer_model = embedding_model.model
    text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=10)

    # --------------------------
    # 5. SimpleKGPipeline
    # --------------------------
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedding_model,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=text_splitter
    )

    # --------------------------
    # 6. Build Knowledge Graph
    #    (Async Execution)
    # --------------------------
    async def run_pipeline_async(docs):
        """Runs the KG builder pipeline on each doc in docs."""
        if isinstance(docs, list):
            for doc in docs:
                text = doc.strip()
                await kg_builder.run_async(text=text)
        else:
            text = docs.strip()
            await kg_builder.run_async(text=text)

    asyncio.run(run_pipeline_async(documents))

    # --------------------------
    # 7. Create Vector Index
    # --------------------------
    # Determine embedding dimensions
    if isinstance(embedding_model, SentenceTransformerEmbeddings):
        dimensions = transformer_model.get_sentence_embedding_dimension()
    else:
        sample_embedding = transformer_model.embed_query("sample text")
        dimensions = len(sample_embedding)

    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=dimensions,
        similarity_fn="cosine",
    )

    # --------------------------
    # 8. Gather Node Embeddings
    # --------------------------
    node_ids = []
    embeddings = []

    with driver.session() as session:
        results = session.run("""
            MATCH (c:Chunk)
            RETURN elementId(c) AS nodeId, c.embedding AS embedding
        """)
        for record in results:
            node_ids.append(record["nodeId"])
            embeddings.append(record["embedding"])

    # Filter and validate embeddings
    embedding_dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], list) else 0
    valid_embeddings = []
    valid_node_ids = []

    for i, emb in enumerate(embeddings):
        if emb is None or not isinstance(emb, list) or len(emb) != embedding_dim:
            print(f"Invalid embedding at index {i}, replacing with zeros.")
            valid_embeddings.append([0] * embedding_dim)  # Placeholder embedding
        else:
            valid_embeddings.append(emb)
        valid_node_ids.append(node_ids[i])

    # Update node_ids and embeddings
    embeddings = valid_embeddings
    node_ids = valid_node_ids

    # Convert to NumPy array
    X = np.array(embeddings)

    # --------------------------
    # 9. k-Nearest Neighbors for Similarity
    # --------------------------
    k = 4
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    # --------------------------
    # 10. Leiden for Communities
    # --------------------------
    edges = []
    weights = []

    for i in range(len(node_ids)):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            # Cosine similarity = 1 - distance
            similarity = max(0, 1 - dist)
            edges.append((i, j_idx))
            weights.append(similarity)

    # Build igraph for Leiden
    g = igraph.Graph(n=len(node_ids), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=1.0
    )
    community_labels = partition.membership

    # --------------------------
    # 11. Create SIMILAR edges
    #     & Assign Communities
    #     in Batches (UNWIND)
    # --------------------------
    # 11a. Build batch data for edges
    edges_data = []
    for i in range(len(node_ids)):
        idA = node_ids[i]
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = 1 - dist  # 1 - cosine_distance
            idB = node_ids[j_idx]
            edges_data.append({"idA": idA, "idB": idB, "sim": sim})

    with driver.session() as session:
        # Remove old relationships
        session.run("MATCH ()-[r:SIMILAR]->() DELETE r")

        # Merge or create nodes + relationships
        session.run(
            """
            UNWIND $edges AS edge
            MERGE (a:Chunk { uuid: edge.idA })
            MERGE (b:Chunk { uuid: edge.idB })
            MERGE (a)-[r:SIMILAR]->(b)
            SET r.score = edge.sim
            """,
            {"edges": edges_data},
        )

    # --------------------------
    # 12. Ensure Each Node Has a UUID
    # --------------------------
    with driver.session() as session:
        session.run(
            """
            MATCH (c:Chunk)
            WHERE c.uuid IS NULL
            SET c.uuid = randomUUID()
            """
        )

    # --------------------------
    # 13. Assign Communities
    #     - Single pass UNWIND
    # --------------------------
    community_data = []
    for i in range(len(node_ids)):
        community_data.append({
            "idVal": node_ids[i],
            "community": int(community_labels[i])
        })

    with driver.session() as session:
        # Assign the computed community to matched nodes
        session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk { uuid: row.idVal })
            SET c.community = row.community
            """,
            {"rows": community_data},
        )

        # Any node not in the "community_data" set => default to -1
        session.run(
            """
            MATCH (c:Chunk)
            WHERE c.community IS NULL
            SET c.community = -1
            """
        )

    # --------------------------
    # 14. LLM-Based Naming
    # --------------------------
    # We'll use the 'llm' object to generate:
    # 1) a "community_label" for each community based on combined text
    # 2) a "node_name" for each node individually

    with driver.session() as session:
        # Retrieve each node's uuid, community, and text
        results = session.run(
            """
            MATCH (c:Chunk)
            RETURN c.uuid AS uuid, c.community AS community, c.text AS text
            ORDER BY c.uuid
            """
        )

        node_info = []
        communities_map = defaultdict(list)

        for record in results:
            uuid_ = record["uuid"]
            comm_ = record["community"]
            text_ = record["text"] if record["text"] else ""
            node_info.append({"uuid": uuid_, "community": comm_, "text": text_})
            communities_map[comm_].append(text_)

    # --- 14a. Name Each Community ---
    for comm_id, texts in communities_map.items():
        # Combine all text from nodes in this community
        combined_text = " ".join(t for t in texts if t.strip())

        # If there's no text, just use a placeholder
        if not combined_text.strip():
            community_label = f"Community_{comm_id}"
        else:
            # Prompt the LLM to create a short descriptive name
            prompt = (
                "You are an expert summarizer. Read the following text and provide a concise, "
                f"descriptive name for this cluster:\n\n{combined_text}"
            )
            response = llm.invoke(prompt)
            community_label = response.content.strip()

        # Update nodes in that community with this label
        with driver.session() as session:
            session.run(
                """
                MATCH (c:Chunk)
                WHERE c.community = $comm_id
                SET c.community_label = $label
                """,
                {"comm_id": comm_id, "label": community_label},
            )

    # --- 14b. Name Each Node Individually ---
    for node in node_info:
        node_uuid = node["uuid"]
        node_text = node["text"]

        if not node_text.strip():
            node_label = f"Node_{node_uuid}"
        else:
            prompt = (
                "You are an expert summarizer. Read the following text and provide a concise, "
                f"descriptive name for this node:\n\n{node_text}"
            )
            response = llm.invoke(prompt)
            node_label = response.content.strip()

        with driver.session() as session:
            session.run(
                """
                MATCH (c:Chunk {uuid: $uuid})
                SET c.node_name = $node_name
                """,
                {"uuid": node_uuid, "node_name": node_label},
            )

    # --------------------------
    # 15. Print Final Results
    # --------------------------
    with driver.session() as session:
        print("Final Node Info with LLM-based Naming:")
        final_results = session.run(
            """
            MATCH (c:Chunk)
            RETURN c.uuid AS uuid, c.community AS community,
                   c.community_label AS community_label,
                   c.node_name AS node_name
            ORDER BY c.uuid
            """
        )
        print()
        for record in final_results:
            print(
                f"UUID: {record['uuid']}, "
                f"community: {record['community']}, "
                f"community_label: {record['community_label']}, "
                f"node_name: {record['node_name']}"
            )
        print()
    
    top_k = 10
    retriever = VectorRetriever(driver, index_name, embedding_model)
    graph_rag = GraphRAG(retriever=retriever, llm=llm)
    print("Question:", question)
    print("\nGRAPH RAG RESULTS:")
    response = graph_rag.search(query_text=question, retriever_config={"top_k": top_k}, return_context=True)
    print(response.answer)
    driver.close()
    display_graph()

if __name__ == "__main__":
    question = "What is the similarity between Leo Messi and System of a Down?"
    web_scraping_questions = ["Who is Leo Messi?", "Who is Till Lindemann?", "What is System of a Down"]
    index_name = "messi_lindemann_soad"
    main(question, web_scraping_questions, index_name)