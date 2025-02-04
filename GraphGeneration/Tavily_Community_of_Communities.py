## Copyright Deniz Askin, 2025. ChatGPT 01-pro was used to edit this code.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tavily import TavilyClient
from codebase import delete_all_indexes
from neo4j_graphrag.generation import GraphRAG
from neo4j import GraphDatabase
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from codebase import display_graph, context_retriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index

from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()

import asyncio
import numpy as np
import itertools
from collections import defaultdict


def main(question, web_scraping_questions):
    # ========== Utility Functions for LLM-based Naming ==========
    def generate_name_for_text(llm, text, prompt="Give a short descriptive name for this text:"):
        """
        LLM-based name for a single chunk of text. 
        """
        if not text.strip():
            return "NoContent"
        response = llm.invoke(
            f"""{prompt}
    Text:
    {text}

    Short name:"""
        )
        return response.content.strip()

    def generate_name_for_group(llm, texts, group_prompt="Name this community based on the following texts:"):
        """
        LLM-based name for a group of texts (e.g., a community). 
        """
        if not texts:
            return "EmptyGroup"
        snippet = "\n\n".join(texts[:10])  # limit to first 10 to avoid huge prompt
        response = llm.invoke(
            f"""{group_prompt}
    Here are some representative texts of this group:
    {snippet}

    Short descriptive name for this group:"""
        )
        return response.content.strip()

    # ========== Instantiate Tavily + LLM ==========
    tavily_client = TavilyClient()

    queries = ["Who is Leo Messi?", "Who is Till Lindemann?", "What is System of a Down"]
    responses = []
    for x in queries:
        responses.append(tavily_client.search(x))

    documents = []
    for resp in responses:
        content = resp["results"][0].get("content", "")
        if content and content.strip():
            documents.append(content.strip())

    # If there's only one doc, ensure we have a list
    if not isinstance(documents, list):
        documents = [documents]

    # Filter out empty docs
    documents = [doc for doc in documents if doc.strip()]

    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    index_name = "messi_lindemann_soad"

    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 2000,
            "temperature": 0,
        },
    )

    embedding_model = SentenceTransformerEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    transformer_model = embedding_model.model

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=10)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedding_model,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=text_splitter
    )

    # ========== 1) Ingest Documents ==========
    for doc in documents:
        asyncio.run(kg_builder.run_async(text=doc))

    # ========== 2) Create Vector Index ==========
    dims = transformer_model.get_sentence_embedding_dimension()
    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=dims,
        similarity_fn="cosine",
    )

    # ========== 3) Remove truly invalid nodes (no embedding or text) ==========
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL OR c.text IS NULL OR c.text = ''
            DETACH DELETE c
        """)

    # ========== 3.5) Ensure all existing Chunks have a uuid ==========
    #     (In case some nodes were created without a uuid property)
    with driver.session() as session:
        session.run("""
            MATCH (c:Chunk)
            WHERE c.uuid IS NULL
            SET c.uuid = elementId(c)
        """)

    # ========== 4) Retrieve remaining chunks ==========
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
        print("No valid chunks found. Exiting.")
        driver.close()
        exit()

    # ========== 5) First-Level Community Detection ==========
    from sklearn.neighbors import NearestNeighbors
    import igraph
    import leidenalg

    k = 3
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
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

    g = igraph.Graph(n=num_nodes, edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=1.0
    )
    community_labels = partition.membership  # each node's first-level community

    # ========== 6) Write SIMILAR edges + community label to Neo4j ==========
    with driver.session() as session:
        # Clear old SIMILAR edges if any
        session.run("MATCH ()-[r:SIMILAR]->() DELETE r")

        for i in range(num_nodes):
            this_id = node_ids[i]
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                sim = 1 - dist
                that_id = node_ids[j_idx]
                session.run(
                    """
                    MERGE (a:Chunk { uuid: $idA })
                    SET a.uuid = $idA
                    MERGE (b:Chunk { uuid: $idB })
                    SET b.uuid = $idB
                    MERGE (a)-[r:SIMILAR]->(b)
                    SET r.score = $sim
                    """,
                    {"idA": this_id, "idB": that_id, "sim": sim},
                )

        # Write first-level community label
        for i in range(num_nodes):
            this_id = node_ids[i]
            comm = int(community_labels[i])
            session.run("""
                MERGE (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal,
                    c.community = $community
            """, {"idVal": this_id, "community": comm})

    # ========== 7) LLM-based naming of chunks + first-level communities ==========
    # 7.1 Name each chunk
    for i in range(num_nodes):
        chunk_text = node_text_map[node_ids[i]]
        chunk_name = generate_name_for_text(llm, chunk_text, 
            prompt="Give a short descriptive name for this chunk's content:")
        with driver.session() as session:
            session.run("""
                MERGE (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal,
                    c.name = $chunkName
            """, {"idVal": node_ids[i], "chunkName": chunk_name})

    # 7.2 Name first-level communities
    community_to_nodes = defaultdict(list)
    for i, nid in enumerate(node_ids):
        comm = community_labels[i]
        community_to_nodes[comm].append(nid)

    community_name_map = {}
    for comm_id, nodelist in community_to_nodes.items():
        chunk_texts = [node_text_map[nid] for nid in nodelist]
        comm_label = generate_name_for_group(
            llm, chunk_texts,
            group_prompt="Name this first-level community based on its chunk texts:"
        )
        community_name_map[comm_id] = comm_label

    with driver.session() as session:
        for comm_id, c_name in community_name_map.items():
            session.run("""
                MATCH (c:Chunk)
                WHERE c.community = $commId
                SET c.community_name = $cName
            """, {"commId": comm_id, "cName": c_name})

    # ========== 8) Second-Level Communities ==========
    # 8.1 Compute centroids for each first-level community
    community_centroids = {}
    for comm_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        mean_vec = np.mean(subset_emb, axis=0)
        community_centroids[comm_id] = mean_vec

    comm_ids = sorted(community_centroids.keys())
    comm_index = {c: idx for idx, c in enumerate(comm_ids)}
    num_comms = len(comm_ids)

    edges2 = []
    weights2 = []

    import math

    for c1, c2 in itertools.combinations(comm_ids, 2):
        v1 = community_centroids[c1]
        v2 = community_centroids[c2]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-12:
            sim = 0
        else:
            sim = np.dot(v1, v2) / denom
        sim = max(0, sim)
        edges2.append((comm_index[c1], comm_index[c2]))
        weights2.append(sim)

    g2 = igraph.Graph(n=num_comms, edges=edges2, directed=False)
    g2.es["weight"] = weights2

    partition2 = leidenalg.find_partition(
        g2,
        leidenalg.RBConfigurationVertexPartition,
        weights=g2.es["weight"],
        resolution_parameter=1.0
    )
    super_community_labels = partition2.membership

    community_to_super_community = {
        comm_ids[i]: super_community_labels[i] for i in range(num_comms)
    }

    # 8.2 Write second-level labels to each chunk
    with driver.session() as session:
        for i in range(num_nodes):
            this_id = node_ids[i]
            first_comm = community_labels[i]
            second_comm = community_to_super_community[first_comm]
            session.run("""
                MERGE (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal,
                    c.super_community = $sc
            """, {"idVal": this_id, "sc": second_comm})

    # 8.3 Name the super-communities
    super_comm_to_first_comm = defaultdict(list)
    for fc in comm_ids:
        sc = community_to_super_community[fc]
        super_comm_to_first_comm[sc].append(fc)

    super_community_name_map = {}
    for sc, fc_list in super_comm_to_first_comm.items():
        fc_names = [community_name_map[fc] for fc in fc_list]
        sc_name = generate_name_for_group(
            llm, fc_names,
            group_prompt="Name this second-level community based on the first-level community names:"
        )
        super_community_name_map[sc] = sc_name

    # Write super_community_name back to the chunks
    with driver.session() as session:
        for i in range(num_nodes):
            sc = community_to_super_community[community_labels[i]]
            sc_name = super_community_name_map[sc]
            session.run("""
                MERGE (c:Chunk { uuid: $idVal })
                SET c.uuid = $idVal,
                    c.super_community_name = $scName
            """, {"idVal": node_ids[i], "scName": sc_name})

    # ========== 9) Print Hierarchy Check ==========
    print("\n=== Hierarchy of Communities ===")
    for i, node_id in enumerate(node_ids):
        fc = community_labels[i]
        sc = community_to_super_community[fc]
        fc_name = community_name_map[fc]
        sc_name = super_community_name_map[sc]
        print(f"Node {node_id}: c1={fc}({fc_name}), c2={sc}({sc_name})")

    # ========== 10) Print the Final List of All Chunks ==========
    print("\n=== Final List of All Chunks with Names and Communities ===")
    with driver.session() as session:
        final_results = session.run("""
            MATCH (c:Chunk)
            RETURN
                c.uuid AS uuid,
                c.name AS chunkName,
                c.community_name AS communityName,
                c.super_community_name AS superCommunityName
            ORDER BY uuid
        """)
        for record in final_results:
            print(f"Chunk {record['uuid']}:")
            print(f"  - chunk name:          {record['chunkName']}")
            print(f"  - community name:      {record['communityName']}")
            print(f"  - super-community name:{record['superCommunityName']}")
            print()
    
     # ========== 11) Basic RAG Tests ==========
    retriever = VectorRetriever(driver, index_name, embedding_model)

    question = (
        question
    )

    top_k = 10
    graph_rag = GraphRAG(retriever=retriever, llm=llm)

    print("\nQUESTION:", question)
    print("\nTRADITIONAL RAG RESULTS:")
    context = retriever.search(query_text=question, top_k=top_k)
    rag_answer_1 = llm.invoke(
        "Based on this context: " + str(context) + f"\nAnswer the question: {question}"
    )
    print(rag_answer_1.content)

    print("\nGRAPH RAG RESULTS:")
    response = graph_rag.search(query_text=question, retriever_config={"top_k": top_k}, return_context=True)
    print(response.answer)

    # ========== 11) Example of Filtering by Community in RAG ==========
    community_id=0
    print(f"\nFiltering to community={community_id} for a sample question:")
    filter_question = "What is the main theme of the context you receive?"
    response_filtered = graph_rag.search(
        query_text=filter_question,
        retriever_config={
            "top_k": top_k,
            "filters": {"community": 0}
        },
        return_context=True
    )
    print(response_filtered.answer)

    # ========== 13) Display the Graph & Close ==========
    driver.close()
    display_graph()

if __name__ == "__main__":
    question = "What is the similarity between Leo Messi and System of a Down?"
    web_scraping_questions = ["Who is Leo Messi?", "Who is Till Lindemann?", "What is System of a Down"]
    main(question, web_scraping_questions)