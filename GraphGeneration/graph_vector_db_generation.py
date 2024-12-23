import asyncio
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from codebase import display_graph
from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from neo4j_graphrag.indexes import create_vector_index, upsert_vector, drop_index_if_exists

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
openai_apikey=os.getenv("OPENAI_API_KEY")

INDEX_NAME = "transcript-1"

# ## Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Create an Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# Instantiate the LLM
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

# Instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    on_error="IGNORE",
    from_pdf=False,
)

# Generate an embedding for some text
f = open("datasets/transcripts/mortgage_loan_1_transcript.txt", "r")
mortgage_loan_transcript_1=''.join(f.read().splitlines())
# Run the pipeline on a piece of text
text = (
    mortgage_loan_transcript_1.strip()
)
asyncio.run(kg_builder.run_async(text=text))

# Create the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Chunk",
    embedding_property="embedding",
    dimensions=3072,
    similarity_fn="euclidean",
)

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# Uncomment to delete index 
# drop_index_if_exists(driver, INDEX_NAME)
