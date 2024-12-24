import asyncio
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from codebase import display_graph
from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.pipeline.types import (
    EntityInputType,
    RelationInputType,
)
from neo4j_graphrag.indexes import create_vector_index, upsert_vector, drop_index_if_exists
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
import os
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
openai_apikey=os.getenv("OPENAI_API_KEY")

INDEX_NAME = "all-transcripts"

## Connect to the Neo4j database
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
    text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
    on_error="IGNORE",
    from_pdf=False)

# Generate an embedding for some text
f = open("datasets/transcripts/mortgage_loan_1_transcript.txt", "r")
mortgage_loan_transcript_1=''.join(f.read().splitlines())
f = open("datasets/transcripts/mortgage_loan_2_transcript.txt", "r")
mortgage_loan_transcript_2=''.join(f.read().splitlines())
f = open("datasets/transcripts/mortgage_loan_3_transcript.txt", "r")
mortgage_loan_transcript_3=''.join(f.read().splitlines())
transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]

for x in transcripts:
# Run the pipeline on a piece of text
    text = (
        x.strip()
    )
    asyncio.run(kg_builder.run_async(text=text))

# Create the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Chunk",
    embedding_property="embedding",
    dimensions=3072,
    similarity_fn="cosine",
)

# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)
driver.close()
# Uncomment to delete index 
# drop_index_if_exists(driver, INDEX_NAME)
