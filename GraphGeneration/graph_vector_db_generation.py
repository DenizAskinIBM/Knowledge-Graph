import asyncio
import os
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists, create_fulltext_index
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from llms import llm_chat_gpt
from prompts import prompt_chunking_chat_gpt
from codebase import display_graph, read, chunking
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

index_name = "financial-transcripts"
full_text_index_name = "financial-transcript-fulltext-index-name"

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Instantiate the LLM
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

def index_creation(index_name):
    # Instantiate the embedder and LLM
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Configure the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        # Uncomment to use a chunker
        # text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
        on_error="IGNORE",
        from_pdf=False,
    )

    # Read and process transcripts
    transcript_files = [
        "datasets/transcripts/mortgage_loan_1_transcript.txt",
        "datasets/transcripts/mortgage_loan_2_transcript.txt",
        "datasets/transcripts/mortgage_loan_3_transcript.txt",
        "datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md",
    ]
    transcripts= read("datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md")

    if isinstance(transcripts, list):
        for x in transcripts:
            for y in chunking(x.strip(), prompt_chunking_chat_gpt, llm_chat_gpt).chunk:
                text = (
                        y.strip()
                    )
                kg_builder.run_async(text=text)
    else:
         for x in chunking(transcripts, prompt_chunking_chat_gpt, llm_chat_gpt).chunk:
                text = (
                        x.strip()
                    )
                kg_builder.run_async(text=x.strip())

    # Dim: 1536 for 
    # Create vector index "text-embedding-ada-002", 3072 for openai "text-embedding-3-large"
    create_vector_index(
        driver,
        name=index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=1536,
        similarity_fn="cosine",
    )

    def store_text_segments(driver, segments):
        with driver.session() as session:
            for segment in segments:
                session.run(
                    "CREATE (n:TextChunk {content: $content})",
                    content=segment
                )

    def split_text_by_heading(text):
        import re
        # Assuming headings are formatted as markdown headings (e.g., ## or ###)
        sections = re.split(r"(?=## )", text)
        return [section.strip() for section in sections if section.strip()]

    create_fulltext_index(
    driver=driver,
    name=full_text_index_name,
    label="Document",
    node_properties=["vectorProperty"]  # Property storing raw text
    )
    print(f"Vector index '{index_name}' created successfully.")

    # Close Neo4j driver
    driver.close()

if __name__ == "__main__":
    index_creation(index_name)
    # Uncomment to delete indexes
    # drop_index_if_exists(driver, index_name)
    # drop_index_if_exists(driver, full_text_index_name)