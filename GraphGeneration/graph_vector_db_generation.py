import asyncio
import os
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists
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

index_name = "parsa-transcripts"
# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Instantiate the embedder and LLM
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
# Instantiate the LLM
llm = OpenAILLM(
        model_name="gpt-4",
        model_params={
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    )   

def index_creation(index_name):

    # Configure the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=100),
        on_error="IGNORE",
        from_pdf=False,
    )

    # Read and process transcripts
    transcript_files = [
        "datasets/transcripts/mortgage_loan_1_transcript.txt",
        "datasets/transcripts/mortgage_loan_2_transcript.txt",
        "datasets/transcripts/mortgage_loan_3_transcript.txt",
        "When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md",
    ]
    transcripts = [read(file) for file in transcript_files]

    # Async function for processing transcripts
    async def process_transcripts(kg_builder, transcripts):
        try:
            if isinstance(transcripts, list):
                for chunk in chunking(transcripts, prompt_chunking_chat_gpt, llm_chat_gpt).chunk:
                    for transcript in transcripts:
                        text = transcript.strip()
                        await kg_builder.run_async(text=text)
            else:
                for transcript in transcripts:
                    text = transcript.strip()
                    await kg_builder.run_async(text=text)
        except Exception as e:
            print(f"Error processing transcripts: {e}")

    # Run the transcript processing
    asyncio.run(process_transcripts(kg_builder, transcripts))

    # Create vector index

    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=3072,
        similarity_fn="cosine",
    )
    print(f"Vector index '{index_name}' created successfully.")
    # Close Neo4j driver
    driver.close()

if __name__ == "__main__":
    index_creation(index_name)
    ## Uncomment to delete index
    drop_index_if_exists(index_name)