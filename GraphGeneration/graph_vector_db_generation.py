import asyncio
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from codebase import read, chunking
from prompts import prompt_chunking_llama, prompt_graph_generation_chat_gpt
from llms import llm_chat_gpt, llm_llama
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists
from neo4j_graphrag.llm.openai_llm import OpenAILLM

# Load environment variables from .env file
load_dotenv()
print("hey")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

## Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Create an Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

index_name = "all-transcripts-llama"

def vector_indexes(index_name, embedder):

    # Instantiate the LLM
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 2000,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        },
    )

    ## Instantiate the TextSplitter
    text_splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=100)

    # Instantiate the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        ## Uncomment to use with text splitter
        # text_splitter=text_splitter,
        on_error="IGNORE",
        from_pdf=False)

    # Generate an embedding for some text
    mortgage_loan_transcript_1 = read("datasets/transcripts/mortgage_loan_1_transcript.txt")
    mortgage_loan_transcript_2 = read("datasets/transcripts/mortgage_loan_2_transcript.txt")
    mortgage_loan_transcript_3 = read("datasets/transcripts/mortgage_loan_3_transcript.txt")
    transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]

    if(isinstance(transcripts, list)):
        for x in transcripts:
            for y in chunking(x.strip(), prompt_chunking_llama, llm_llama).chunk:
                # Run the pipeline on a chunks of text
                text = (
                    y.strip()
                )
                asyncio.run(kg_builder.run_async(text=text))
    else:
        text = (
                transcripts.strip()
            )
        asyncio.run(kg_builder.run_async(text=text))

    # Create the index
    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=3072,
        similarity_fn="cosine",
    )

if __name__ == "__main__":
    # Create indexes
    vector_indexes(index_name, embedder)
    driver.close()
    # Uncomment to delete index 
    # drop_index_if_exists(driver, INDEX_NAME)
