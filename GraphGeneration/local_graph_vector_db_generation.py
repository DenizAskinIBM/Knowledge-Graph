import asyncio
from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder, open_ai_text_ada_002_embedder
from codebase import read, delete_all_indexes, print_index_names, chunking
from prompts import prompt_chunking_chat_gpt
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index, drop_index_if_exists
from Custom_LLM import CustomLLM
from llms import llm_chat_gpt
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

index_name = "local_financial_gpt_index"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Select Embedding Model
embedding_model = sentence_transformer_embedder
transformer_model = embedding_model.model

## Set True if you want to print all indexes
print_all_indexes=True
if(print_all_indexes):
    print_index_names(driver)
## Set True if you want to delete all indexes
del_all_indexes=False
if(del_all_indexes):
    delete_all_indexes(driver)

# Instantiate the LLM
llm_open_ai_gpt = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

# Instantiate your custom LLM
custom_llm = CustomLLM(model_name="meta-llama/llama-3-405b-instruct", model_params={"temperature": 0})

def create_graph_embeddings(transcripts, graph_index_name, embedder):
    ## Use LLM Based Chunking
    advanced_chunking=False

    # Define your splitter
    if(advanced_chunking):
        text_splitter = None
    else:
        text_splitter = FixedSizeSplitter(chunk_size=10000, chunk_overlap=0)

    # Instantiate the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm_open_ai_gpt,
        driver=driver,
        embedder=embedder,
        on_error="IGNORE",
        from_pdf=False,
        text_splitter=text_splitter
    )

    # ## Text to generate Knowledge Graphs from
    # mortgage_loan_transcript_1 = read("datasets/transcripts/mortgage_loan_1_transcript.txt")
    # mortgage_loan_transcript_2 = read("datasets/transcripts/mortgage_loan_2_transcript.txt")
    # mortgage_loan_transcript_3 = read("datasets/transcripts/mortgage_loan_3_transcript.txt")
    # transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]
    # transcripts=read("datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md")

    if(isinstance(transcripts, list)):
        if(advanced_chunking):
            for x in transcripts:
                for y in chunking(x.strip(), prompt_chunking_chat_gpt, llm_chat_gpt).chunk: 
                    # Run the pipeline on a piece of text
                    text = (
                        y.strip()
                    )
                    asyncio.run(kg_builder.run_async(text=text))
        else:
            for x in transcripts:
                # Run the pipeline on a piece of text
                text = (
                    x.strip()
                )
                asyncio.run(kg_builder.run_async(text=text)) 
    else:
        if(advanced_chunking):
            for x in chunking(transcripts, prompt_chunking_chat_gpt, llm_chat_gpt).chunk: 
            # Run the pipeline on a piece of text
                text = (
                    x.strip()
                )
                asyncio.run(kg_builder.run_async(text=text)) 
        else:
            # Run the pipeline on a piece of text
            text = (
                transcripts.strip()
            )
            asyncio.run(kg_builder.run_async(text=text)) 
            
    if(isinstance(embedding_model, SentenceTransformerEmbeddings)):
        dimensions=transformer_model.get_sentence_embedding_dimension()
    else:
        sample_embedding = open_ai_text_3_large_embedder.embed_query("sample text")
        dimensions = len(sample_embedding)
    # Create the index
    create_vector_index(
        driver,
        graph_index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=1024,
        similarity_fn="cosine",
    )
    driver.close()

