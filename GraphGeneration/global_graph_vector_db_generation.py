import asyncio
from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder, open_ai_text_ada_002_embedder
from codebase import read, delete_all_indexes, print_index_names, chunking
from prompts import prompt_chunking_chat_gpt
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index, drop_index_if_exists
from Custom_LLM import CustomLLM
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from prompts import prompt_chunking_chat_gpt
from llms import llm_chat_gpt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

index_name = "global_financial_gpt_index"

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
llm_openi_gpt = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)
# Instantiate your custom LLM
custom_llm = CustomLLM(model_name="meta-llama/llama-3-405b-instruct", model_params={"temperature": 0})

def create_full_text_graph_embeddings(transcripts, graph_full_text_index_name, embedder):
    ## Use LLM Based Chunking
    advanced_chunking=False

    # Define your splitter
    if(advanced_chunking):
        text_splitter = None
    else:
        text_splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=50)

    # Instantiate the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm_openi_gpt,
        driver=driver,
        embedder=embedder,
        on_error="IGNORE",
        from_pdf=False,
    )
    
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

    create_fulltext_index(
        driver,
        graph_full_text_index_name,
        label="Document",
        node_properties=["vectorProperty"],
        fail_if_exists=False,
    )
    # For Financial Document
    # create_fulltext_index(
    #     driver=driver,
    #     name=index_name,
    #     label="FinancialEntity",
    #     node_properties=[
    #         "TransactionType",
    #         "ThresholdAmount",
    #         "VerificationReason",
    #         "EntityType",
    #         "ExemptionCriteria",
    #         "CurrencyType",
    #         "VerificationMethod",
    #         "AccountOpenings"
    #     ],
    #     fail_if_exists=False,
    # )

    # ## For Mortgage Document
    # create_fulltext_index(
    #     driver,
    #     index_name,
    #     label="MortgageTranscript",
    #     node_properties=["conversationText", "customerName", "representativeName"],
    #     fail_if_exists=False,
    # )

    # ## Set True if you want to delete the index you created
    delete_index=False
    if(delete_index):
        drop_index_if_exists(driver, index_name)
    driver.close()
