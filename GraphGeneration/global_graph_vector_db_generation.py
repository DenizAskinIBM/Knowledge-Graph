import asyncio
from codebase import read, delete_all_indexes, print_index_names
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index, drop_index_if_exists
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

index_name = ""

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Uncomment to print and/or delete all indexes
# print_index_names(driver)
# delete_all_indexes(driver)

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

transcripts=read("datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md")
# Run the pipeline on a piece of text
text = (
    transcripts.strip()
)
asyncio.run(kg_builder.run_async(text=text))

create_fulltext_index(
    driver=driver,
    name=index_name,
    label="FinancialEntity",
    node_properties=[
        "TransactionType",
        "ThresholdAmount",
        "VerificationReason",
        "EntityType",
        "ExemptionCriteria",
        "CurrencyType",
        "VerificationMethod"
    ],
    fail_if_exists=False
)
driver.close()
# # drop_index_if_exists(driver, full_text_index_name)