from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder
from neo4j import GraphDatabase
from codebase import hybrid_retrieve_answer
from llms import llm_chat_gpt, llm_llama
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

llm = llm_llama
# Credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

local_index_name = "local_mortgage_index_sentence_transformer"
global_index_name = "global_mortgage_index_sentence_transformer"
embedding_model = open_ai_text_3_large_embedder

# Connect to Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME,NEO4J_PASSWORD))
if __name__ == "__main__":
    question="What does the customer want and how happy were they with the solution proposed by the representative?"
    answer, retrieved_context= hybrid_retrieve_answer(question, local_index_name, global_index_name, driver, embedding_model, llm)
    # Uncomment to delete indexes
    # drop_index_if_exists(driver, index_name)
    # drop_index_if_exists(driver, full_text_index_name)
    print("Question:",question)
    ()
    print("Answer:", answer)
    print()
    print("Retrieved Context:", retrieved_context)