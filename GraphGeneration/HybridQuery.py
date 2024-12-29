from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
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

index_name = "local_mortgage_index"
full_text_index_name = "global_mortgage_index"

# Connect to Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME,NEO4J_PASSWORD))
# Instantiate the embedder and LLM
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

if __name__ == "__main__":
    question="What does the customer want and how happy were they with the solution proposed by the representative?"
    answer, retrieved_context= hybrid_retrieve_answer(question, index_name, full_text_index_name, driver, embedder, llm)
    # Uncomment to delete indexes
    # drop_index_if_exists(driver, index_name)
    # drop_index_if_exists(driver, full_text_index_name)
    print("Question:",question)
    ()
    print("Answer:", answer)
    print()
    print("Retrieved Context:", retrieved_context)