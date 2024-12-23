from graph_vector_db_generation import retriever, driver
from neo4j_graphrag.generation import GraphRAG
from dotenv import load_dotenv
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j import GraphDatabase
import os
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
openai_apikey=os.getenv("OPENAI_API_KEY")

# Instantiate the LLM
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# Instantiate the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Query the graph
query_text = "Who is the customer?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5}, return_context=True)
print(response.answer)
print()
print("Retrieved Chunks:",response.retriever_result.items)
driver.close()
# display_graph()


