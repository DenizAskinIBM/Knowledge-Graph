from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j import GraphDatabase
from langchain.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
openai_apikey=os.getenv("OPENAI_API_KEY")

INDEX_NAME = "parsa-transcripts"

## Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Create an Embedder object
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

##Initialize the Retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# Instantiate the LLM
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# Instantiate the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Query the graph
query_text = "How many conversations does Sarah and the representative have?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 10}, return_context=True)
print(response.answer)
print()
print("Retrieved Chunks:",response.retriever_result.items)
# display_graph()


