from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j import GraphDatabase
import ast 
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()

# Credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

index_name = "financial-transcripts"
full_text_index_name = "financial-transcript-fulltext-index-name"

# Connect to Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME,NEO4J_PASSWORD))
# Instantiate the embedder and LLM
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
retriever = HybridRetriever(
          driver, index_name, full_text_index_name, embedder
      )
query_text = "A client wants to open a savings account. Should I verify their identity?"
retriever_result = retriever.search(query_text=query_text, top_k=20)

# Retrieve the text of the context
context = []
for item in retriever_result.items:
    # Parse the `content` string into a dictionary
    content_dict = ast.literal_eval(item.content)  # Convert string to dictionary
    if 'text' in content_dict:
        context.append(content_dict['text'])  # Extract the 'text' field

from llms import llm_chat_gpt
print(llm_chat_gpt.invoke("Based on this context: "+str(context)+" Answer the question: 'Should I verify the identity of the customer?'").content)