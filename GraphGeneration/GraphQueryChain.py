from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
openai_apikey=os.getenv("OPENAI_API_KEY")

graph = Neo4jGraph()
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, allow_dangerous_requests=True
)
result = chain.invoke({"query": "Who is Sara?"})
print(result['result'])

