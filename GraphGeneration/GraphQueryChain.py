from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

## Initialize Neo4J Graph
graph = Neo4jGraph()
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, allow_dangerous_requests=True
)
result = chain.invoke({"query": "Who is the customer?"})
print(result['result'])

