from neo4j_graphrag.generation import GraphRAG
from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder
from neo4j import GraphDatabase
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from dotenv import load_dotenv
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from codebase import display_graph
from llms import llm_chat_gpt
import ast
import os
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

## Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

index_name = "financial_index"

def graphrag(index_name, embedder):

    # Instantiate the LLM
    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    # Create the Retriever
    retriever = VectorRetriever(driver, index_name, embedder)

    # Instantiate the RAG pipeline
    rag = GraphRAG(retriever=retriever, llm=llm)

    # Query the graph
    query_text = "Should I verify the cusomter's identification??"
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5}, return_context=True)
    print(response.answer)
    print()
    print("Retrieved Context:",response)

if __name__ == "__main__":
    # Create indexes
    graphrag(index_name, sentence_transformer_embedder)
    driver.close()
    ## Uncomment to diplay the graph of chunks
    display_graph()
