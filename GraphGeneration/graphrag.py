from neo4j_graphrag.generation import GraphRAG
from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder, open_ai_text_ada_002_embedder
from neo4j import GraphDatabase
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from dotenv import load_dotenv
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from codebase import display_graph, context_retriever
from llms import llm_chat_gpt, llm_llama
import ast
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

index_name = "local_financial_gpt_index"

embedding_model = sentence_transformer_embedder

# Instantiate the LLM
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

def graphrag(question, index_name, embedder, top_k, llm):

    ## Connect to the Neo4j database
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Create the Retriever
    retriever = VectorRetriever(driver, index_name, embedder)
   
    # Instantiate the RAG pipeline
    rag = GraphRAG(retriever=retriever, llm=llm)

    print("QUESTION:", question)
    print()
    # Uncomment for Traditional RAG
    print("TRADITIONAL RAG RESULTS:")
    context=context_retriever(retriever.search(query_text=question, top_k=top_k))
    print(llm.invoke("Based on this context: "+str(context)+f" Answer the question: {question}").content)
    print()

    # Uncomment for GraphRAG
    print("GRAPH RAG RESULTS:")
    response = rag.search(query_text=question, retriever_config={"top_k": top_k}, return_context=True)
    print(response.answer)
    # print()
    # print("Retrieved Context:",response)
    # print()
    driver.close()

if __name__ == "__main__":
    # Query the graph
    question="A client wants to open a savings account, should I verify their identity?"
    graphrag(question, index_name, embedding_model, top_k=10, llm=llm_llama)
    ## Uncomment to diplay the graph of chunks
    display_graph()
