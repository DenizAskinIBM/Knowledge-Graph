from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder, open_ai_text_ada_002_embedder
from neo4j import GraphDatabase
from codebase import hybrid_retrieve_answer, read, delete_all_indexes, print_index_names, context_retriever
from llms import llm_chat_gpt
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from llms import llm_chat_gpt
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.generation import GraphRAG
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

 # Instantiate the LLM
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})
# Credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Enter indexes
local_index_name = "local_financial_gpt_index"
global_index_name = "global_financial_gpt_index"

# Select embedding model
embedding_model = sentence_transformer_embedder


def hybrid_graphrag(question, graph_index_name, graph_full_text_index_name, embedder, top_k, llm):

    ## Connect to the Neo4j database
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Create the Retriever
    retriever = HybridRetriever(driver, graph_index_name, graph_full_text_index_name, embedder)
   
    # Instantiate the RAG pipeline
    rag = GraphRAG(retriever=retriever, llm=llm)

    print("QUESTION:", question)
    print()
    # Uncomment for GraphRAG
    print("HYBRID GRAPH RAG RESULTS:")
    response = rag.search(query_text=question, retriever_config={"top_k": top_k}, return_context=True)
    print(response.answer)
    print()
    print("Retrieved Context:",response.retriever_result.items)
    print()
    driver.close()
    return response.answer

if __name__ == "__main__":
    # Connect to Neo4j database
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME,NEO4J_PASSWORD))
    # Uncomment to print and/or delete all indexes
    # print_index_names(driver)
    # delete_all_indexes(driver)
    question="A client wants to open a savings account, should I verify their identity?"
    top_k=10
    retriever = HybridRetriever(
            driver, local_index_name, global_index_name, embedding_model
        )
    # Initialize the RAG pipeline
    rag = GraphRAG(retriever=retriever, llm=llm)
    print("Top_k", top_k)
    print("Question:",question)
    hybrid_retrieve_answer = rag.search(query_text=question, retriever_config={"top_k": top_k}, return_context=True)
    print("Answer:", hybrid_retrieve_answer.answer)
    print()
    print("Answer 2:", llm_chat_gpt.invoke("Based on the provided context, answer the question. Context: "+str(retriever.search(query_text=question, top_k=10))+" Question: "+question).content)
    print()
    # for x in hybrid_retrieve_answer.retriever_result:
    #     print()
    #     print(x)
    #     print()
    driver.close()