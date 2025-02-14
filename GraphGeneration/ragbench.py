import warnings
from urllib3.exceptions import InsecureRequestWarning
import ast
# Suppress the insecure request warning (for unverified HTTPS requests)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Attempt to suppress LangChain deprecation warnings (if the type is available)
try:
    from langchain import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    # If the warning type is not available, you can alternatively filter by message:
    warnings.filterwarnings("ignore", message=".*get_relevant_documents.*")

from datasets import load_dataset
from graphrag import graphrag
from local_graph_vector_db_generation import create_graph_embeddings
from global_graph_vector_db_generation import create_full_text_graph_embeddings
from embedders import sentence_transformer_embedder
from llms import llm_chat_gpt
from HybridQuery import hybrid_graphrag
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import drop_index_if_exists
from sentence_transformers import SentenceTransformer
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import HybridRetriever
from codebase import delete_all_indexes
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# Import the Embedder interface (adjust the import path as needed)
from neo4j_graphrag.embeddings.base import Embedder

class HuggingFaceEmbedderWrapper(Embedder):
    def __init__(self, model_name: str):
        # Instantiate the HuggingFaceEmbeddings model
        self.hf_embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_query(self, text: str) -> list[float]:
        # Delegate the embedding to the underlying HuggingFaceEmbeddings instance.
        # The HuggingFaceEmbeddings class should provide a method named embed_query.
        return self.hf_embedder.embed_query(text)

# Usage example:
embedder_wrapper = HuggingFaceEmbedderWrapper(model_name="mixedbread-ai/mxbai-embed-large-v1")

# wx discovery
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

from langchain_elasticsearch import ElasticsearchStore, ElasticsearchRetriever
from elasticsearch import Elasticsearch
es_client = Elasticsearch(
    os.getenv("ES_URL"),
    basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
    verify_certs=False,  # Note: Disabling certificate verification will cause InsecureRequestWarning unless suppressed.
    request_timeout=3600,
)

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Instantiate the LLM
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# for dataset in ['covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa']:

index_name = "rag_index"
graph_index_name = "rag_index2"
graph_full_text_index_name = "rag_index2_full_text"
## Set True if you want to delete the index you created
delete_index = True

cnt = 0
hybrid=False
k=2
# from itertools import islice


if __name__ == "__main__":
    for dataset in ['covidqa']:
        f = open("/Users/denizaskin/CodeBase/agenticchunking/GraphGeneration/results-covidqa.txt", "w")   # 'r' for reading and 'w' for writing
        rag_score_number=0
        graphrag_score_number=0
        hybrid_graphrag_score_number=0
        # Load a specific split of a subset dataset
        ragbench_dataset = load_dataset("rungalileo/ragbench", dataset, split="test")
        # This creates a list containing examples from the 28th element onward.
        # ragbench_dataset = list(islice(ragbench_dataset, 27, None))
        print("Size of dataset:",len(ragbench_dataset))
        f.write("Dataset:"+" "+dataset+"\n")
        f.write("Size of dataset:"+" "+str(len(ragbench_dataset))+"\n")
        try:
            # Delete index
            es_client.indices.delete(index=index_name)
            print(f"Successfully deleted index '{index_name}'")
        except:
            pass
        try:
            drop_index_if_exists(driver, graph_index_name)
            print(f"Successfully deleted index '{graph_index_name}'")
        except:
            pass
        for x in ragbench_dataset:
            # Query the graph
            question = x["question"]
            # Create graph embeddings here:
            create_graph_embeddings(x["documents"], graph_index_name, embedder_wrapper)
            # Create Elasticsearch vector store
            vector_store = ElasticsearchStore.from_texts(
                texts=x["documents"],
                embedding=embedding_function,
                es_connection=es_client,
                index_name=index_name
            )
            # Retrieve from vector store using the updated 'invoke' method
            retriever = vector_store.as_retriever(search_kwargs={"k": k})
            # Using the new 'invoke' method to avoid the deprecation warning
            context = "\n".join([doc.page_content for doc in retriever.invoke(question)])
            print("============================TRADITIONAL RAG============================")
            print("QUESTION:",question)
            print("CONTEXT:", context)
            f.write("============================TRADITIONAL RAG============================"+"\n")
            # Write File
            f.write("Question: "+question+"\n")
            f.write("Context: "+context+"\n")
            rag_answer = llm.invoke("Based on the provided context, answer the question. Only use the information in the context to answer, no other sources. Context: " + context + " Question: " + question).content
            # Delete index
            es_client.indices.delete(index=index_name)
            print(f"Successfully deleted index '{index_name}'")
            print(f"Answer: {rag_answer}")  
            f.write("Answer: "+rag_answer+"\n")
            print()

            print("============================GRAPH RAG============================")
            graphrag_answer, context = graphrag(question, graph_index_name, embedder_wrapper, top_k=k, llm=llm_chat_gpt)
            cntxt=""
            ## List of retrieved results
            for c in context:
                ## Dictionary
                cntxt+=ast.literal_eval(c.content)["text"]

            print("CONTEXT:", cntxt+"\n")
            print()
            f.write("============================GRAPH RAG============================"+"\n")
            f.write("Context: "+str(context)+"\n")
            f.write("Answer: "+graphrag_answer+"\n")
            if(hybrid):
                print("============================HYBRID GRAPH RAG============================")
                # Create full text graph embeddings here:
                create_full_text_graph_embeddings(x["documents"], graph_full_text_index_name, embedder_wrapper)
                hybrid_graphrag_answer = hybrid_graphrag(question, graph_index_name, graph_full_text_index_name, embedder_wrapper, top_k=k, llm=llm_chat_gpt)
                hybrid_graphrag_score=llm.invoke("Your task is to look at a ground truth and a candidate answer. Print 1 if the candidate answer means the same thing as the ground truth. Otherwise print 0. Do not print anything else. Ground Truth: "+x["response"]+" Candidate Answer: "+hybrid_graphrag_answer).content
                hybrid_graphrag_score_number+=int(hybrid_graphrag_score)
                f.write("Answer:"+" "+hybrid_graphrag_answer+"\n")  
                f.write("Hybrid GraphRAG Score:"+" "+str(hybrid_graphrag_score))  
                print("HybridGraphRAG Score:",hybrid_graphrag_score_number)
                print()

            rag_score=llm.invoke("Your task is to look at a ground truth and a candidate answer. Print 1 if the candidate answer means the same thing as the ground truth. Otherwise print 0. Do not print anything else. Ground Truth: "+x["response"]+" Candidate Answer: "+rag_answer).content
            graphrag_score=llm.invoke("Your task is to look at a ground truth and a candidate answer. Print 1 if the candidate answer means the same thing as the ground truth. Otherwise print 0. Do not print anything else. Ground Truth: "+x["response"]+" Candidate Answer: "+graphrag_answer).content

            rag_score_number+=int(rag_score)
            graphrag_score_number+=int(graphrag_score)

            if delete_index:
                drop_index_if_exists(driver, graph_index_name)
                # drop_index_if_exists(driver, graph_full_text_index_name)
                # delete_all_indexes(driver)
                driver.close()
            cnt+=1
            print(cnt)
            print("Dataset:",dataset)
            print("RAG Score:",rag_score_number)
            print("GraphRAG Score:",graphrag_score_number)
            f.write("RAG Score:"+" "+str(rag_score_number)+"\n")  
            f.write("GraphRAG Score:"+" "+str(graphrag_score_number))   
                
        f.close()    