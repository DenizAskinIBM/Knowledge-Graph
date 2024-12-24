from graph_vector_db_generation import retriever
from neo4j_graphrag.generation import GraphRAG
from dotenv import load_dotenv
from neo4j_graphrag.llm.openai_llm import OpenAILLM
# Load environment variables from .env file
load_dotenv()

# Instantiate the LLM
llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# Instantiate the RAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Query the graph
query_text = "How many conversations does Sarah and the representative have?"
response = rag.search(query_text=query_text, retriever_config={"top_k": 5}, return_context=True)
print(response.answer)
print()
print("Retrieved Chunks:",response.retriever_result.items)
# display_graph()


