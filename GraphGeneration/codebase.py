from llms import llm_code, llm_chat_gpt, llm_granite, llm_llama, llm_mistral
from langchain_experimental.graph_transformers import LLMGraphTransformer
from chunks import Chunks
import webbrowser
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

llm_transformer = LLMGraphTransformer(llm=llm_chat_gpt)


def display_graph():
    return webbrowser.open('http://localhost:7474/browser/')

def chunking(text, prompt, llm):
    structured_llm_chunking = llm.with_structured_output(Chunks)
    prompt_and_model_chunking = prompt | structured_llm_chunking
    response_chunking = prompt_and_model_chunking.invoke({"input": text})
    return response_chunking

def reviewing(text, prompt, knowledge_graph, llm):
    prompt_and_new_graph = prompt | llm
    new_graph = prompt_and_new_graph.invoke({"input": text, "knowledge_graph": knowledge_graph})
    return new_graph

def graph_generation(text, prompt, llm):
    prompt_and_graph = prompt | llm
    graph = prompt_and_graph.invoke({"input": text})
    return graph

def graph_comparison(text, prompt, llm, graph1, graph2):
    prompt_and_model = prompt | llm
    response_chunking = prompt_and_model.invoke({"text": text, "knowledge_graph_1": graph1, "knowledge_graph_2": graph2})
    return response_chunking

def reset_graph(knowledge_graph):
    knowledge_graph.query("MATCH (n) DETACH DELETE n")
    # Remove properties from nodes
    knowledge_graph.query("MATCH (n) SET n = {}")
    # Remove properties from relationships
    knowledge_graph.query("MATCH ()-[r]->() SET r = {}")

def graph_query(question, prompt, knowledge_graph, llm):
    prompt_and_graph = prompt | llm
    graph = prompt_and_graph.invoke({"question": question, "knowledge_graph":knowledge_graph})
    return graph

def retrieve_graph(knowledge_graph):
     # Retrieve all nodes and relationships
    query = """
    MATCH (n)-[r]->(m)
    RETURN n, r, m
    """
    return str(knowledge_graph.query(query))

def graph_query(question, prompt, knowledge_graph, llm):
    prompt_and_graph = prompt | llm
    answer = prompt_and_graph.invoke({"question": question, "knowledge_graph": knowledge_graph})
    return answer

def multiple_graph_query(question, prompt, llm, knowledge_graphs):
    prompt_and_graph = prompt | llm
    graph = prompt_and_graph.invoke({"question": question, "knowledge_graphs": knowledge_graphs})
    return graph

def graph_generation_with_review(llm, text, prompt_chunking, prompt_generation, knowledge_graph, print_chunks):
    print("GENERATING GRAPH")
    for x in chunking(text, prompt_chunking, llm).chunk:
        if(print_chunks):
            print(x.strip())
        if(llm!=llm_chat_gpt):
            response=graph_generation(x.strip(), prompt_generation, llm).content
            knowledge_graph.add_graph_documents([eval(response)])
        else:
            documents = [Document(page_content=x.strip())]
            response=llm_transformer.convert_to_graph_documents(documents)
            knowledge_graph.add_graph_documents(response)
        # reviewed_response=reviewing(x.strip(), prompt_graph_review_llama, response, llm).content
        # if(reviewed_response=="DONE"):
        #     reviewed_response=response
    knowledge_graph.refresh_schema()
    knowledge_graph_schema = knowledge_graph.get_structured_schema
    print("GRAPH Schema:", knowledge_graph_schema)
    return knowledge_graph, knowledge_graph_schema