from llms import llm_code, llm_chat_gpt, llm_granite, llm_llama, llm_mistral
from langchain_experimental.graph_transformers import LLMGraphTransformer
from chunks import Chunks
import webbrowser
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

llm_transformer = LLMGraphTransformer(llm=llm_chat_gpt)

def read(filepath):
    f = open(filepath, "r")
    return ''.join(f.read().splitlines())

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

def graph_correction(text, prompt, llm):
    prompt_and_graph = prompt | llm
    graph = prompt_and_graph.invoke({"knowledge_graph": text})
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

def single_graph_query(question, prompt, knowledge_graph, llm):
    prompt_and_graph = prompt | llm
    answer = prompt_and_graph.invoke({"question": question, "knowledge_graph": knowledge_graph})
    return answer

def multiple_graph_query(question, prompt, llm, knowledge_graphs):
    prompt_and_graph = prompt | llm
    graph = prompt_and_graph.invoke({"question": question, "knowledge_graphs": knowledge_graphs})
    return graph

def ask_question(prompt, llm, knowledge_graphs):
    question = input("Your question (type 'exit' to quit): ").strip()
    if question.lower() == 'exit':
        return
    else:
        answer = multiple_graph_query(question, prompt, llm, knowledge_graphs).content
        print(answer)
        print()
    ask_question(prompt, llm, knowledge_graphs)  # Recursively call the function again

def graph_generation_with_review(llm, text, prompt_chunking, prompt_generation, prompt_correction, knowledge_graph, print_chunks, use_langchain_transformer):
    print("GENERATING GRAPH")
    for x in chunking(text, prompt_chunking, llm).chunk:
        if(print_chunks):
            print(x.strip())
        if(use_langchain_transformer):
            documents = [Document(page_content=x.strip())]
            response=llm_transformer.convert_to_graph_documents(documents)
            knowledge_graph.add_graph_documents(response)
        else:
            response=graph_generation(x.strip(), prompt_generation, llm).content
            response=graph_correction(response, prompt_correction, llm).content
            knowledge_graph.add_graph_documents([eval(response)])
    knowledge_graph.refresh_schema()
    knowledge_graph_schema = knowledge_graph.get_structured_schema
    return knowledge_graph, knowledge_graph_schema

def main(transcripts, llm, prompt_chunking_llama, prompt_graph_generation_llama, prompt_correction, knowledge_graph, print_chunks, use_langchain_transformer):
    if(isinstance(transcripts,list)):
        for x in range(0,len(transcripts)):
            # Create Knowledge Graphs of each text
            generated_graph, graph_schema=graph_generation_with_review(llm, transcripts[x], 
                                                                    prompt_chunking_llama, prompt_graph_generation_llama, 
                                                                    prompt_correction, knowledge_graph, 
                                                                    print_chunks,
                                                                    use_langchain_transformer)
            print()
            print(f"Graph Schema {x+1}:",graph_schema)
            print()
    else:
        generated_graph, graph_schema=graph_generation_with_review(llm, transcripts[x], 
                                                                    prompt_chunking_llama, prompt_graph_generation_llama, 
                                                                    prompt_correction, knowledge_graph, 
                                                                    print_chunks,
                                                                   use_langchain_transformer)
        print()
        print(f"Graph Schema:",graph_schema)
        print()
    return retrieve_graph(generated_graph)