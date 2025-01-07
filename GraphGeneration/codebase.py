from llms import llm_code, llm_chat_gpt, llm_granite, llm_llama, llm_mistral
from langchain_experimental.graph_transformers import LLMGraphTransformer
from chunks import Chunks
import webbrowser
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from neo4j_graphrag.retrievers import HybridRetriever
import ast

llm_transformer = LLMGraphTransformer(llm=llm_chat_gpt)

def read(filepath):
    f = open(filepath, "r")
    return ''.join(f.read().splitlines())

def print_index_names(driver):
    with driver.session() as session:
        result = session.run("SHOW INDEXES YIELD name")
        index_names = [record["name"] for record in result]
        print("Indexes in the database:")
        for name in index_names:
            print(name)

def delete_all_indexes(driver):
    with driver.session() as session:
        # Fetch all indexes
        result = session.run("SHOW INDEXES YIELD name")
        index_names = [record["name"] for record in result]
        
        # Drop each index
        for index_name in index_names:
            try:
                # Enclose the index name in backticks
                session.run(f"DROP INDEX `{index_name}`")
                print(f"Index '{index_name}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting index '{index_name}': {e}")

def display_graph():
    return webbrowser.open('http://localhost:7474/browser/')

def context_retriever(retriever_result):
    chunk_texts = []

    for item in retriever_result.items:
        # item.content looks like:
        # "{'id': ':4', 'index': 4, 'text': '...lots of text...', 'embedding': None}"
        
        # Convert the string to a real Python dict
        parsed_dict = ast.literal_eval(item.content)  
        # Now just grab the text key
        text_value = parsed_dict['text']
        
        chunk_texts.append(text_value)
        return chunk_texts

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
    

def hybrid_retrieve_answer(question, index_name, full_text_index_name, driver, embedder, top_k, llm):
    retriever = HybridRetriever(
            driver, index_name, full_text_index_name, embedder
        )
    retriever_result = retriever.search(query_text=question, top_k=top_k)
   
    # # 1. Grab the list of items from the retriever result
    # items = retriever_result.items

    # # 2. Sort them in descending order by score (assuming every item has 'metadata["score"]')
    # sorted_items = sorted(items, key=lambda x: x.metadata.get("score", 0.0), reverse=True)

    # # 3. (Optional) Replace the original list with the sorted version
    # retriever_result.items = sorted_items

    import ast

    chunk_texts = []

    for item in retriever_result.items:
        # item.content looks like:
        # "{'id': ':4', 'index': 4, 'text': '...lots of text...', 'embedding': None}"
        
        # Convert the string to a real Python dict
        parsed_dict = ast.literal_eval(item.content)  
        # Now just grab the text key
        text_value = parsed_dict['text']
        
        chunk_texts.append(text_value)

    answer = llm.invoke("Based on this context: "+str(chunk_texts)+f" Answer the question: {question}").content
    return answer, retriever_result

def main(input, llm, prompt_chunking_llama, prompt_graph_generation_llama, prompt_correction, knowledge_graph, print_chunks, use_langchain_transformer):
    if(isinstance(input,list)):
        for x in range(0,len(input)):
            # Create Knowledge Graphs of each text
            generated_graph, graph_schema=graph_generation_with_review(llm, input[x], 
                                                                    prompt_chunking_llama, prompt_graph_generation_llama, 
                                                                    prompt_correction, knowledge_graph, 
                                                                    print_chunks,
                                                                    use_langchain_transformer)
            print()
            print(f"Graph Schema {x+1}:",graph_schema)
            print()
    else:
        generated_graph, graph_schema=graph_generation_with_review(llm, input, 
                                                                    prompt_chunking_llama, prompt_graph_generation_llama, 
                                                                    prompt_correction, knowledge_graph, 
                                                                    print_chunks,
                                                                   use_langchain_transformer)
        print()
        print(f"Graph Schema:",graph_schema)
        print()
    return retrieve_graph(generated_graph)