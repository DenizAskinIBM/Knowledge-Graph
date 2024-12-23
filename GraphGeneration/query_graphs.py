from llms import llm_code, llm_chat_gpt, llm_granite, llm_llama, llm_mistral
from codebase import retrieve_graph, multiple_graph_query
from prompts import prompt_multiple_graph_query_chat_gpt
from langchain_community.graphs import Neo4jGraph
from codebase import display_graph
## Initialize Neo4J Graph
graph = Neo4jGraph()

## Select LLM
llm = llm_chat_gpt 

if __name__ == '__main__':
    display_graph()
    question="Which customers that inquired about amortization also inquired about credit card information? If so, what did they inquire about credit cards?"
    print("Question:",question)
    print()
    list_of_graphs=retrieve_graph(graph)
    ## Query the Knowledge Graphs
    print("REPORT OF TRANSCRIPTS:",multiple_graph_query(question=question,prompt=prompt_multiple_graph_query_chat_gpt, llm=llm, knowledge_graphs=list_of_graphs).content)
