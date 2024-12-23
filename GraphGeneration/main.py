from texts import text_eu, text_osfi
from llms import llm_code, llm_chat_gpt, llm_granite, llm_llama, llm_mistral
from codebase import graph_generation_with_review, graph_comparison, display_graph, reset_graph, graph_query, retrieve_graph, multiple_graph_query
from prompts import *
from langchain_community.graphs import Neo4jGraph

## Initialize Neo4J Graph
graph = Neo4jGraph()

## Select LLM
llm = llm_chat_gpt 

## Text to generate Knowledge Graphs from
f = open("datasets/transcripts/mortgage_loan_1_transcript.txt", "r")
mortgage_loan_transcript_1=''.join(f.read().splitlines())
f = open("datasets/transcripts/mortgage_loan_2_transcript.txt", "r")
mortgage_loan_transcript_2=''.join(f.read().splitlines())
f = open("datasets/transcripts/mortgage_loan_3_transcript.txt", "r")
mortgage_loan_transcript_3=''.join(f.read().splitlines())

transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]
list_of_graphs=[]
if __name__ == '__main__':
    # for x in range(0,len(transcripts)):
    #     # Create Knowledge Graphs of each text
    #     generated_graph, graph_schema=graph_generation_with_review(llm, transcripts[x], prompt_chunking_chat_gpt, prompt_graph_generation_chat_gpt, graph, False)
    #     ## Once the graph data is successfully stored in Neo4j, the graph will be visualized in Neo4j browser: http://localhost:7474/browser/
    #     generated_graph = retrieve_graph(generated_graph)
    #     list_of_graphs.append(f"Transcript #:{x+1}"+generated_graph)
    #     print()
    #     print(f"Transcript #:{x+1}"+generated_graph)
    #     print()
    #     ## Uncomment if you want the transcripts to be stored as separate Graphs
    #     # reset_graph(graph)
    # display_graph()
    # print()
    # question="List the common problems customers raised in these list of knowledge graphs in bulletpoints, and for each bulletpoint give a detail account of what specifically each problems entailed."
    question="The recurring issue seems to be a lack of understanding and surprise regarding payment amounts, particularly related to mortgage insurance and its impact on monthly payments. Can you explain in detail what this lack of understanding and surprise is due to? What is not being understood and is confusing the clients?"

    print("Question:",question)
    print()
    list_of_graphs=retrieve_graph(graph)
    ## Query the Knowledge Graphs
    print("REPORT OF TRANSCRIPTS:",multiple_graph_query(question=question,prompt=prompt_multiple_graph_query_chat_gpt, llm=llm, knowledge_graphs=list_of_graphs).content)

