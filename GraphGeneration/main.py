from texts import text_eu, text_osfi
from llms import llm_code, llm_chat_gpt, llm_granite, llm_llama, llm_mistral
from codebase import graph_generation_with_review, retrieve_graph, multiple_graph_query, ask_question, main, display_graph
from prompts import *
from langchain_community.graphs import Neo4jGraph

## Initialize Neo4J Graph
graph = Neo4jGraph()

## Select LLM
llm = llm_llama

## Text to generate Knowledge Graphs from
f = open("datasets/transcripts/mortgage_loan_1_transcript.txt", "r")
mortgage_loan_transcript_1=''.join(f.read().splitlines())
f = open("datasets/transcripts/mortgage_loan_2_transcript.txt", "r")
mortgage_loan_transcript_2=''.join(f.read().splitlines())
f = open("datasets/transcripts/mortgage_loan_3_transcript.txt", "r")
mortgage_loan_transcript_3=''.join(f.read().splitlines())
transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]

generate = True
if(generate):
    if __name__ == '__main__':
        ## Generate graph
        graph = main(
            transcripts,
            llm,
            prompt_chunking_llama,
            prompt_graph_generation_llama,
            prompt_correction=None,
            knowledge_graph=graph,
            print_chunks=False,
            use_langchain_transformer=True
        )
    else:
        retrieve_graph(graph)

    # Once the graph data is successfully stored in Neo4j, the graph will be visualized in Neo4j browser: http://localhost:7474/browser/
    display_graph()
    ## Query the Knowledge Graphs
    ask_question(prompt_multiple_graph_query_llama, llm, graph)

