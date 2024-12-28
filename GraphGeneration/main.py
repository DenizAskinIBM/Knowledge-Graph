from texts import text_eu, text_osfi
from llms import llm_code, llm_chat_gpt, llm_granite, llm_llama, llm_mistral
from codebase import graph_generation_with_review, retrieve_graph, multiple_graph_query, ask_question, main, display_graph, read
from prompts import *
from langchain_neo4j import Neo4jGraph
## Initialize Neo4J Graph
graph = Neo4jGraph()

# ## Select LLM
llm = llm_chat_gpt

## Text to generate Knowledge Graphs from
mortgage_loan_transcript_1 = read("datasets/transcripts/mortgage_loan_1_transcript.txt")
mortgage_loan_transcript_2 = read("datasets/transcripts/mortgage_loan_2_transcript.txt")
mortgage_loan_transcript_3 = read("datasets/transcripts/mortgage_loan_3_transcript.txt")
transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]

financial_transcript = read("datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md")
input = mortgage_loan_transcript_1

generate = True
if __name__ == '__main__':
    if(generate):
        ## Generate graph
        generated_graph = main(
            input,
            llm,
            prompt_chunking_chat_gpt,
            prompt_graph_generation_chat_gpt,
            prompt_correction=None,
            knowledge_graph=graph,
            print_chunks=False,
            use_langchain_transformer=True
        )
    else:
        generated_graph = retrieve_graph(graph)

    # Once the graph data is successfully stored in Neo4j, the graph will be visualized in Neo4j browser: http://localhost:7474/browser/
    display_graph()
    ## Query the Knowledge Graphs
    ask_question(prompt_multiple_graph_query_chat_gpt, llm, generated_graph)

