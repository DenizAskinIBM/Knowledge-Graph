from langgraph.graph import END, StateGraph
from mypy_extensions import TypedDict
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
import os
load_dotenv()

class GraphState(TypedDict):
    question: str
    retrieved_context: str
    router_decision: str
    answer_to_question: str

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def retrieve(state: GraphState):
    question = state["question"]
    tavily_client = TavilyClient()
    result = tavily_client.search(question, max_results=3)
    retrieved_context = "\n".join([r["content"] for r in result["results"]])
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[{"role": "user", "content": "You are a retrieval validator. Your role is to look through chunks of retrieved text and then a question. If the retrieved chunks contain the answer to the question, print 'VALID'. Or else, print 'INVALID'. Do not print anything else. Context: "+retrieved_context+"\n The Question: "+question}],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=True,
    )

    full_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            text_part = chunk.choices[0].delta.content
            print(text_part, end="")
            full_text += text_part
    router_decision = full_text.split("</think>")[1].strip()
    valid_decision = "VALID" if "VALID" in router_decision else "INVALID"
    print(f"\nRetrieval Validation: {valid_decision}")
    return {"router_decision": valid_decision, "retrieved_context": retrieved_context}

def answer(state: GraphState):
    question = state["question"]
    context = state["retrieved_context"]

    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[{"role": "user", "content": f"Answer concisely using only the provided context. Context: {context}\nQuestion: {question}\nAnswer:"}],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=True,
    )
    full_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            text_part = chunk.choices[0].delta.content
            print(text_part, end="")
            full_text += text_part
    answer = full_text.split("</think>")[1].strip()
    print(f"\nFinal Answer: {answer}")
    return {"answer_to_question": answer}

def decide_route(state: GraphState):
    return state["router_decision"]

def create_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("answer", answer)
    
    workflow.set_entry_point("retrieve")
    
    workflow.add_conditional_edges(
        "retrieve",
        decide_route,
        {
            "VALID": "answer",
            "INVALID": "retrieve"
        }
    )
    
    workflow.add_edge("answer", END)
    return workflow.compile()

if __name__ == "__main__":
    workflow = create_workflow()
    result = workflow.invoke({"question": "Who is George Washington?"})
    print("\nFinal Result:", result["answer_to_question"])