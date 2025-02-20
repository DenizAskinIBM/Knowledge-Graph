
from langchain_core.exceptions import OutputParserException
import os
import re
import json
import PyPDF2
import concurrent.futures
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_ibm import ChatWatsonx
from langgraph.graph import END, StateGraph
from IPython.display import Image, display

################################################
# 1. Planner Prompt Text
################################################
PLANNER_PROMPT_TEXT = """
You are the Compliance Planner Agent, and your task is to create a sequence of subtasks needed to address the user’s question thoroughly.

Context Provided:
1) PDF Content: {{pdf_text}}
2) Existing user question/input: "{{user_question}}"

Your Objective:
1) Figure out each step or piece of information required to provide a complete, accurate answer. 
2) If the needed information is already available in the context (the PDF text or the user’s existing question/answers), you can answer with an "LLM" subtask. 
3) If any piece of information is missing—i.e., it is not in the PDF text or the user’s previous answers—then you must not guess or assume. Instead, you must create a "USER" subtask to explicitly ask the user for that missing detail. 

Instructions:
- Return only valid JSON, structured as follows:
  {
    "subtasks": [
      {
        "task": "Subtask text here",
        "type": "LLM or USER"
      },
      ...
    ]
  }
- "type": "LLM" if the subtask is fully solvable with the provided PDF text + any known user inputs.
- "type": "USER" if you do NOT have all necessary data to complete that subtask using only the PDF text and prior user inputs. In that case, prompt the user for the missing details.

Remember:
- Do NOT fabricate user data or guess any missing numerical values, dates, or other specifics. 
- Do NOT provide your final reasoning or steps outside of JSON. 
- Only produce valid JSON in the exact format shown above, with no extra commentary.

Now, generate the required JSON plan that covers all steps needed to answer the user’s question.
"""

# print("NEW PLANNER PROMPT:\n")
# print(PLANNER_PROMPT_TEXT)

################################################
# 2. Setup Watsonx LLM
################################################
load_dotenv()

url = os.getenv("WATSONX_URL")
apikey = os.getenv("API_KEY")
project_id = os.getenv("PROJECT_ID")
openai_apikey = os.getenv("OPENAI_API_KEY")

model_id_llama = "meta-llama/llama-3-405b-instruct"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42
}

llm_llama = ChatWatsonx(
    model_id=model_id_llama,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

################################################
# 3. Helper Functions
################################################
def read_pdf_text(file_path: str) -> str:
    text_content = []
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text_content.append(page.extract_text())
    return "\n".join(text_content)

def simulate_streaming(text: str) -> str:
    """No streaming; returns text immediately."""
    return text.replace("\n", " ")

def stream_llm_call(prompt: str) -> str:
    """Call the Watsonx LLM, return the response text, no streaming."""
    response_msg = llm_llama.invoke([HumanMessage(content=prompt)])
    if hasattr(response_msg, "content"):
        text_content = response_msg.content
    else:
        text_content = response_msg[0].content
    return simulate_streaming(text_content)

################################################
# 4. Agent Workflow State and Models
################################################
class AgentWorkflowState(TypedDict):
    pdf_path: str
    user_question: str
    pdf_text: str
    planner_output: str
    execution_result: dict
    final_answer: str

class SubTask(BaseModel):
    task: str = Field(..., description="Subtask description")
    type: str = Field(..., description="'LLM' or 'USER'")

class PlannerOutput(BaseModel):
    subtasks: List[SubTask] = Field(..., description="List of required subtasks")

parser = PydanticOutputParser(pydantic_object=PlannerOutput)

################################################
# 5. Compliance Planner Node
################################################
planner_prompt_str = f"""
{PLANNER_PROMPT_TEXT}
"""

planner_prompt = PromptTemplate(
    template=planner_prompt_str,
    input_variables=["pdf_text", "user_question"],
    partial_variables={
        "pdf_text": "{{pdf_text}}",
        "user_question": "{{user_question}}"
    }
)

def compliance_planner_node(state: AgentWorkflowState) -> AgentWorkflowState:
    pdf_text = state["pdf_text"]
    user_question = state["user_question"]

    # Manually replace placeholders in the final planner prompt
    final_planner_prompt = planner_prompt_str.replace("{{pdf_text}}", pdf_text)\
                                             .replace("{{user_question}}", user_question)
    result = llm_llama.invoke([HumanMessage(content=final_planner_prompt)])
    raw_output = result.content if hasattr(result, "content") else result[0].content

    try:
        structured_output = parser.parse(raw_output)
        planner_response = structured_output.model_dump_json()
    except OutputParserException as e:
        planner_response = f'{{"subtasks":[],"error":"Invalid JSON: {e}"}}'

    state["planner_output"] = planner_response
    return state

################################################
# 6. Executer Node
################################################
def executer_node(state: AgentWorkflowState) -> AgentWorkflowState:
    """
    Executes each subtask in the plan. 
    Some subtasks may be "USER", requiring user input.
    Some subtasks may be "LLM", requiring an LLM call.

    If any LLM subtask includes 'INSUFFICIENT:' in the answer,
    the LLM must specify what question(s) it needs. We parse that text,
    ask the user, and re-run the LLM subtask with the user's answers.

    We store and print each subtask's outcome in the form:
      [Subtask # - LLM/USER]
      Task:
      Answer:

    with a blank line between blocks, 
    and ensure "Answer:" is printed only once for USER subtasks.
    """
    pdf_text = state["pdf_text"]
    plan_json_str = state["planner_output"]

    try:
        plan = json.loads(plan_json_str)
        subtasks = plan.get("subtasks", [])
    except Exception as e:
        state["execution_result"] = {
            "error": f"Invalid JSON from planner: {e}",
            "subtask_answers": []
        }
        return state

    subtask_answers = [None] * len(subtasks)

    def run_llm_subtask_with_retry(subtask_index: int, subtask_text: str) -> str:
        # Build context from all subtask answers so far
        context_str_list = []
        for i, ans in enumerate(subtask_answers):
            if ans is not None and ans["answer"] is not None:
                context_str_list.append(
                    f"[Subtask #{i+1} - {ans['type']}]\n"
                    f"Task: {ans['task']}\n"
                    f"Answer: {ans['answer']}\n"
                )
        context_str = "\n".join(context_str_list)

        prompt_llm = f"""
You are the Executer Agent. You have the following PDF content:
{pdf_text}

Context from completed subtasks so far:
{context_str}

Now execute this subtask:
Task: {subtask_text}

If additional user info is needed, respond with:
"INSUFFICIENT: <one or more questions the user must answer>"
Otherwise provide your best possible answer.
"""
        llm_answer = stream_llm_call(prompt_llm).strip()

        # Check if the LLM says "INSUFFICIENT:"
        pattern = re.compile(r"INSUFFICIENT\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
        match = pattern.search(llm_answer)
        if match:
            # This means the LLM asked for more info
            missing_info_str = match.group(1).strip()
            if not missing_info_str:
                missing_info_str = "What additional info is needed?"

            # Possibly multiple questions line by line
            lines = [line.strip() for line in missing_info_str.split("\n") if line.strip()]

            # Ask user for each question
            user_additions = []
            if lines:
                for missing_q in lines:
                    print(f"\nLLM indicates insufficient info for subtask #{subtask_index+1}. Question: {missing_q}")
                    user_input_for_q = input("Your answer: ")
                    user_additions.append(f"{missing_q}: {user_input_for_q}")
            else:
                # If no lines were found, just ask once
                print(f"\nLLM indicates insufficient info for subtask #{subtask_index+1}.")
                user_input_for_q = input("Please provide additional info: ")
                user_additions.append(user_input_for_q)

            # Append user answers to context
            user_info_str = "\n".join(user_additions)
            context_str += f"\n[User Additional Info]\n{user_info_str}\n"

            # Re-run the subtask with updated context
            retry_prompt = f"""
You are the Executer Agent. You have the following PDF content:
{pdf_text}

Updated context (including user-provided info):
{context_str}

Re-execute subtask:
Task: {subtask_text}

Do NOT respond with 'INSUFFICIENT' if the info suffices.
"""
            llm_answer = stream_llm_call(retry_prompt).strip()

        return llm_answer

    for idx, subtask_obj in enumerate(subtasks):
        subtask_type = subtask_obj.get("type", "LLM").upper()
        subtask_text = subtask_obj.get("task", "")

        print(f"\n[Subtask #{idx+1} - {subtask_type}]")
        print(f"Task: {subtask_text}")

        if subtask_type == "USER":
            user_ans = input("Answer: ")
            subtask_answers[idx] = {
                "type": subtask_type,
                "task": subtask_text,
                "answer": user_ans
            }
        else:
            llm_response = run_llm_subtask_with_retry(idx, subtask_text)
            subtask_answers[idx] = {
                "type": subtask_type,
                "task": subtask_text,
                "answer": llm_response
            }
            print(f"Answer: {llm_response}")

    state["execution_result"] = {"subtask_answers": subtask_answers}
    return state

################################################
# 7. Question Answerer Node
################################################
def question_answerer_node(state: AgentWorkflowState) -> AgentWorkflowState:
    pdf_text = state["pdf_text"]
    user_question = state["user_question"]
    planner_output = state["planner_output"]
    execution_result = state["execution_result"]
    subtask_answers = execution_result.get("subtask_answers", [])

    context_str = ""
    for i, ans in enumerate(subtask_answers):
        context_str += (
            f"[Subtask #{i+1} - {ans['type']}]\n"
            f"Task: {ans['task']}\n"
            f"Answer: {ans['answer']}\n\n"
        )

    final_prompt = f"""
You are the Question Answerer Agent.

You have:
1) PDF Content:
{pdf_text}

2) The user's question:
{user_question}

3) The plan (raw JSON):
{planner_output}

4) The completed subtasks and their answers:
{context_str}

Provide a final consolidated answer that addresses the user's question in full detail, 
using subtask answers, user-provided data, and the PDF content. 
Do NOT ask for additional user input.
"""
    final_answer = stream_llm_call(final_prompt).strip()
    state["final_answer"] = final_answer
    return state

################################################
# 8. Build the Workflow Graph
################################################
def build_workflow():
    wf = StateGraph(AgentWorkflowState)
    wf.add_node("compliance_planner", compliance_planner_node)
    wf.add_node("executer_agent", executer_node)
    wf.add_node("question_answerer", question_answerer_node)
    wf.set_entry_point("compliance_planner")
    wf.add_edge("compliance_planner", "executer_agent")
    wf.add_edge("executer_agent", "question_answerer")
    wf.add_edge("question_answerer", END)
    return wf

################################################
# 9. Orchestrator
################################################
def run_workflow(pdf_path: str, user_question: str) -> str:
    initial_state: AgentWorkflowState = {
        "pdf_path": pdf_path,
        "user_question": user_question,
        "pdf_text": read_pdf_text(pdf_path),
        "planner_output": "",
        "execution_result": {},
        "final_answer": ""
    }
    wf = build_workflow().compile()
    final_state = wf.invoke(initial_state)
    return final_state["final_answer"]

################################################
# 10. Example Usage & Graph Image
################################################
if __name__ == "__main__":
    # Display & Save the Workflow Graph
    app = build_workflow()
    compiled_app = app.compile()
    png_graph = compiled_app.get_graph(xray=True).draw_mermaid_png()

    with open("workflow_graph.png", "wb") as f:
        f.write(png_graph)
    pdf_file_path = "GraphGeneration/ontario.ca.pdf"
    user_question = input("Enter your question: ")
    answer = run_workflow(pdf_file_path, user_question)
    print("\n\nFinal Answer:\n", answer)

    