## Copyright Deniz Askin. Edited with GPT-o1-pro
import os
import re
import json
import httpcore
import sys
import time
from dotenv import load_dotenv
from typing_extensions import TypedDict

# If you installed via "pip install langchain-core", adjust imports accordingly:
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# Regex patterns (only needed now for the final answer and JSON extraction)
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
STRIP_TAGS_RE = re.compile(r"<.*?>")

def strip_tags(text: str) -> str:
    """Remove HTML-like tags from text."""
    return STRIP_TAGS_RE.sub("", text).strip()

class VerificationState(TypedDict):
    question: str
    current_answer: str
    reasoning_history: list[str]
    judge_feedback: dict
    iteration: int
    max_retries: int

class TeeOutput:
    """Custom sys.stdout that writes to both console and a file, unbuffered."""
    def __init__(self, filename: str):
        self.file = open(filename, "w", encoding="utf-8", buffering=1)
        self.console = sys.__stdout__

    def write(self, data):
        self.console.write(data)
        self.console.flush()
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.console.flush()
        self.file.flush()

class AnsweringPrompt:
    BASE_TEMPLATE = (
        "You are a concise analytical problem solver. Use short chain-of-thought sentences. "
        "Keep the reasoning brief and directly relevant.\n"
        "Previous reasoning (if any): {reasoning_history}\n\n"
        "Question: {question}\n\n"
        "Output only the final answer within <answer> tags.\n"
        "<answer>\n"
        "[Final answer]\n"
        "</answer>"
    )
    PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["question", "reasoning_history"],
        template=BASE_TEMPLATE
    )

class ReviewerPrompt:
    REVISION_TEMPLATE = PromptTemplate(
        input_variables=["question", "previous_reasoning", "feedback"],
        template=(
            "Previous Reasoning Attempt:\n"
            "{previous_reasoning}\n\n"
            "Validation Feedback:\n"
            "{feedback}\n\n"
            "Revise the reasoning using a different approach to solve the problem. "
            "Do not include any extra or irrelevant reasoning.\n\n"
            "Output only the revised chain-of-thought.\n"
        )
    )

##########################################################################
# LangGraph-based EnhancedVerificationAgent using strict workflow
##########################################################################
class EnhancedVerificationAgent:
    """
    Implements a workflow that cycles:
       generate_answer → llm_as_a_judge → (if not CORRECT and retries remain) 
       → revise_reasoning → regenerate_answer → llm_as_a_judge → ...
    It repeats until the judge returns CORRECT or until max_retries is reached.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        # Requested max tokens
        self.max_tokens = 128000
        # Check against a presumed model limit (adjust MODEL_MAX as needed)
        MODEL_MAX = 128000
        if self.max_tokens > MODEL_MAX:
            sys.stdout.write(f"[Warning] Requested max_tokens {self.max_tokens} exceeds the model limit of {MODEL_MAX}. Using {MODEL_MAX} instead.\n")
            sys.stdout.flush()
            self.max_tokens = MODEL_MAX

        self.temperature = 0.8
        self.workflow = self.build_workflow().compile()
        # This variable accumulates the chain-of-thought from delta.reasoning
        self.extracted_reasoning = ""

    def invoke_llm(self, prompt: str, only_think: bool = False) -> str:
        """
        Invokes the LLM, streaming tokens as they arrive.

        If only_think is False, both normal content (including the <answer> block)
        and chain-of-thought tokens (from delta.reasoning) are printed.
        If only_think is True, only chain-of-thought tokens are printed.
        The chain-of-thought is accumulated in self.extracted_reasoning,
        and response_text collects content tokens.
        A timeout is set to 600 seconds to keep the stream open even if generation lags.
        Retries are attempted if a streaming error occurs.
        """
        response_text = ""
        self.extracted_reasoning = ""
        last_token_time = time.time()
        retries = 0
        max_streaming_retries = 2

        while retries <= max_streaming_retries:
            try:
                completion = self.client.chat.completions.create(
                    model="deepseek/deepseek-r1:free",
                    messages=[
                        {"role": "system", "content": "You are an analytical problem solver. Keep chain-of-thoughts short."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=600
                )

                for chunk in completion:
                    content = getattr(chunk.choices[0].delta, "content", "")
                    reasoning = getattr(chunk.choices[0].delta, "reasoning", "")

                    if not only_think:
                        if content:
                            response_text += content
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            last_token_time = time.time()
                        if reasoning:
                            self.extracted_reasoning += reasoning
                            sys.stdout.write(reasoning)
                            sys.stdout.flush()
                            last_token_time = time.time()
                    else:
                        if reasoning:
                            self.extracted_reasoning += reasoning
                            sys.stdout.write(reasoning)
                            sys.stdout.flush()
                            last_token_time = time.time()

                    # If no tokens for more than 30 seconds, log and wait briefly.
                    if time.time() - last_token_time > 30:
                        sys.stdout.write("[Debug] More than 30 seconds since last token. Continuing wait.\n")
                        sys.stdout.flush()
                        time.sleep(1)
                break  # Completed streaming successfully; exit retry loop.

            except httpcore.RemoteProtocolError as e:
                sys.stdout.write(f"\n[Warning] Streaming interrupted on attempt {retries + 1}. Partial output used.\nError: {str(e)}\n")
                sys.stdout.flush()
                retries += 1
                if retries <= max_streaming_retries:
                    sys.stdout.write(f"[Info] Retrying streaming... Attempt {retries + 1}.\n")
                    sys.stdout.flush()
                    last_token_time = time.time()
                else:
                    break
            except Exception as e:
                sys.stdout.write(f"\n[Warning] An error occurred during streaming.\nError: {str(e)}\n")
                sys.stdout.flush()
                break

        sys.stdout.write("\n")
        sys.stdout.flush()
        return response_text.strip()

    def parse_response_tags(self, text: str) -> tuple[str, str]:
        """
        Extracts the chain-of-thought from self.extracted_reasoning
        and the final answer from <answer>...</answer> in the text.
        """
        reasoning = self.extracted_reasoning

        ans_start = text.find("<answer>")
        ans_end = text.find("</answer>")
        if ans_start != -1 and ans_end != -1:
            answer = text[ans_start + len("<answer>"):ans_end]
        elif ans_start != -1:
            answer = text[ans_start + len("<answer>"):]
            sys.stdout.write("[Warning] Final answer tag not closed. Answer might be incomplete.\n")
            sys.stdout.flush()
        else:
            answer = "No answer extracted."

        return strip_tags(reasoning).strip(), strip_tags(answer).strip()

    ############################################################
    # 1) generate_answer
    ############################################################
    def generate_answer(self, state: VerificationState) -> VerificationState:
        prompt = AnsweringPrompt.PROMPT_TEMPLATE.format(
            question=state["question"],
            reasoning_history=("No previous reasoning." if not state["reasoning_history"] else "\n\n".join(state["reasoning_history"][-3:]))
        )

        sys.stdout.write("=========================\n")
        sys.stdout.write("Chain of Thought from generate_answer()\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        full_response = self.invoke_llm(prompt, only_think=False)
        reasoning, answer = self.parse_response_tags(full_response)
        if not reasoning:
            reasoning = "No chain-of-thought extracted."
        if not answer:
            answer = "No answer extracted."
        state["current_answer"] = answer
        state["reasoning_history"].append(reasoning)
        state["iteration"] += 1

        sys.stdout.write("=========================\n")
        sys.stdout.write("Generated Answer from generate_answer()\n")
        sys.stdout.write(f"{answer}\n")
        sys.stdout.write("=========================\n")
        sys.stdout.flush()

        return state

    ############################################################
    # 2) llm_as_a_judge
    ############################################################
    def llm_as_a_judge(self, state: VerificationState) -> VerificationState:
        prompt = (
            f"Validate this answer for '{state['question']}':\n"
            f"Answer: {state['current_answer']}\n"
            f"Reasoning History: {' → '.join(state['reasoning_history'][-2:])}\n\n"
            "Your response must be a valid JSON object and nothing else. Do not include any extra text or explanation.\n"
            "Output JSON with the following keys:\n"
            "- \"status\": should be either \"CORRECT\", \"PARTIAL\", or \"INCORRECT\".\n"
            "- \"detailed_feedback\": bullet points of issues.\n"
            "- \"retainable_elements\": good parts to keep.\n"
        )

        sys.stdout.write("=========================\n")
        sys.stdout.write("LLM Judge's Output\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        # Capture full output (chain-of-thought and JSON) from the judge.
        full_response = self.invoke_llm(prompt, only_think=False)
        # Extract the JSON block from the full response.
        match = JSON_RE.search(full_response)
        if not match:
            state["judge_feedback"] = {"status": "INCORRECT", "detailed_feedback": "Invalid or missing JSON", "retainable_elements": "None"}
            return state

        try:
            decision = json.loads(match.group())
        except json.JSONDecodeError:
            decision = {"status": "INCORRECT", "detailed_feedback": "Invalid JSON", "retainable_elements": "None"}

        decision.setdefault("detailed_feedback", "No feedback")
        decision.setdefault("retainable_elements", "None")

        # If no answer was generated, force judge status to INCORRECT.
        if state["current_answer"].strip() in ("", "No answer extracted."):
            decision["status"] = "INCORRECT"
            decision["detailed_feedback"] = "No answer was generated."

        # Print the JSON output in its own block.
        sys.stdout.write("\n=========================\n")
        sys.stdout.write("LLM JSON Output\n")
        sys.stdout.write(json.dumps(decision, indent=2))
        sys.stdout.write("\n=========================\n")
        sys.stdout.flush()

        state["judge_feedback"] = decision
        return state

    ############################################################
    # 3) revise_reasoning
    ############################################################
    def revise_reasoning(self, state: VerificationState) -> VerificationState:
        sys.stdout.write("=========================\n")
        sys.stdout.write("Revised Chain of Thought from revise_reasoning()\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        # Use the entire chain-of-thought history
        previous_reasoning = "\n\n".join(state["reasoning_history"])
        feedback = state["judge_feedback"].get("detailed_feedback", "No feedback")

        prompt = ReviewerPrompt.REVISION_TEMPLATE.format(
            question=state["question"],
            previous_reasoning=previous_reasoning,
            feedback=feedback
        )

        revised_text = self.invoke_llm(prompt, only_think=False)
        # Now, since there are no <revised_think> tags, take the entire output as the revised chain-of-thought.
        revised_coT = revised_text.strip()

        state["reasoning_history"].append(revised_coT)
        return state

    ############################################################
    # 4) regenerate_answer
    ############################################################
    def regenerate_answer(self, state: VerificationState) -> VerificationState:
        iteration_count = state["iteration"] + 1
        sys.stdout.write("=========================\n")
        sys.stdout.write(f"Regenerated Answer's Chain of Thought #{iteration_count}\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        prompt = AnsweringPrompt.PROMPT_TEMPLATE.format(
            question=state["question"],
            reasoning_history="\n\n".join(state["reasoning_history"][-3:])
        )

        response_text = self.invoke_llm(prompt, only_think=False)
        reasoning, answer = self.parse_response_tags(response_text)
        if not answer:
            answer = "No answer extracted."

        sys.stdout.write("=========================\n")
        sys.stdout.write("Regenerated Answer\n")
        sys.stdout.write(f"{answer}\n")
        sys.stdout.write("=========================\n")
        sys.stdout.flush()

        state["current_answer"] = strip_tags(answer)
        state["reasoning_history"].append(strip_tags(reasoning))
        state["iteration"] += 1
        return state

    ############################################################
    # Build Workflow with Judge → Revise → Regenerate → Judge cycle
    ############################################################
    def build_workflow(self):
        workflow = StateGraph(VerificationState)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("llm_as_a_judge", self.llm_as_a_judge)
        workflow.add_node("revise_reasoning", self.revise_reasoning)
        workflow.add_node("regenerate_answer", self.regenerate_answer)

        workflow.set_entry_point("generate_answer")
        workflow.add_edge("generate_answer", "llm_as_a_judge")

        def judge_condition(state: VerificationState):
            if state["judge_feedback"].get("status") == "CORRECT":
                return "accept"
            if state["iteration"] >= state["max_retries"]:
                return "accept"
            return "revise"

        workflow.add_conditional_edges("llm_as_a_judge", judge_condition, {
            "accept": END,
            "revise": "revise_reasoning"
        })

        workflow.add_edge("revise_reasoning", "regenerate_answer")
        workflow.add_edge("regenerate_answer", "llm_as_a_judge")
        return workflow

    def run(self, question: str) -> VerificationState:
        state: VerificationState = {
            "question": question,
            "current_answer": "",
            "reasoning_history": [],
            "judge_feedback": {},
            "iteration": 0,
            "max_retries": 3
        }
        state = self.workflow.invoke(state)
        return state

if __name__ == "__main__":
    sys.stdout = TeeOutput("output_log.txt")
    agent = EnhancedVerificationAgent()
    question_text = "What letter comes next in this series: W, L, C, N, I, T?"
    final_state = agent.run(question_text)
    sys.stdout.write("\n===== FINAL ANSWER =====\n")
    sys.stdout.write(f"{final_state['current_answer']}\n")
    sys.stdout.write("========================\n\n")
    sys.stdout.flush()