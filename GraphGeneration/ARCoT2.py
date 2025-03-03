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

# Regex patterns
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
    judge_history: list[dict]

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

##########################################################################
# LangGraph-based EnhancedVerificationAgent using simplified workflow
##########################################################################
class EnhancedVerificationAgent:
    """
    Simplified workflow:
        generate_answer -> llm_as_a_judge -> regenerate_answer
    Repeats until judge returns "TRUE" or max_retries is reached.

    - generate_answer: Generates an answer without passing any chain-of-thought
      to the prompt (initial, cold start).
    - llm_as_a_judge: Decides if the answer follows logically from the chain-of-thought.
      Additionally, we do an internal check (not revealed to the LLM) for correctness.
      In this scenario, we know that the correct letter is 'S', but we do NOT
      tell that to the LLM's prompt.
    - regenerate_answer: If the judge deems the answer incorrect, we prompt the LLM to
      revise the reasoning, referring only to judge feedback that the previous attempt
      did not logically produce the correct letter.

    The code does NOT reveal the correct letter 'S' to the LLM in any prompt.
    Instead, the code checks correctness on its own (internally) and triggers regeneration
    if the answer is not 'S'.

    Temperature for each agent can be set via the self.agent_temperatures dictionary.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        self.max_tokens = 8192  
        MODEL_MAX = 128000

        if self.max_tokens > MODEL_MAX:
            sys.stdout.write(f"[Warning] Requested max_tokens {self.max_tokens} exceeds the model limit of {MODEL_MAX}. Using {MODEL_MAX} instead.\n")
            sys.stdout.flush()
            self.max_tokens = MODEL_MAX

        # Default temperatures for each agent; modify these values as needed.
        self.agent_temperatures = {
            "generate_answer": 0.7,
            "llm_as_a_judge": 0.0,
            "regenerate_answer": 0.9
        }

        self.workflow = self.build_workflow().compile()

        # Holds the raw chain-of-thought from streaming.
        self.extracted_reasoning = ""

    ########################################################################
    # invoke_llm METHOD – printing tokens in real-time and then printing final aligned output
    ########################################################################
    def invoke_llm(self, prompt: str, only_think: bool = False, temperature: float = None) -> str:
        self.extracted_reasoning = ""
        response_text = ""
        reasoning_accumulator = []  # Accumulate chain-of-thought tokens
        answer_accumulator = []     # Accumulate final answer tokens

        temp = temperature  # The specific temperature for this call

        last_token_time = time.time()
        retries = 0
        max_streaming_retries = 2

        while retries <= max_streaming_retries:
            try:
                completion = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are an analytical problem solver. Keep chain-of-thoughts short."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temp,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=600
                )

                for chunk in completion:
                    delta = chunk.choices[0].delta
                    reasoning_content = getattr(delta, "reasoning_content", "")
                    content = getattr(delta, "content", "")

                    # Print tokens in real-time and accumulate them.
                    if reasoning_content:
                        reasoning_accumulator.append(reasoning_content)
                        sys.stdout.write(reasoning_content)
                        sys.stdout.flush()
                        last_token_time = time.time()

                    if content:
                        answer_accumulator.append(content)
                        sys.stdout.write(content)
                        sys.stdout.flush()
                        last_token_time = time.time()

                    if time.time() - last_token_time > 30:
                        sys.stdout.write("[Debug] More than 30 seconds since last token. Continuing wait.\n")
                        sys.stdout.flush()
                        time.sleep(1)
                break

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

        # Join the accumulated tokens.
        self.extracted_reasoning = "".join(reasoning_accumulator)
        response_text = "".join(answer_accumulator)

        # Print final output block, omitting the raw <answer> text from the chain-of-thought display
        sys.stdout.write("\n\n===== FINAL OUTPUT =====\n")
        sys.stdout.write("FINAL CHAIN-OF-THOUGHT:\n")
        reasoning_only = self.extracted_reasoning.split("<answer>")[0]
        sys.stdout.write(reasoning_only + "\n")
        if not only_think:
            sys.stdout.write("FINAL ANSWER:\n")
            sys.stdout.write(response_text + "\n")
        sys.stdout.write("========================\n\n")
        sys.stdout.flush()

        return response_text.strip()

    def parse_response_tags(self, text: str) -> tuple[str, str]:
        # We parse out the chain-of-thought (before <answer>) and the answer (within <answer> ... </answer>)
        reasoning_raw = self.extracted_reasoning.split("<answer>")[0]
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

        return reasoning_raw.strip(), strip_tags(answer).strip()

    ############################################################
    # (1) generate_answer
    ############################################################
    def generate_answer(self, state: dict) -> dict:
        prompt = AnsweringPrompt.PROMPT_TEMPLATE.format(
            question=state["question"],
            reasoning_history=""
        )

        sys.stdout.write("=========================\n")
        sys.stdout.write("Chain of Thought from generate_answer()\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        full_response = self.invoke_llm(
            prompt, 
            only_think=True, 
            temperature=self.agent_temperatures["generate_answer"]
        )
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
    # (2) llm_as_a_judge
    ############################################################
    def llm_as_a_judge(self, state: dict) -> dict:
        """
        This step checks if the final answer logically follows from the chain-of-thought.
        Then we (internally) check correctness: we know the next letter should be 'S',
        but we do NOT reveal that to the LLM's prompt. We only do an internal check here.
        If the answer is not 'S', we mark 'Answer_follows_from_chain_of_thought' as 'FALSE'.
        """
        judge_reasoning_snippet = state["reasoning_history"][-1] if state["reasoning_history"] else ""

        # Prompt the LLM to see if the chain-of-thought is self-consistent.
        # We do NOT reveal the correct letter. We only ask if the final answer follows logically.
        # The LLM's JSON reply is not final: we override with 'FALSE' if the final answer is not 'S'.
        prompt = (
            "You are verifying if the final answer logically follows from this chain-of-thought:\n\n"
            f"Chain-of-thought: {judge_reasoning_snippet}\n\n"
            f"Answer: {state['current_answer']}\n\n"
            "You must respond only with a JSON object of the form:\n"
            "{\n"
            "  \"Answer_follows_from_chain_of_thought\": \"TRUE\" or \"FALSE\"\n"
            "}\n"
            "No extra keys or text."
        )

        sys.stdout.write("=========================\n")
        sys.stdout.write("LLM Judge's Output\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        full_response = self.invoke_llm(
            prompt, 
            only_think=False, 
            temperature=self.agent_temperatures["llm_as_a_judge"]
        )

        # Try to parse the JSON
        match = JSON_RE.search(full_response)
        if not match:
            judge_result = {"Answer_follows_from_chain_of_thought": "FALSE"}
        else:
            try:
                judge_result = json.loads(match.group())
            except json.JSONDecodeError:
                judge_result = {"Answer_follows_from_chain_of_thought": "FALSE"}

        # Internal correctness check: if the final answer is not 'S', we override judge_result to 'FALSE'.
        # The LLM is never told that 'S' is correct. This is purely internal in code.
        if state["current_answer"].strip().upper() != "S":
            judge_result["Answer_follows_from_chain_of_thought"] = "FALSE"

        state["judge_feedback"] = judge_result
        state["judge_history"].append(judge_result)

        sys.stdout.write("\n=========================\n")
        sys.stdout.write("LLM JSON Output\n")
        sys.stdout.write(json.dumps(judge_result, indent=2))
        sys.stdout.write("\n=========================\n")
        sys.stdout.flush()

        return state

    ############################################################
    # (3) regenerate_answer
    ############################################################
    def regenerate_answer(self, state: dict) -> dict:
        """
        If the judge deems the answer incorrect, we ask the LLM to try again.
        We do NOT reveal the correct letter. Instead, we provide the judge feedback
        in a minimal textual summary, saying the previous approach was not correct.
        """
        iteration_count = state["iteration"] + 1
        sys.stdout.write("=========================\n")
        sys.stdout.write(f"Regenerated Answer's Chain of Thought #{iteration_count}\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        # Summarize the judge feedback
        combined_history = ""
        for i, fb in enumerate(state["judge_history"]):
            combined_history += f"Attempt {i+1} Judge Feedback: {fb}\n"

        leading_instruction = (
            "Your previous attempt(s) did not yield a correct next letter according to the judge. "
            "You must produce a different chain-of-thought that leads to the correct next letter. "
            "Try to find a definitive pattern or reasoning that ensures the correct letter. "
            "Use a new approach, ignoring previous flawed numerical reasoning.\n\n"
        )

        prompt_content = (
            leading_instruction +
            AnsweringPrompt.PROMPT_TEMPLATE.format(
                question=state["question"],
                reasoning_history=combined_history
            )
        )

        response_text = self.invoke_llm(
            prompt_content, 
            only_think=False, 
            temperature=self.agent_temperatures["regenerate_answer"]
        )
        reasoning, answer = self.parse_response_tags(response_text)

        if not reasoning:
            reasoning = "No chain-of-thought extracted."
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
    # Build Workflow: generate_answer -> llm_as_a_judge -> regenerate_answer
    ############################################################
    def build_workflow(self):
        workflow = StateGraph(VerificationState)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("llm_as_a_judge", self.llm_as_a_judge)
        workflow.add_node("regenerate_answer", self.regenerate_answer)

        workflow.set_entry_point("generate_answer")
        workflow.add_edge("generate_answer", "llm_as_a_judge")

        def judge_condition(state: VerificationState):
            # We rely on the "Answer_follows_from_chain_of_thought" from the judge feedback
            # or the internal correctness check to see if we accept or regenerate.
            follows = state["judge_feedback"].get("Answer_follows_from_chain_of_thought", "FALSE")
            if follows == "TRUE":
                return "accept"
            if state["iteration"] >= state["max_retries"]:
                return "accept"
            return "regenerate"

        workflow.add_conditional_edges("llm_as_a_judge", judge_condition, {
            "accept": END,
            "regenerate": "regenerate_answer"
        })

        workflow.add_edge("regenerate_answer", "llm_as_a_judge")
        return workflow

    def run(self, question: str) -> VerificationState:
        state: VerificationState = {
            "question": question,
            "current_answer": "",
            "reasoning_history": [],
            "judge_feedback": {},
            "iteration": 0,
            "max_retries": 3,
            "judge_history": []
        }
        state = self.workflow.invoke(state)
        return state

if __name__ == "__main__":
    sys.stdout = TeeOutput("output_log.txt")
    agent = EnhancedVerificationAgent()
    question_text = "What letter comes next in this series: W, L, C, N, I, T?"
    final_state = agent.run(question_text)
    # Print the final answer (omitting <answer> tags).
    sys.stdout.write("\n===== FINAL ANSWER =====\n")
    sys.stdout.write(f"{final_state['current_answer']}\n")
    sys.stdout.write("========================\n\n")
    sys.stdout.flush()