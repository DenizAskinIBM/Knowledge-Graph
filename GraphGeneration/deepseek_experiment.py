## Copyright Deniz Askin. Edited with GPT-o1-pro
import os
import re
import json
import httpcore
import sys
from dotenv import load_dotenv
from typing_extensions import TypedDict

# If you installed via "pip install langchain-core", adjust imports accordingly:
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# Regex patterns
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
REVISED_THINK_BLOCK_PATTERN = re.compile(r"<revised_think>(.*?)</revised_think>", re.DOTALL)
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
        # buffering=1 means line-buffered in text mode; combined with flush() calls
        # after every write, this effectively prints in real-time to file as well.
        self.file = open(filename, "w", encoding="utf-8", buffering=1)
        self.console = sys.__stdout__

    def write(self, data):
        # Write to console unbuffered
        self.console.write(data)
        self.console.flush()
        # Write to file unbuffered
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
        "Provide a brief, relevant chain-of-thought within <think> tags, then output only the final answer "
        "within <answer> tags.\n"
        "<think>\n"
        "[Brief chain-of-thought]\n"
        "</think>\n"
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
            "Output ONLY the revised chain-of-thought within <revised_think> tags.\n"
            "<revised_think>\n"
            "[Revised chain-of-thought]\n"
            "</revised_think>"
        )
    )

##########################################################################
# LangGraph-based EnhancedVerificationAgent using strict workflow
##########################################################################
class EnhancedVerificationAgent:
    """
    Implements a linear workflow:
       generate_answer -> llm_as_a_judge -> revise_reasoning -> regenerate_answer

    * Streams in real-time *.
    """

    def __init__(self):
        # Placeholder endpoint for demonstration.
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        self.temperature = 0.1
        self.max_tokens = 1000
        self.workflow = self.build_workflow().compile()

    def invoke_llm(self, prompt: str, only_think: bool = False) -> str:
        """
        Invokes the LLM, streaming tokens as they arrive.

        If only_think=True, we only print text that appears between <think>...</think>
        in real time. Everything else (including <answer> tags) is captured but *not* printed.

        If only_think=False, we print *all* tokens in real time.
        """
        response_text = ""
        in_think_block = False
        buffer = ""

        try:
            completion = self.client.chat.completions.create(
                model="deepseek-ai/deepseek-r1",
                messages=[
                    {"role": "system", "content": "You are an analytical problem solver. Keep chain-of-thoughts short."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in completion:
                token = getattr(chunk.choices[0].delta, "content", "")
                if not token:
                    continue

                # Accumulate for later parsing
                response_text += token

                # If we are printing everything, just print and flush
                if not only_think:
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    continue

                # If only_think=True, we parse and print only what's inside <think>...</think>
                buffer += token
                # We may get partial or complete tags. Process them in a loop:
                while True:
                    if not in_think_block:
                        # We look for <think>
                        start_idx = buffer.find("<think>")
                        if start_idx == -1:
                            # No opening tag found -> break
                            break
                        # Found an opening tag
                        in_think_block = True
                        # Discard everything up to (and including) <think> from the buffer
                        buffer = buffer[start_idx + len("<think>"):]
                    else:
                        # Already in a <think> block. Look for closing </think>
                        end_idx = buffer.find("</think>")
                        if end_idx == -1:
                            # No closing tag found -> print everything so far
                            sys.stdout.write(buffer)
                            sys.stdout.flush()
                            buffer = ""  # empty it out
                            break
                        # If we found a closing tag
                        content_to_print = buffer[:end_idx]
                        sys.stdout.write(content_to_print)
                        sys.stdout.flush()

                        # Remove that printed content + </think> from buffer
                        buffer = buffer[end_idx + len("</think>"):]
                        in_think_block = False

            # End of streaming: If we ended inside a <think> with no closing tag,
            # print whatever remains in the buffer
            if only_think and in_think_block and buffer:
                sys.stdout.write(buffer)
                sys.stdout.flush()

            # Print a newline at the end of the entire stream (for neatness)
            sys.stdout.write("\n")
            sys.stdout.flush()

        except httpcore.RemoteProtocolError as e:
            sys.stdout.write(f"\n[Warning] Streaming interrupted. Partial output used.\nError: {str(e)}\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"\n[Warning] An error occurred during streaming.\nError: {str(e)}\n")
            sys.stdout.flush()

        return response_text.strip()

    def parse_response_tags(self, text: str) -> tuple[str, str]:
        """
        Extract the chain-of-thought from <think>...</think>
        and the final answer from <answer>...</answer>.
        Ensures that reasoning and answer are returned even if incomplete.
        """
        # Remove accidental <answer<think> blocks
        cleaned_text = re.sub(r"<answer<think>.*?</think>", "", text, flags=re.DOTALL)

        # Extract the first <think> block
        m = re.search(r"<think>(.*?)</think>", cleaned_text, flags=re.DOTALL)
        reasoning = m.group(1) if m else "Partial or missing reasoning."

        # Remove all <think> blocks from the text so we can find <answer>
        text_without_think = THINK_BLOCK_PATTERN.sub("", cleaned_text)
        ans_start = text_without_think.find("<answer>")
        ans_end = text_without_think.find("</answer>")
        answer = ""
        
        if ans_start != -1 and ans_end != -1:
            answer = text_without_think[ans_start + len("<answer>"):ans_end]
        elif ans_start != -1:
            answer = text_without_think[ans_start + len("<answer>"):]  # Capture partial answer
            answer = answer.strip() + " (incomplete)"
        else:
            answer = "No answer extracted."

        return strip_tags(reasoning).strip(), strip_tags(answer).strip()


    ############################################################
    # 1) generate_answer
    ############################################################
    def generate_answer(self, state: VerificationState) -> VerificationState:
        """
        Generates an initial answer. 
        **Prints only <think> blocks** in real time (omits <answer> from streaming).
        """
        prompt = AnsweringPrompt.PROMPT_TEMPLATE.format(
            question=state["question"],
            reasoning_history=(
                "No previous reasoning."
                if not state["reasoning_history"]
                else "\n\n".join(state["reasoning_history"][-3:])
            )
        )

        sys.stdout.write("=========================\n")
        sys.stdout.write("Chain of Thought from generate_answer()\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        # Stream only chain-of-thought
        full_response = self.invoke_llm(prompt, only_think=True)

        # Parse out the reasoning and final answer
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
        """
        Judge correctness of the current answer.
        **Prints only <think> blocks** in real time (omits <answer> from streaming).
        """
        prompt = (
            f"Validate this answer for '{state['question']}':\n"
            f"Answer: {state['current_answer']}\n"
            f"Reasoning History: {' → '.join(state['reasoning_history'][-2:])}\n\n"
            "Output JSON with:\n"
            "- \"status\": \"CORRECT\", \"PARTIAL\", or \"INCORRECT\"\n"
            "- \"detailed_feedback\": Bullet points of issues\n"
            "- \"retainable_elements\": Good parts to keep\n"
        )

        sys.stdout.write("=========================\n")
        sys.stdout.write("LLM Judge's Chain of Thought from llm_as_a_judge\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        # Stream only chain-of-thought
        response_text = self.invoke_llm(prompt, only_think=True)

        # Remove chain-of-thought blocks to parse JSON
        text_stripped = THINK_BLOCK_PATTERN.sub("", response_text)
        text_stripped = REVISED_THINK_BLOCK_PATTERN.sub("", text_stripped)

        match = JSON_RE.search(text_stripped)
        if not match:
            state["judge_feedback"] = {
                "status": "INCORRECT",
                "detailed_feedback": "Invalid or missing JSON",
                "retainable_elements": "None"
            }
            return state

        try:
            decision = json.loads(match.group())
        except json.JSONDecodeError:
            decision = {
                "status": "INCORRECT",
                "detailed_feedback": "Invalid JSON",
                "retainable_elements": "None"
            }

        decision.setdefault("detailed_feedback", "No feedback")
        decision.setdefault("retainable_elements", "None")

        sys.stdout.write("=========================\n")
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
        """
        Revise chain-of-thought based on judge feedback. 
        **Prints ALL tokens** in real time.
        """
        sys.stdout.write("=========================\n")
        sys.stdout.write("Revised Chain of Thought from revise_reasoning()\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        previous_reasoning = state["reasoning_history"][-1]
        feedback = state["judge_feedback"].get("detailed_feedback", "No feedback")

        prompt = ReviewerPrompt.REVISION_TEMPLATE.format(
            question=state["question"],
            previous_reasoning=previous_reasoning,
            feedback=feedback
        )

        # Stream all tokens
        revised_text = self.invoke_llm(prompt, only_think=False)
        revised_matches = REVISED_THINK_BLOCK_PATTERN.findall(revised_text)
        revised_coT = revised_matches[-1].strip() if revised_matches else revised_text.strip()

        # Update the last reasoning with the newly revised chain-of-thought
        state["reasoning_history"][-1] = revised_coT
        return state

    ############################################################
    # 4) regenerate_answer
    ############################################################
    def regenerate_answer(self, state: VerificationState) -> VerificationState:
        """
        Produce a final answer using the revised chain-of-thought.
        **Prints ALL tokens** in real time.
        """
        iteration_count = state["iteration"] + 1
        sys.stdout.write("=========================\n")
        sys.stdout.write(f"Regenerated Answer's Chain of Thought #{iteration_count}\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        prompt = AnsweringPrompt.PROMPT_TEMPLATE.format(
            question=state["question"],
            reasoning_history="\n\n".join(state["reasoning_history"][-3:])
        )

        # Stream all tokens
        response_text = self.invoke_llm(prompt, only_think=False)

        # Parse out final answer
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
    # Build LangGraph Workflow (strict linear)
    ############################################################
    def build_workflow(self):
        workflow = StateGraph(VerificationState)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("llm_as_a_judge", self.llm_as_a_judge)
        workflow.add_node("revise_reasoning", self.revise_reasoning)
        workflow.add_node("regenerate_answer", self.regenerate_answer)

        workflow.set_entry_point("generate_answer")
        workflow.add_edge("generate_answer", "llm_as_a_judge")

        # Condition: If judge says "CORRECT", end. Otherwise, revise.
        def judge_condition(state: VerificationState):
            return "accept" if state["judge_feedback"].get("status") == "CORRECT" else "revise"

        workflow.add_conditional_edges("llm_as_a_judge", judge_condition, {
            "accept": END,
            "revise": "revise_reasoning"
        })

        workflow.add_edge("revise_reasoning", "regenerate_answer")
        workflow.add_edge("regenerate_answer", END)
        return workflow

    def run(self, question: str) -> VerificationState:
        """
        Entry point to run the entire workflow on a single question.
        """
        state: VerificationState = {
            "question": question,
            "current_answer": "",
            "reasoning_history": [],
            "judge_feedback": {},
            "iteration": 0,
            "max_retries": 1
        }
        # Run the compiled workflow
        state = self.workflow.invoke(state)
        return state

if __name__ == "__main__":
    # This ensures we see unbuffered output in many shells:
    #   PYTHONUNBUFFERED=1 python -u deepseek_experiment.py
    # Or you can set it in the environment or run with -u explicitly.

    sys.stdout = TeeOutput("output_log.txt")
    agent = EnhancedVerificationAgent()
    question_text = "What letter comes next in this series: W, L, C, N, I, T?"
    final_state = agent.run(question_text)

    sys.stdout.write("\n===== FINAL ANSWER =====\n")
    sys.stdout.write(f"{final_state['current_answer']}\n")
    sys.stdout.write("========================\n\n")
    sys.stdout.flush()