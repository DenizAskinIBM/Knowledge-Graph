import os
import re
import json
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

load_dotenv()

# Pre-compile regex patterns
JSON_RE = re.compile(r'\{.*\}', re.DOTALL)
STRIP_TAGS_RE = re.compile(r'<.*?>')
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def strip_tags(text: str) -> str:
    """Remove HTML-like tags from text."""
    return STRIP_TAGS_RE.sub('', text).strip()

# --------------------- ENHANCED PROMPTS ---------------------
class AnsweringPrompt:
    BASE_TEMPLATE = (
        "You are a concise analytical problem solver. Use minimal chain-of-thought sentences. Keep the reasoning brief and straight to the answer."
        "and include only reasoning directly relevant to the question.\n"
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
    COT_TEMPLATE = PromptTemplate(
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

# --------------------- STATE & WORKFLOW ---------------------
class VerificationState(TypedDict):
    question: str
    current_answer: str
    reasoning_history: list
    judge_feedback: dict
    iteration: int

class EnhancedVerificationAgent:
    def __init__(self):
        """
        A minimal, synchronous workflow that uses the OpenAI client in streaming mode.
        This version instructs the model to be concise and relevant in its chain-of-thought.
        """
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        self.temperature = 0.2
        self.max_tokens = 512
        self.workflow = self.build_workflow()

    def invoke_llm(self, prompt: str) -> str:
        """
        Invoke the LLM using the streaming API.
        We stop chain-of-thought printing once <answer> is encountered,
        so the final answer isn't revealed in the chain-of-thought portion.

        We also remove any "CHAIN-OF-THOUGHT:" text to avoid duplication.
        """
        completion = self.client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            messages=[
                {"role": "system", "content": "You are an analytical problem solver. Make sure to keep your chain-of-thoughts short. "},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=self.max_tokens,
            stream=True
        )
        tokens = []

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token is not None:
                # Remove model-emitted "CHAIN-OF-THOUGHT:" text if present
                if "CHAIN-OF-THOUGHT:" in token:
                    token = token.replace("CHAIN-OF-THOUGHT:", "")

                if "<answer>" in token:
                    # Encountered the start of <answer>, so separate out what's before it
                    before, sep, after = token.partition("<answer>")
                    if before:
                        # Print chain-of-thought portion (no final answer)
                        print(before, end="", flush=True)
                        tokens.append(before)
                    print("\n-------------------------------------------------")
                    print("FINAL ANSWER:")
                    print(after, end="", flush=True)
                    tokens.append(sep + after)
                else:
                    # Print chain-of-thought or other text
                    print(token, end="", flush=True)
                    tokens.append(token)

        print()  # Newline after streaming completes
        return "".join(tokens)

    def parse_response_tags(self, text: str) -> tuple[str, str]:
        """
        Extract the content between <think>...</think> as the chain-of-thought,
        and <answer>...</answer> as the final answer.
        """
        reasoning = ""
        answer = ""
        think_start = text.find("<think>")
        if think_start != -1:
            think_end = text.find("</think>", think_start + len("<think>"))
            if think_end != -1:
                reasoning = text[think_start + len("<think>"):think_end]
                answer_start = text.find("<answer>", think_end)
                answer_end = text.find("</answer>", answer_start)
                if answer_start != -1 and answer_end != -1:
                    answer = text[answer_start + len("<answer>"):answer_end]
                else:
                    answer = text[think_end + len("</think>"):]
            else:
                reasoning = text[think_start + len("<think>"):]
        else:
            answer = text
        reasoning = strip_tags(reasoning)
        answer = strip_tags(answer)
        return reasoning.strip(), answer.strip()

    # ------------ CORE NODES (Synchronous) ------------
    def generate_answer(self, state: VerificationState) -> VerificationState:
        """
        This node generates an initial answer using the model.
        """
        prompt = AnsweringPrompt.COT_TEMPLATE.format(
            question=state["question"],
            reasoning_history="No previous reasoning."
        )
        print("GENERATED CHAIN-OF-THOUGHT (streamed) from generate_answer():")
        response_text = self.invoke_llm(prompt)

        reasoning, answer = self.parse_response_tags(response_text)
        state["current_answer"] = answer
        state["reasoning_history"].append(reasoning)
        state["iteration"] += 1

        print("-------------------------------------------------")
        print("GENERATED ANSWER from generate_answer():")
        print(answer)
        print("-------------------------------------------------\n")
        return state

    def regenerate_answer(self, state: VerificationState) -> VerificationState:
        """
        This node regenerates an answer (after feedback/revision).
        """
        prompt = AnsweringPrompt.COT_TEMPLATE.format(
            question=state["question"],
            reasoning_history="\n\n".join(state["reasoning_history"][-3:])
        )
        print("-------------------------------------------------\n")
        print("REGENERATED CHAIN-OF-THOUGHT (streamed) from regenerate_answer():")
        response_text = self.invoke_llm(prompt)

        reasoning, answer = self.parse_response_tags(response_text)
        state["current_answer"] = answer
        state["reasoning_history"].append(reasoning)
        state["iteration"] += 1

        print("-------------------------------------------------\n")
        print("REGENERATED ANSWER from regenerate_answer():")
        print(answer)
        print("-------------------------------------------------\n")
        return state

    def judge_answer(self, state: VerificationState) -> VerificationState:
        """
        This node validates the current answer. If it's correct, the workflow ends.
        Otherwise, we proceed to revise the reasoning.
        
        We also handle the possibility that the model might produce multiple <think> blocks
        by labeling each one clearly before we strip them out for JSON parsing.
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

        # Print a label so we know what's happening if the chain-of-thought starts right away
        print("JUDGE CHAIN-OF-THOUGHT (streamed) from judge_answer():")

        response_text = self.invoke_llm(prompt)

        # If the judge output contains one or more <think> blocks, handle each:
        matches = list(THINK_BLOCK_PATTERN.finditer(response_text))
        for i, match in enumerate(matches, start=1):
            judge_think = match.group(1)
            print("-------------------------------------------------")
            if len(matches) == 1:
                print("JUDGE CHAIN-OF-THOUGHT from judge_answer():")
            else:
                # If there's more than one <think>, label them individually
                print(f"JUDGE CHAIN-OF-THOUGHT from judge_answer() (paragraph {i}):")
            print(strip_tags(judge_think))
            print("-------------------------------------------------\n")

        # Remove all <think>...</think> blocks from the response before we do JSON parsing
        response_text = THINK_BLOCK_PATTERN.sub("", response_text)

        match = JSON_RE.search(response_text)
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
            decision = {"status": "INCORRECT", "detailed_feedback": "Invalid JSON"}

        decision.setdefault("detailed_feedback", "No feedback")
        decision.setdefault("retainable_elements", "None")

        # Even if the model says CORRECT, override if the answer is empty
        if decision["status"] == "CORRECT" and not state["current_answer"].strip():
            decision["status"] = "INCORRECT"
            decision["detailed_feedback"] = "Empty answer received"

        state["judge_feedback"] = decision
        print("-------------------------------------------------")
        print("JUDGE DECISION (JSON) from judge_answer():")
        print(str(decision))
        print("-------------------------------------------------\n")
        return state

    def revise_reasoning(self, state: VerificationState) -> VerificationState:
        """
        This node prompts the model to revise its chain-of-thought, based on judge feedback.
        """
        previous_reasoning = state["reasoning_history"][-1]
        feedback = state["judge_feedback"].get("detailed_feedback", "No feedback")

        prompt = ReviewerPrompt.REVISION_TEMPLATE.format(
            question=state["question"],
            previous_reasoning=previous_reasoning,
            feedback=feedback
        )
        revised_text = self.invoke_llm(prompt)

        # Append the revised chain-of-thought to the last reasoning step
        state["reasoning_history"][-1] = f"{previous_reasoning}\nREVISED: {revised_text}"
        return state

    # ------------ WORKFLOW ------------
    def build_workflow(self):
        """
        Build the StateGraph so that we always go:
          generate -> judge
        Then judge decides if we accept (END) or go to revise.
        """
        workflow = StateGraph(VerificationState)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("regenerate", self.regenerate_answer)
        workflow.add_node("judge", self.judge_answer)
        workflow.add_node("revise", self.revise_reasoning)

        # Entry point: generate
        workflow.set_entry_point("generate")

        # Always call judge next (unconditional)
        workflow.add_edge("generate", "judge")

        # The judge condition: accept if status == "CORRECT", else revise
        workflow.add_conditional_edges(
            "judge",
            lambda s: "accept" if s["judge_feedback"].get("status") == "CORRECT" else "revise",
            {"accept": END, "revise": "revise"}
        )

        # If we revise, we regenerate next
        workflow.add_edge("revise", "regenerate")

        return workflow.compile()

    def run(self, question: str, max_retries: int = 3):
        """
        Run the entire verification workflow for a given question.
        We do not break early if the answer is empty.
        We let judge_answer() run so it can parse and provide feedback.
        """
        state: VerificationState = {
            "question": question,
            "current_answer": "",
            "reasoning_history": [],
            "judge_feedback": {},
            "iteration": 1
        }

        for _ in range(max_retries):
            state = self.workflow.invoke(state)
            # If the judge says CORRECT, we stop
            if state["judge_feedback"].get("status") == "CORRECT":
                break

        return self.format_result(state)

    def format_result(self, state: VerificationState):
        """
        Return a dictionary of final results: the answer, reasoning steps, iteration count, and feedback.
        """
        return {
            "final_answer": state["current_answer"],
            "reasoning_steps": state["reasoning_history"],
            "iterations": state["iteration"],
            "feedback_history": state["judge_feedback"]
        }

if __name__ == "__main__":
    agent = EnhancedVerificationAgent()
    question_text = (
        "Think of a common greeting in a country that is not the United States. You can rearrange its letters to get the capital of a country that neighbors the country where this greeting is commonly spoken. What greeting is it?"
    )
    result = agent.run(question_text)
    print("Final Answer:\n", result["final_answer"], "\n")
    # Uncomment below to see the chain-of-thought steps in detail:
    # print("Reasoning Steps:")
    # for idx, step in enumerate(result["reasoning_steps"], start=1):
    #     print(f"Iteration {idx}:\n{step}\n")