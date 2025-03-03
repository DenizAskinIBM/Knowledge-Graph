## Copyright Deniz Askin. Edited with GPT-o1
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

# Define the state for the cooking process
class CookingState(TypedDict):
    recipe: str
    food_question: str
    cooks: list[str]
    subtasks: str
    cooked_meal: str
    planner_history: list[str]
    executer_history: list[str]
    iteration: int
    max_retries: int

# Custom sys.stdout that writes to both console and a file, unbuffered.
class TeeOutput:
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

# Define prompt templates for the Planner and Executer LLMs
class CookingPrompt:
    PLANNER_TEMPLATE = (
        "You are a creative culinary strategist. Given the following recipe, the dish to be cooked, and a list of kitchen roles, "
        "generate a detailed list of subtasks. Assign each subtask to the appropriate cook from the list provided, ensuring that "
        "the final sequence of subtasks will produce the dish as specified in the recipe.\n\n"
        "Recipe: {recipe}\n\n"
        "Dish to be cooked: {food_question}\n\n"
        "Kitchen Roles: {cooks}\n\n"
        "List the subtasks with the assigned cook role for each step, and enclose your final answer within <answer> and </answer> tags."
    )
    EXECUTER_TEMPLATE = (
        "You are a meticulous kitchen executor. Given the following list of subtasks with assigned cook roles, execute each subtask exactly "
        "as instructed to prepare the final dish strictly following the recipe. Output the final cooked meal, enclosed within <answer> and </answer> tags.\n\n"
        "Subtasks:\n{subtasks}\n\n"
        "Please provide the final cooked meal as the output."
    )
    planner_prompt = PromptTemplate(
        input_variables=["recipe", "food_question", "cooks"],
        template=PLANNER_TEMPLATE
    )
    executer_prompt = PromptTemplate(
        input_variables=["subtasks"],
        template=EXECUTER_TEMPLATE
    )

##########################################################################
# LangGraph-based CookingAgent using a two-step culinary workflow:
# 1. plan_recipe: The Planner LLM creates subtasks for the cooks.
# 2. execute_subtasks: The Executer LLM carries out the subtasks to produce the final dish.
##########################################################################
class CookingAgent:
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
            "plan_recipe": 0.7,
            "execute_subtasks": 0.9
        }

        self.workflow = self.build_workflow().compile()

        # Holds the raw chain-of-thought from streaming.
        self.extracted_reasoning = ""

    ########################################################################
    # invoke_llm METHOD – prints tokens in real-time and then prints final aligned output
    ########################################################################
    def invoke_llm(self, prompt: str, only_think: bool = False, temperature: float = None) -> str:
        self.extracted_reasoning = ""
        response_text = ""
        reasoning_accumulator = []  # Accumulate chain-of-thought tokens
        answer_accumulator = []     # Accumulate final answer tokens

        temp = temperature  # Every call now must pass an explicit temperature.
        last_token_time = time.time()
        retries = 0
        max_streaming_retries = 2

        while retries <= max_streaming_retries:
            try:
                completion = self.client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "You are a culinary expert."},
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
                        sys.stdout.write("[Debug] More than 30 seconds since last token. Waiting...\n")
                        sys.stdout.flush()
                        time.sleep(1)
                break

            except httpcore.RemoteProtocolError as e:
                sys.stdout.write(f"\n[Warning] Streaming interrupted on attempt {retries + 1}. Error: {str(e)}\n")
                sys.stdout.flush()
                retries += 1
                if retries <= max_streaming_retries:
                    sys.stdout.write(f"[Info] Retrying streaming... Attempt {retries + 1}.\n")
                    sys.stdout.flush()
                    last_token_time = time.time()
                else:
                    break
            except Exception as e:
                sys.stdout.write(f"\n[Warning] An error occurred during streaming. Error: {str(e)}\n")
                sys.stdout.flush()
                break

        # Join the accumulated tokens.
        self.extracted_reasoning = "".join(reasoning_accumulator)
        response_text = "".join(answer_accumulator)

        # Print final output without including any answer text in the chain-of-thought.
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
        # Extract chain-of-thought only up to the first occurrence of "<answer>".
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
            answer = text
        return reasoning_raw.strip(), strip_tags(answer).strip()

    ############################################################
    # 1) plan_recipe: Planner LLM generates subtasks assigned to the correct cooks.
    ############################################################
    def plan_recipe(self, state: dict) -> dict:
        prompt = CookingPrompt.planner_prompt.format(
            recipe=state["recipe"],
            food_question=state["food_question"],
            cooks=", ".join(state["cooks"])
        )

        sys.stdout.write("=========================\n")
        sys.stdout.write("Planner LLM Output (Plan Recipe)\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        full_response = self.invoke_llm(
            prompt, 
            only_think=True, 
            temperature=self.agent_temperatures["plan_recipe"]
        )
        reasoning, answer = self.parse_response_tags(full_response)

        if not reasoning:
            reasoning = "No chain-of-thought extracted."
        if not answer:
            answer = "No subtasks generated."

        state["subtasks"] = answer
        state["planner_history"].append(reasoning)
        state["iteration"] += 1

        sys.stdout.write("=========================\n")
        sys.stdout.write("Generated Subtasks from Planner LLM\n")
        sys.stdout.write(f"{answer}\n")
        sys.stdout.write("=========================\n")
        sys.stdout.flush()

        return state

    ############################################################
    # 2) execute_subtasks: Executer LLM carries out the subtasks to produce the final dish.
    ############################################################
    def execute_subtasks(self, state: dict) -> dict:
        prompt = CookingPrompt.executer_prompt.format(
            subtasks=state["subtasks"]
        )
        sys.stdout.write("=========================\n")
        sys.stdout.write("Executer LLM Output (Execute Subtasks)\n")
        sys.stdout.write("=========================\n\n")
        sys.stdout.flush()

        full_response = self.invoke_llm(
            prompt, 
            only_think=False, 
            temperature=self.agent_temperatures["execute_subtasks"]
        )
        reasoning, answer = self.parse_response_tags(full_response)

        if not reasoning:
            reasoning = "No chain-of-thought extracted."
        if not answer:
            answer = "No cooked meal produced."

        state["cooked_meal"] = answer
        state["executer_history"].append(reasoning)
        state["iteration"] += 1

        sys.stdout.write("=========================\n")
        sys.stdout.write("Final Cooked Meal from Executer LLM\n")
        sys.stdout.write(f"{answer}\n")
        sys.stdout.write("=========================\n")
        sys.stdout.flush()

        return state

    ############################################################
    # Build Workflow: plan_recipe -> execute_subtasks
    ############################################################
    def build_workflow(self):
        workflow = StateGraph(CookingState)
        workflow.add_node("plan_recipe", self.plan_recipe)
        workflow.add_node("execute_subtasks", self.execute_subtasks)

        workflow.set_entry_point("plan_recipe")
        workflow.add_edge("plan_recipe", "execute_subtasks")
        return workflow

    def run(self, recipe: str, food_question: str, cooks: list[str]) -> CookingState:
        state: CookingState = {
            "recipe": recipe,
            "food_question": food_question,
            "cooks": cooks,
            "subtasks": "",
            "cooked_meal": "",
            "planner_history": [],
            "executer_history": [],
            "iteration": 0,
            "max_retries": 1
        }
        state = self.workflow.invoke(state)
        return state

if __name__ == "__main__":
    sys.stdout = TeeOutput("cooking_output_log.txt")
    agent = CookingAgent()
    # Sample recipe, dish to be cooked, and list of kitchen roles.
    sample_recipe = (
        "1. Preheat the oven to 375°F. "
        "2. Season the chicken with salt and pepper. "
        "3. Sear the chicken in a hot pan until golden brown. "
        "4. Transfer the chicken to a baking dish and bake for 25 minutes. "
        "5. Prepare a sauce by simmering garlic, lemon, and herbs. "
        "6. Drizzle the sauce over the chicken before serving."
    )
    sample_food_question = "Prepare Roast Chicken with Lemon Herb Sauce"
    cooks_list = [
        "Chef de cuisine",
        "Sous-chef de cuisine",
        "Saucier",
        "Chef de partie",
        "Cuisinier",
        "Commis",
        "Apprenti(e)",
        "Plongeur",
        "Marmiton",
        "Rôtisseur",
        "Grillardin",
        "Friturier",
        "Poissonnier",
        "Entremétier",
        "Potager",
        "Legumier",
        "Garde manger",
        "Charcutier",
        "Tournant",
        "Pâtissier",
        "Confiseur",
        "Glacier",
        "Décorateur",
        "Boulanger",
        "Chocolatier",
        "Fromager",
        "Boucher",
        "Aboyeur",
        "Communard",
        "Garçon de cuisine",
        "commis de débarrasseur"
    ]
    final_state = agent.run(sample_recipe, sample_food_question, cooks_list)
    sys.stdout.write("\n===== FINAL COOKED MEAL =====\n")
    sys.stdout.write(f"{final_state['cooked_meal']}\n")
    sys.stdout.write("=============================\n\n")
    sys.stdout.flush()
