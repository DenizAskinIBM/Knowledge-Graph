from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are an analytical problem solver. Keep chain-of-thoughts short."},
        {"role": "user", "content": "how are you?"}
    ],
    stream=True
)

chain_of_thought_printed = False
answer_printed = False

for chunk in response:
    delta = chunk.choices[0].delta
    
    # If there's chain-of-thought content
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not chain_of_thought_printed:
            print("CHAIN OF THOUGHT:")
            chain_of_thought_printed = True
        print(delta.reasoning_content, end="", flush=True)
        
    # If there's final content
    if hasattr(delta, "content") and delta.content is not None:
        # If first time seeing final content, print ANSWER heading
        if not answer_printed:
            # Print a newline to separate from chain-of-thought if needed
            if chain_of_thought_printed:
                print()
            print("ANSWER:")
            answer_printed = True
        print(delta.content, end="", flush=True)