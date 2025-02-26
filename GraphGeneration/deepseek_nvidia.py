from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

from openai import OpenAI

client = OpenAI(
  base_url = "https://openrouter.ai/api/v1",
  api_key = os.getenv("NVIDIA_API_KEY")
)

completion = client.chat.completions.create(
  model="deepseek/deepseek-r1:free",
  messages=[{"role":"user","content":"Which number is larger, 9.11 or 9.8?"}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
  stream=True
)

chain_of_thought=""
for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
    try:
      chain_of_thought+=chunk.choices[0].delta.reasoning
    except:
      pass
print(chain_of_thought)


