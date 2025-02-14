from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key=os.getenv("NVIDIA_API_KEY")
)
# deepseek-chat, deepseek-ai/deepseek-r1
completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-r1",
  messages=[{"role": "system", "content": '''Answer the following question using only the chain-of-thought provided below. Do not generate or output any additional reasoning or internal chain-of-thought. Do not reveal any internal thoughts—only use the given chain-of-thought as a guide to compute the final answer, and then output only the final answer. Do not print anything between the <think> and </think> tokens
Chain of Thought:
<think>
1. 1 + 1 = 2
2. 1 + 1 + 1 = (1 + 1) + 1
3. 1 + 1 + 1 = 2 + 1
</think>'''},
    {"role":"user","content":"What is 1 + 1 + 1?"}],
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
  stream=True
)
for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")


