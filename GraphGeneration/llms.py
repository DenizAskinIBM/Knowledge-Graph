import os
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv
load_dotenv("./.env")

url=os.getenv("WX_URL")
apikey=os.getenv("API_KEY")
project_id=os.getenv("PROJECT_ID")
openai_apikey=os.getenv("OPENAI_API_KEY")

model_id_llama="meta-llama/llama-3-405b-instruct"
model_id_mistral="mistralai/mixtral-8x7b-instruct-v01"
model_id_code="ibm/granite-34b-code-instruct"
model_id_granite="ibm/granite-3-8b-instruct"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
}

llm_code = ChatWatsonx(
    model_id=model_id_code,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

llm_mistral = ChatWatsonx(
    model_id=model_id_mistral,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

llm_granite = ChatWatsonx(
    model_id=model_id_granite,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

llm_llama = ChatWatsonx(
    model_id=model_id_llama,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=parameters
)

llm_chat_gpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_apikey
)

