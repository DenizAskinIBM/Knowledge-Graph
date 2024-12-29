import asyncio
from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder
from codebase import read, delete_all_indexes, print_index_names
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index, drop_index_if_exists
from Custom_LLM import CustomLLM
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

index_name = ""

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
embedding_model = sentence_transformer_embedder
transformer_model = embedding_model.model

# Uncomment to print and/or delete all indexes
# print_index_names(driver)
# delete_all_indexes(driver)

# Instantiate the LLM
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)

# Instantiate your custom LLM
custom_llm = CustomLLM(model_name="meta-llama/llama-3-405b-instruct")

# Instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=custom_llm,
    driver=driver,
    embedder=embedding_model,
    on_error="IGNORE",
    from_pdf=False,
)

## Text to generate Knowledge Graphs from
mortgage_loan_transcript_1 = read("datasets/transcripts/mortgage_loan_1_transcript.txt")
mortgage_loan_transcript_2 = read("datasets/transcripts/mortgage_loan_2_transcript.txt")
mortgage_loan_transcript_3 = read("datasets/transcripts/mortgage_loan_3_transcript.txt")
transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]
transcripts=read("datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md")

if(isinstance(transcripts, list)):
    for x in transcripts:
        # Run the pipeline on a piece of text
        text = (
            x.strip()
        )
        asyncio.run(kg_builder.run_async(text=text))
else:
   # Run the pipeline on a piece of text
    text = (
        transcripts
    )
    asyncio.run(kg_builder.run_async(text=text)) 

if(embedding_model == sentence_transformer_embedder):
    dimensions=transformer_model.get_sentence_embedding_dimension()
else:
    sample_embedding = open_ai_text_3_large_embedder.embed_query("sample text")
    dimensions = len(sample_embedding)
# Create the index
create_vector_index(
    driver,
    index_name,
    label="Chunk",
    embedding_property="embedding",
    dimensions=dimensions,
    similarity_fn="cosine",
)
driver.close()
# drop_index_if_exists(driver, full_text_index_name)