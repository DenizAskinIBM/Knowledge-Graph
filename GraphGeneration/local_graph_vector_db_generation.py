import asyncio
from embedders import sentence_transformer_embedder, open_ai_text_3_large_embedder
from codebase import read, delete_all_indexes, print_index_names
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index, drop_index_if_exists
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

index_name = "local_mortgage_index_sentence_transformer"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Connect to the Neo4j database
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
embedding_model = open_ai_text_3_large_embedder
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

# Instantiate the SimpleKGPipeline
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedding_model,
    on_error="IGNORE",
    from_pdf=False,
)

transcripts=read("datasets/When_to_verify_the_identity_of_persons_and_entities—Financial_entities.md")
## Text to generate Knowledge Graphs from
mortgage_loan_transcript_1 = read("datasets/transcripts/mortgage_loan_1_transcript.txt")
mortgage_loan_transcript_2 = read("datasets/transcripts/mortgage_loan_2_transcript.txt")
mortgage_loan_transcript_3 = read("datasets/transcripts/mortgage_loan_3_transcript.txt")
transcripts=[mortgage_loan_transcript_1,mortgage_loan_transcript_2,mortgage_loan_transcript_3]
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
    similarity_fn="euclidean",
)
driver.close()
# drop_index_if_exists(driver, full_text_index_name)