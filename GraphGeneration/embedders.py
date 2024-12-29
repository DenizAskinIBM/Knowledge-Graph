from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# embedder = OpenAIEmbeddings(model="text-embedding-3-large")
sentence_transformer_embedder = SentenceTransformerEmbeddings(model="paraphrase-MiniLM-L6-v2")
open_ai_text_3_large_embedder = OpenAIEmbeddings(model="text-embedding-3-large")
