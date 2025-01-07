from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
open_ai_text_ada_002_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
sentence_transformer_embedder = SentenceTransformerEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
open_ai_text_3_large_embedder = OpenAIEmbeddings(model="text-embedding-3-large")
