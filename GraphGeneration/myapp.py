import os
import tempfile
import asyncio
import logging
import traceback

from flask import Flask, request, redirect, url_for, render_template_string, session
from flask_session import Session
from werkzeug.utils import secure_filename

# --------------------------------------------------
# Local/Custom Imports
# --------------------------------------------------
from embedders import sentence_transformer_embedder
from llms import llm_chat_gpt, llm_llama, llm_granite
from codebase import hybrid_retrieve_answer, read, delete_all_indexes, print_index_names, context_retriever

from neo4j_graphrag.llm.openai_llm import OpenAILLM
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

from neo4j import GraphDatabase
from dotenv import load_dotenv

# --------------------------------------------------
# LangChain Imports (for Traditional RAG)
# --------------------------------------------------
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Reduce noisy logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------------------------------------------------
# 1) Load ENV Vars & Initialize Key Components
# --------------------------------------------------
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # if needed

# Index names for Graph ingestion
local_index_name = "local_financial_gpt_index"
global_index_name = "global_financial_gpt_index"

# Our chosen embedding model (from your custom "sentence_transformer_embedder")
embedding_model = sentence_transformer_embedder

# Instantiate LLMs
llm_openi_gpt = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    },
)
llm_for_queries = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

# Create the Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Prepare chunk splitter for Graph ingestion
text_splitter = FixedSizeSplitter(chunk_size=1000, chunk_overlap=50)

# --------------------------------------------------
# 2) Create or Reuse Graph Indexes On Startup
# --------------------------------------------------
try:
    create_fulltext_index(
        driver=driver,
        name=global_index_name,
        label="FinancialEntity",
        node_properties=[
            "vectorProperty"
        ],
        fail_if_exists=False,
    )

    dimensions = embedding_model.model.get_sentence_embedding_dimension()
    create_vector_index(
        driver,
        local_index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=dimensions,
        similarity_fn="cosine",
    )
except Exception as e:
    print("Index creation error:", e)

# --------------------------------------------------
# 3) Build a pipeline for Graph ingestion
# --------------------------------------------------
kg_builder = SimpleKGPipeline(
    llm=llm_openi_gpt,
    driver=driver,
    embedder=embedding_model,
    on_error="IGNORE",
    from_pdf=False,
    text_splitter=text_splitter
)

# --------------------------------------------------
# 4) LangChain-compatible Embeddings for Traditional RAG
# --------------------------------------------------
class MyLangChainEmbedder(Embeddings):
    """Wraps your sentence_transformer_embedder in a LangChain Embeddings interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [embedding_model.model.encode(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return embedding_model.model.encode(text).tolist()

my_langchain_embedder = MyLangChainEmbedder()

# --------------------------------------------------
# 5) Prepare a Traditional VectorStore (Chroma)
# --------------------------------------------------
traditional_vectorstore = Chroma(
    collection_name="traditional_rag_collection",
    embedding_function=my_langchain_embedder
)

# --------------------------------------------------
# 6) Flask Setup
# --------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

GLOBAL_CHAT_LOG = []
GLOBAL_LAST_CONTEXT = []
UPLOADED_FILES = []

# --------------------------------------------------
# 7) Front-End HTML
# --------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>IBM Agent - Click "Apply" to Confirm RAG</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400&display=swap');

      body, html {
          margin: 0;
          padding: 0;
          font-family: 'Open Sans', sans-serif;
          font-weight: 300;
          height: 100%%;
          background: #f2f4f8;
      }
      header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: #000;
          color: #fff;
          padding: 10px 20px;
          font-weight: 400;
      }
      header h1 {
          font-size: 1.2rem;
          margin: 0;
          font-weight: 400;
      }
      header .powered {
          font-size: 0.9rem;
          opacity: 0.7;
      }
      .container {
          display: flex;
          height: calc(100vh - 50px);
      }

      /* Left panel dynamic width: 75% if there is context, else 100% */
      .left-panel {
          background: #ffffff;
          padding: 20px;
          box-sizing: border-box;
          overflow-y: auto;
      }
      .left-panel.with-context {
          width: 75%%;
      }
      .left-panel.no-context {
          width: 100%%;
      }

      .left-panel h2 {
          margin-top: 0;
          font-weight: 400;
      }
      .left-panel ul {
          list-style: none;
          margin: 0;
          padding: 0;
      }
      .left-panel li {
          margin: 8px 0;
          display: flex;
          align-items: center;
      }
      .left-panel li::before {
          content: "•";
          color: #0f62fe;
          margin-right: 6px;
      }

      .chat-history {
          background: #fcfcfc;
          border: 1px solid #ccc;
          border-radius: 6px;
          flex: 1;
          margin-top: 20px;
          padding: 10px;
          min-height: 300px;
          overflow-y: auto;
      }
      .message {
          margin: 10px 0;
          line-height: 1.5;
      }
      .user-question {
          background: #eaf3ff;
          display: inline-block;
          padding: 8px;
          border-radius: 6px;
          margin-bottom: 5px;
          color: #0043ce;
          font-weight: 400;
      }
      .bot-answer {
          background: #ffffff;
          display: inline-block;
          padding: 8px;
          border-radius: 6px;
          color: #333;
          font-weight: 300;
      }

      .forms-container {
          margin-top: 20px;
          display: flex;
          align-items: center;
          gap: 10px;
          flex-wrap: wrap;
      }
      .rag-select-form {
          margin-bottom: 15px;
      }
      .rag-select-form label {
          margin-right: 8px;
          font-weight: 300;
      }
      .rag-select-form button {
          border-radius: 6px;
          padding: 8px 15px;
          background: #0f62fe;
          color: #fff;
          border: none;
          cursor: pointer;
          font-weight: 300;
      }
      .rag-select-form button:hover {
          opacity: 0.9;
      }

      form.qa-form {
          display: flex;
          align-items: center;
          margin: 0;
          font-weight: 300;
      }
      form.qa-form input[type="text"] {
          padding: 6px;
          width: 250px;
          margin-right: 10px;
          border-radius: 6px;
          border: 1px solid #ccc;
      }
      form.qa-form button {
          padding: 8px 15px;
          background: #0f62fe;
          color: #fff;
          border: none;
          cursor: pointer;
          margin-right: 8px;
          border-radius: 6px;
          font-weight: 300;
      }
      form.qa-form button:hover {
          opacity: 0.9;
      }

      form.upload-form {
          display: flex;
          align-items: center;
          margin: 0;
          font-weight: 300;
      }
      form.upload-form label {
          padding: 8px 12px;
          background: #393939;
          color: #fff;
          cursor: pointer;
          margin-right: 8px;
          border-radius: 6px;
          font-weight: 300;
      }
      form.upload-form label:hover {
          background: #4d4d4d;
      }
      form.upload-form button {
          padding: 8px 15px;
          background: #0f62fe;
          color: #fff;
          border: none;
          cursor: pointer;
          margin-right: 8px;
          border-radius: 6px;
          font-weight: 300;
      }
      form.upload-form button:hover {
          opacity: 0.9;
      }
      form.upload-form input[type="file"] {
          display: none;
      }

      .file-name-display {
          margin-right: 10px;
          font-size: 0.9rem;
          font-style: italic;
          color: #555;
      }
      .upload-success {
          margin-top: 10px;
          color: #0043ce;
          font-weight: 300;
      }

      .files-container {
          margin-top: 20px;
          background: #fefefe;
          border: 1px solid #ccc;
          border-radius: 6px;
          padding: 10px;
          max-height: 150px;
          overflow-y: auto;
          font-weight: 300;
      }
      .files-container h4 {
          margin: 0 0 6px 0;
      }
      .files-container ul {
          list-style: none;
          margin: 0;
          padding: 0;
      }
      .files-container li {
          margin: 6px 0;
          display: flex;
          justify-content: space-between;
          align-items: center;
      }
      .files-container button {
          background: #da1e28;
          color: #fff;
          border: none;
          padding: 5px 10px;
          cursor: pointer;
          font-size: 0.8rem;
          border-radius: 6px;
          font-weight: 300;
      }
      .files-container button:hover {
          opacity: 0.8;
      }

      .right-panel {
          width: 25%%;
          background: #fff;
          box-sizing: border-box;
          padding: 20px;
          display: flex;
          flex-direction: column;
          justify-content: flex-start;
      }
      .right-panel h2 {
          margin-top: 0;
          font-weight: 400;
      }
      .context-box {
          background: #f9f9f9;
          padding: 10px;
          border: 1px dashed #aaa;
          border-radius: 6px;
          height: calc(100%% - 40px);
          overflow-y: auto;
      }
      .context-snippet {
          margin-bottom: 20px;
          line-height: 1.4;
          font-weight: 300;
      }
      .context-snippet a {
          color: #0f62fe;
          text-decoration: underline;
          cursor: pointer;
          margin-left: 6px;
          font-size: 0.85rem;
          font-weight: 300;
      }

      ::-webkit-scrollbar {
          width: 8px;
      }
      ::-webkit-scrollbar-thumb {
          background: #ccc;
          border-radius: 6px;
      }
    </style>
</head>
<body>
<header>
    <h1>IBM Agent</h1>
    <div class="powered">Powered by watsonx</div>
</header>

<div class="container">
    <!-- Left Panel: dynamically choose class based on whether we have context -->
    <div class="left-panel {% if last_context %}with-context{% else %}no-context{% endif %}">

        <form method="POST" action="{{ url_for('set_rag_type') }}" class="rag-select-form">
            <label>Select RAG Type:</label>
            <label>
              <input type="radio" name="rag_type" value="traditional"
                {% if current_rag_type == 'traditional' %} checked {% endif %}>
              Traditional
            </label>
            <label>
              <input type="radio" name="rag_type" value="graph"
                {% if current_rag_type == 'graph' %} checked {% endif %}>
              Graph
            </label>
            <button type="submit">Apply</button>
        </form>

        <div class="chat-history">
            {% if chat_history %}
                {% for entry in chat_history %}
                    <div class="message">
                        <div class="user-question">You: {{ entry.question }}</div><br>
                        <div class="bot-answer">Answer: {{ entry.answer }}</div>
                    </div>
                {% endfor %}
            {% else %}
                <p>No conversation yet.</p>
            {% endif %}
        </div>

        {% if upload_message %}
        <div class="upload-success">{{ upload_message }}</div>
        {% endif %}

        <div class="forms-container">
            <!-- Q&A Form -->
            <form method="POST" action="{{ url_for('ask_question') }}" class="qa-form">
                <input type="text" name="user_question" placeholder="Ask a question..." required />
                <button type="submit">Send</button>
            </form>

            <!-- Upload Form -->
            <form method="POST" action="{{ url_for('upload_file') }}" enctype="multipart/form-data" class="upload-form">
                <label for="fileInput">Attach</label>
                <span class="file-name-display" id="fileNameDisplay"></span>
                <input type="file" name="uploaded_file" id="fileInput" accept=".txt,.pdf,.doc,.docx,.md" required />
                <button type="submit">Upload</button>
            </form>

            <!-- Show/Hide Files -->
            <form method="POST" action="{{ url_for('toggle_files') }}">
                <button type="submit">
                    {% if show_files %}Hide{% else %}Show{% endif %} Files
                </button>
            </form>

            <!-- Clear Chat -->
            <form method="POST" action="{{ url_for('clear_chat') }}">
                <button type="submit">Clear Chat</button>
            </form>
        </div>

        {% if show_files and uploaded_files %}
        <div class="files-container">
            <h4>Uploaded Files</h4>
            <ul>
                {% for f in uploaded_files %}
                <li>
                    <span>{{ f.filename }}</span>
                    <form method="POST" action="{{ url_for('delete_file', filename=f.filename) }}">
                        <button type="submit">Delete</button>
                    </form>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>

    <!-- Right Panel: only show if we have retrieved context -->
    {% if last_context %}
    <div class="right-panel">
        <h2>Retrieved Content</h2>
        <div class="context-box">
            {% for ctx in last_context %}
            <div class="context-snippet">
                <span id="truncated-{{ loop.index }}">
                    {{ ctx.snippet }}
                    {% if ctx.truncated %}
                        <a onclick="toggleSnippet({{ loop.index }}, true)">[Show more]</a>
                    {% endif %}
                </span>
                <span id="full-{{ loop.index }}" style="display:none;">
                    {{ ctx.full|safe }}
                    <a onclick="toggleSnippet({{ loop.index }}, false)">[Show less]</a>
                </span>
            </div>
            <hr>
            {% endfor %}
        </div>
    </div>
    {% endif %}
</div>

<script>
  const fileInput = document.getElementById('fileInput');
  const fileNameDisplay = document.getElementById('fileNameDisplay');
  if(fileInput) {
    fileInput.addEventListener('change', function() {
      if (this.files && this.files.length > 0) {
        fileNameDisplay.textContent = this.files[0].name;
      } else {
        fileNameDisplay.textContent = '';
      }
    });
  }

  function toggleSnippet(index, showFull) {
    const truncatedSpan = document.getElementById('truncated-' + index);
    const fullSpan = document.getElementById('full-' + index);
    if (showFull) {
      truncatedSpan.style.display = 'none';
      fullSpan.style.display = 'inline';
    } else {
      fullSpan.style.display = 'none';
      truncatedSpan.style.display = 'inline';
    }
  }
</script>
</body>
</html>
"""

# --------------------------------------------------
# 8) Helpers for ingestion
# --------------------------------------------------
from langchain.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader
)

#####################################################
# Make sure to define the load_and_chunk_file() function!
#####################################################
def load_and_chunk_file(file_path: str):
    """
    Loads a file from `file_path` as Documents.
    We'll feed these Documents into the pipeline for Graph ingestion
    AND chunk them for the Traditional RAG vector store.
    """
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        loader = UnstructuredPDFLoader(file_path)
        raw_docs = loader.load()
    elif ext.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        raw_docs = loader.load()
    elif ext.endswith(".doc"):
        loader = UnstructuredWordDocumentLoader(file_path)
        raw_docs = loader.load()
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        raw_docs = loader.load()

    return raw_docs

#####################################################
# Accelerated ingestion with concurrency
#####################################################
MAX_CONCURRENT_CHUNKS = 8
_sem = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

async def _ingest_chunk(chunk_text, pipeline):
    async with _sem:
        return await pipeline.run_async(text=chunk_text)

async def ingest_all_chunks_async(docs, pipeline):
    tasks = []
    for i, doc_chunk in enumerate(docs, start=1):
        print(f"[ASYNC] Queuing doc {i}/{len(docs)} for ingestion...")
        tasks.append(asyncio.create_task(_ingest_chunk(doc_chunk.page_content, pipeline)))
    results = await asyncio.gather(*tasks)
    return results

def chunk_for_traditional_rag(docs):
    """
    Convert the loaded Documents into smaller chunks
    using a standard LangChain splitter so we can store them
    into the local Chroma vector store.
    """
    chunked_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    for doc in docs:
        splitted = splitter.split_text(doc.page_content)
        for chunk in splitted:
            new_doc = Document(page_content=chunk, metadata=doc.metadata)
            chunked_docs.append(new_doc)

    return chunked_docs

# --------------------------------------------------
# 9) Flask Routes
# --------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    upload_msg = session.pop("upload_message", "")
    show_files = session.get("show_files", False)
    current_rag_type = session.get("rag_type", "graph")  # default to Graph

    return render_template_string(
        HTML_TEMPLATE,
        chat_history=GLOBAL_CHAT_LOG,
        last_context=GLOBAL_LAST_CONTEXT,
        upload_message=upload_msg,
        show_files=show_files,
        uploaded_files=UPLOADED_FILES,
        current_rag_type=current_rag_type
    )

@app.route("/set_rag_type", methods=["POST"])
def set_rag_type():
    chosen = request.form.get("rag_type", "graph")
    session["rag_type"] = chosen
    print(f"User set RAG type to: {chosen}")
    return redirect(url_for("index"))

@app.route("/", methods=["POST"])
def ask_question():
    user_question = request.form.get("user_question", "")
    rag_type = session.get("rag_type", "graph")
    top_k = 10

    print(f"RAG type at ask: {rag_type}")
    print(f"User question: {user_question}")

    if rag_type == "graph":
        # -- Graph RAG --
        try:
            retriever = HybridRetriever(
                driver=driver,
                vector_index_name=local_index_name,
                fulltext_index_name=global_index_name,
                embedder=embedding_model
            )
            rag = GraphRAG(retriever=retriever, llm=llm_for_queries)
            results = rag.search(
                query_text=user_question,
                retriever_config={"top_k": top_k},
                return_context=True
            )
            answer = results.answer

            GLOBAL_CHAT_LOG.append({"question": user_question, "answer": answer})
            GLOBAL_LAST_CONTEXT.clear()

            if results.retriever_result:
                for ctx_chunk in results.retriever_result.items:
                    full_text = str(ctx_chunk.content)
                    snippet = full_text[:300]
                    truncated_flag = (len(full_text) > 300)
                    if truncated_flag:
                        snippet += "...(truncated)"
                    GLOBAL_LAST_CONTEXT.append({
                        "snippet": snippet,
                        "full": full_text,
                        "truncated": truncated_flag
                    })

        except Exception as e:
            error_msg = f"Graph RAG error: {str(e)}"
            print(traceback.format_exc())
            GLOBAL_CHAT_LOG.append({"question": user_question, "answer": error_msg})
            GLOBAL_LAST_CONTEXT.clear()

    else:
        # -- Traditional RAG --
        try:
            retrieved_docs = traditional_vectorstore.similarity_search(user_question, k=top_k)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            prompt = (
                f"Use the following context to help answer the question:\n\n"
                f"{context_text}\n\n"
                f"Question: {user_question}\n\n"
                f"Answer concisely:"
            )
            answer = llm_for_queries.invoke(prompt).content

            GLOBAL_CHAT_LOG.append({"question": user_question, "answer": answer})
            GLOBAL_LAST_CONTEXT.clear()

            for doc in retrieved_docs:
                snippet = doc.page_content[:300]
                truncated_flag = (len(doc.page_content) > 300)
                if truncated_flag:
                    snippet += "...(truncated)"
                GLOBAL_LAST_CONTEXT.append({
                    "snippet": snippet,
                    "full": doc.page_content,
                    "truncated": truncated_flag
                })

        except Exception as e:
            error_msg = f"Traditional RAG error: {str(e)}"
            print(traceback.format_exc())
            GLOBAL_CHAT_LOG.append({"question": user_question, "answer": error_msg})
            GLOBAL_LAST_CONTEXT.clear()

    return redirect(url_for("index"))

@app.route("/upload_file", methods=["POST"])
def upload_file():
    """
    On file upload:
      - Load doc(s)
      - Ingest to Graph via SimpleKGPipeline (async)
      - Chunk & add to Chroma for Traditional RAG
    """
    uploaded_file = request.files.get("uploaded_file", None)
    if not uploaded_file or uploaded_file.filename == "":
        print("No file selected or empty filename.")
        return redirect(url_for("index"))

    filename = secure_filename(uploaded_file.filename)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    uploaded_file.save(temp_path)
    print(f"Uploaded file saved at: {temp_path}")

    try:
        # 1) Load raw docs
        docs = load_and_chunk_file(temp_path)

        # 2) Graph ingestion (async, concurrency for speed)
        asyncio.run(ingest_all_chunks_async(docs, kg_builder))

        # 3) Traditional RAG ingestion
        chunked_docs = chunk_for_traditional_rag(docs)
        traditional_vectorstore.add_documents(chunked_docs)

        session["upload_message"] = f"File '{filename}' uploaded & ingested (Graph + Traditional)."
        print(f"Ingested {len(docs)} doc(s) for Graph, plus {len(chunked_docs)} chunks for Traditional RAG.")
    except Exception as e:
        full_tb = traceback.format_exc()
        print("Error during ingestion:", full_tb)
        session["upload_message"] = f"Ingestion error for {filename}: {str(e)}"

    UPLOADED_FILES.append({"filename": filename, "path": temp_path})
    return redirect(url_for("index"))

@app.route("/toggle_files", methods=["POST"])
def toggle_files():
    current = session.get("show_files", False)
    session["show_files"] = not current
    return redirect(url_for("index"))

@app.route("/delete_file/<filename>", methods=["POST"])
def delete_file(filename):
    global UPLOADED_FILES
    new_files = []
    for f in UPLOADED_FILES:
        if f["filename"] == filename:
            if os.path.exists(f["path"]):
                os.remove(f["path"])
            print(f"Deleted file from local disk: {f['path']}")
        else:
            new_files.append(f)
    UPLOADED_FILES = new_files
    session["upload_message"] = f"Deleted {filename} from uploaded list."
    return redirect(url_for("index"))

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    GLOBAL_CHAT_LOG.clear()
    GLOBAL_LAST_CONTEXT.clear()
    session["upload_message"] = "Chat cleared."
    return redirect(url_for("index"))

@app.route("/shutdown", methods=["GET"])
def shutdown():
    if driver:
        driver.close()
    return "Neo4j driver closed, server shutting down."

# --------------------------------------------------
# 10) Main
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)