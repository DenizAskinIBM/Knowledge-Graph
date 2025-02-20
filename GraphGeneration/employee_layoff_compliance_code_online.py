import os
import json
import PyPDF2
import concurrent.futures
from flask import Flask, render_template_string, request, redirect, url_for, Response
from flask_session import Session
from typing_extensions import TypedDict

# Watsonx LLM
from langchain.schema import HumanMessage
from langchain_ibm import ChatWatsonx
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------
# 1) Flask Setup
# ---------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ---------------------------------------------------------------
# 2) Watsonx LLM Setup (optimized params)
# ---------------------------------------------------------------
WATSONX_URL = os.getenv("WATSONX_URL", "")
API_KEY = os.getenv("API_KEY", "")
PROJECT_ID = os.getenv("PROJECT_ID", "")

model_id = "meta-llama/llama-3-1-70b-instruct"
parameters = {
    "decoding_method": "greedy",
    # Lower max tokens to speed up
    "max_new_tokens": 256,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42
}

llm_llama = ChatWatsonx(
    model_id=model_id,
    url=WATSONX_URL,
    apikey=API_KEY,
    project_id=PROJECT_ID,
    params=parameters
)

# ---------------------------------------------------------------
# 3) Global State
# ---------------------------------------------------------------
GLOBAL_CHAT_LOG = []         # store Q&A pairs (and subtask results)
UPLOADED_FILES = []          # list of uploaded files
UPLOADED_PDF_TEXT = ""       # text extracted from last uploaded PDF
SHOW_FILES = False
UPLOAD_MESSAGE = ""
GLOBAL_LOG = []              # minimal logs for UI
LLM_SUBTASK_EXECUTIONS = []  # store LLM subtask (prompt/answer) for display

# We'll limit how much PDF text we pass to the LLM (to speed up prompts)
MAX_PDF_CHARS = 3000

def maybe_shorten_text(text: str, limit: int = MAX_PDF_CHARS) -> str:
    """Shorten text if it's too long, to reduce prompt size."""
    if len(text) > limit:
        return text[:limit] + f"\n... (truncated {len(text)-limit} chars) ..."
    return text

def log(msg: str):
    """Append a single log line to the global log."""
    GLOBAL_LOG.append(str(msg))

# ---------------------------------------------------------------
# 4) Pipeline Code (Plan -> Execute -> Final)
# ---------------------------------------------------------------
class AgentWorkflowState(TypedDict):
    pdf_text: str
    user_question: str
    planner_output: str
    execution_result: dict
    final_answer: str

def stream_llm_call(prompt: str) -> str:
    """Fetch entire LLM response at once (non-streaming version)."""
    response_msg = llm_llama.invoke([HumanMessage(content=prompt)])
    if hasattr(response_msg, "content"):
        return response_msg.content
    else:
        return response_msg[0].content

# Node 1: Planner
def compliance_planner_node(state: AgentWorkflowState) -> AgentWorkflowState:
    pdf_text = maybe_shorten_text(state["pdf_text"])
    user_question = state["user_question"]
    prompt = f"""
You are the Compliance Planner Agent with the following PDF content:

{pdf_text}

The user asked: '{user_question}'.

Your job:
1) Figure out all the subtasks needed to answer the user's question in full detail.
2) **Never assume** any unknown user or employee data (e.g., how long they've worked, pay rates, etc.).
   If any piece is missing, you must create a USER subtask to explicitly ask for it.
3) Output the subtasks in valid JSON of the form:
   {{
     "subtasks": [
       {{"task": "Describe subtask here", "type": "<LLM or USER>"}}
       ...
     ]
   }}

Return ONLY valid JSON. No extra commentary.
"""
    log("[Planner] Generating plan...")
    planner_response = stream_llm_call(prompt)
    log(f"[Planner] Plan JSON:\n{planner_response}")
    state["planner_output"] = planner_response
    return state

# Node 2: Executor sets up subtasks (but does not yet run LLM tasks)
def executer_node(state: AgentWorkflowState) -> AgentWorkflowState:
    planner_json = state["planner_output"]
    try:
        plan = json.loads(planner_json)
        subtasks = plan["subtasks"]
        subtask_answers = [None] * len(subtasks)
        log(f"[Executer] Found {len(subtasks)} subtasks.")
        state["execution_result"] = {
            "subtasks": subtasks,
            "subtask_answers": subtask_answers
        }
    except Exception as e:
        log(f"[Executer] Invalid planner JSON: {e}")
        state["execution_result"] = {
            "error": f"Invalid JSON: {e}",
            "subtasks": [],
            "subtask_answers": []
        }
    return state

# LLM subtask concurrency (non-streamed; these results are appended to chat log)
def execute_llm_subtasks(state: AgentWorkflowState, user_answers: dict) -> AgentWorkflowState:
    global LLM_SUBTASK_EXECUTIONS
    LLM_SUBTASK_EXECUTIONS = []  # clear previous runs

    pdf_text = maybe_shorten_text(state["pdf_text"])
    subtasks = state["execution_result"].get("subtasks", [])
    subtask_answers = state["execution_result"].get("subtask_answers", [])

    # Fill in USER subtask answers
    for idx, st in enumerate(subtasks):
        if st.get("type", "").upper() == "USER":
            ans = user_answers.get(idx, "(No user answer provided)")
            subtask_answers[idx] = {"subtask": st["task"], "answer": ans}
            log(f"[Executer] Subtask #{idx+1} (USER) => {ans}")

    # Identify LLM subtasks
    llm_indices = [i for i, st in enumerate(subtasks) if st.get("type", "").upper() == "LLM"]
    if not llm_indices:
        log("[Executer] No LLM subtasks found.")
        return state

    log(f"[Executer] Running LLM subtasks concurrently: {llm_indices}")

    def run_llm_subtask(idx: int):
        # Build context from already answered subtasks
        context_lines = []
        for i, ans in enumerate(subtask_answers):
            if ans is not None:
                context_lines.append(f"Subtask #{i+1}:\nTask: {ans['subtask']}\nAnswer: {ans['answer']}\n")
        context_str = "\n".join(context_lines)
        subtask_prompt = subtasks[idx]["task"]
        prompt_llm = f"""
You are the Executer Agent. You have the following PDF content:
{pdf_text}

Below are the answers to previously completed subtasks:
{context_str}

Now, please address this subtask:
{subtask_prompt}

Return only your solution or explanation for this subtask.
"""
        log(f"[Executer] Subtask #{idx+1} (LLM) starting...")
        try:
            response = stream_llm_call(prompt_llm)
            log(f"[Executer] Subtask #{idx+1} (LLM) done.")
            return (subtask_prompt, response)
        except Exception as e:
            return (subtask_prompt, f"Error: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {idx: executor.submit(run_llm_subtask, idx) for idx in llm_indices}
        for idx, fut in future_map.items():
            subtask_prompt, result = fut.result()
            subtask_answers[idx] = {"subtask": subtask_prompt, "answer": result}
            LLM_SUBTASK_EXECUTIONS.append({
                "index": idx,
                "task": subtask_prompt,
                "answer": result
            })
            # Also, add each executed LLM subtask to the main chat log
            GLOBAL_CHAT_LOG.append({
                "question": f"[Subtask {idx+1} - LLM] {subtask_prompt}",
                "answer": result
            })
    return state

# Node 3: Final Answer (non-streamed version; used in streaming mode below)
def question_answerer_node(state: AgentWorkflowState) -> AgentWorkflowState:
    pdf_text = maybe_shorten_text(state["pdf_text"])
    user_question = state["user_question"]
    subtask_answers = state["execution_result"].get("subtask_answers", [])

    subtask_txt = ""
    for i, ans in enumerate(subtask_answers):
        if ans:
            subtask_txt += f"Subtask #{i+1}:\nTask: {ans['subtask']}\nAnswer: {ans['answer']}\n\n"

    prompt = f"""
You are the Question Answerer Agent.

PDF Content:
{pdf_text}

User's Question:
{user_question}

Subtask Tasks and Answers:
{subtask_txt}

Provide a concise final answer. Do not ask for more input now.
"""
    log("[QAnswerer] Generating final answer...")
    final_ans = stream_llm_call(prompt)
    log("[QAnswerer] Done.")
    state["final_answer"] = final_ans
    return state

def run_workflow_partial_plan(pdf_text: str, user_question: str) -> AgentWorkflowState:
    """Plan + partial executer to see if we have user subtasks."""
    st: AgentWorkflowState = {
        "pdf_text": pdf_text,
        "user_question": user_question,
        "planner_output": "",
        "execution_result": {},
        "final_answer": ""
    }
    st = compliance_planner_node(st)
    st = executer_node(st)
    return st

def run_workflow_llm_and_final(state: AgentWorkflowState, user_subtask_answers: dict) -> AgentWorkflowState:
    """Execute LLM subtasks, then final node (synchronously)."""
    state = execute_llm_subtasks(state, user_subtask_answers)
    state = question_answerer_node(state)
    return state

# --- STREAMING FINAL ANSWER GENERATOR ---
def stream_final_answer_generator(state: AgentWorkflowState):
    """
    This generator builds the final answer prompt (as in question_answerer_node)
    and then simulates token-by-token streaming by splitting the final answer.
    """
    pdf_text = maybe_shorten_text(state["pdf_text"])
    user_question = state["user_question"]
    subtask_answers = state["execution_result"].get("subtask_answers", [])
    subtask_txt = ""
    for i, ans in enumerate(subtask_answers):
        if ans:
            subtask_txt += f"Subtask #{i+1}:\nTask: {ans['subtask']}\nAnswer: {ans['answer']}\n\n"
    prompt = f"""
You are the Question Answerer Agent.

PDF Content:
{pdf_text}

User's Question:
{user_question}

Subtask Tasks and Answers:
{subtask_txt}

Provide a concise final answer. Do not ask for more input now.
"""
    log("[QAnswerer] Generating final answer (streaming)...")
    final_ans = llm_llama.invoke([HumanMessage(content=prompt)]).content
    # Simulate token-by-token streaming (here we split by whitespace)
    tokens = final_ans.split()
    for token in tokens:
        yield token + " "
    log("[QAnswerer] Done streaming final answer.")

# ---------------------------------------------------------------
# 5) HTML Template (original layout with an added streaming block)
# ---------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Plan & Execute Pipeline (Optimized, Subtasks Display)</title>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400&display=swap');
      body, html {
          margin: 0; padding: 0;
          font-family: 'Open Sans', sans-serif; font-weight: 300;
          height: 100%%; background: #f2f4f8;
      }
      header {
          display: flex; justify-content: space-between;
          align-items: center; background: #000; color: #fff;
          padding: 10px 20px; font-weight: 400;
      }
      header h1 { font-size:1.2rem; margin:0; font-weight:400; }
      header .powered { font-size:0.9rem; opacity:0.7; }
      .container { display:flex; height:calc(100vh - 50px); }
      .left-panel {
          background:#fff; padding:20px; box-sizing:border-box;
          overflow-y:auto; width:70%%;
      }
      .right-panel {
          background:#fff; padding:20px; box-sizing:border-box;
          overflow-y:auto; width:30%%;
          border-left: 1px solid #ccc;
      }
      .chat-history {
          background:#fcfcfc; border:1px solid #ccc; border-radius:6px;
          flex:1; margin-top:20px; padding:10px; min-height:300px; overflow-y:auto;
      }
      .message { margin:10px 0; line-height:1.5; }
      .user-question {
          background:#eaf3ff; display:inline-block; padding:8px;
          border-radius:6px; margin-bottom:5px; color:#0043ce; font-weight:400;
      }
      .bot-answer {
          background:#fff; display:inline-block; padding:8px;
          border-radius:6px; color:#333; font-weight:300;
      }
      .forms-container {
          margin-top:20px; display:flex;
          align-items:center; gap:10px; flex-wrap:wrap;
      }
      form.qa-form { display:flex; align-items:center; margin:0; font-weight:300; }
      form.qa-form input[type="text"] {
          padding:6px; width:250px; margin-right:10px;
          border-radius:6px; border:1px solid #ccc;
      }
      form.qa-form button {
          padding:8px 15px; background:#0f62fe; color:#fff;
          border:none; cursor:pointer; margin-right:8px; border-radius:6px;
          font-weight:300;
      }
      form.qa-form button:hover { opacity:0.9; }
      form.upload-form { display:flex; align-items:center; margin:0; font-weight:300; }
      form.upload-form label {
          padding:8px 12px; background:#393939; color:#fff; cursor:pointer;
          margin-right:8px; border-radius:6px; font-weight:300;
      }
      form.upload-form label:hover { background:#4d4d4d; }
      form.upload-form button {
          padding:8px 15px; background:#0f62fe; color:#fff;
          border:none; cursor:pointer; margin-right:8px; border-radius:6px;
          font-weight:300;
      }
      form.upload-form button:hover { opacity:0.9; }
      form.upload-form input[type="file"] { display:none; }
      .file-name-display {
          margin-right:10px; font-size:0.9rem;
          font-style:italic; color:#555;
      }
      .upload-success { margin-top:10px; color:#0043ce; font-weight:300; }
      .files-container {
          margin-top:20px; background:#fefefe; border:1px solid #ccc;
          border-radius:6px; padding:10px; max-height:150px; overflow-y:auto; font-weight:300;
      }
      .files-container ul { list-style:none; margin:0; padding:0; }
      .files-container li {
          margin:6px 0; display:flex;
          justify-content:space-between; align-items:center;
      }
      .files-container button {
          background:#da1e28; color:#fff; border:none;
          padding:5px 10px; cursor:pointer; font-size:0.8rem;
          border-radius:6px; font-weight:300;
      }
      .files-container button:hover { opacity:0.8; }
      .user-subtasks {
          margin-top:20px; background:#ffe; border:1px solid #ccc;
          padding:10px;
      }
      .user-subtasks h3 {
          margin-top:0; font-weight:400; margin-bottom:10px;
      }
      .executed-llm-subtasks {
          margin-top:20px; background:#edf0ff; border:1px solid #ccc;
          padding:10px;
      }
      .executed-llm-subtasks h3 {
          margin-top:0; font-weight:400; margin-bottom:10px;
      }
      .executed-llm-subtasks ul {
          list-style:none; margin:0; padding:0;
      }
      .executed-llm-subtasks li {
          margin-bottom:15px; line-height:1.4;
      }
      .console-log {
          background:#fafafa; border:1px solid #ccc;
          border-radius:6px; padding:10px; height:100%%; overflow-y:auto;
          font-size:0.9rem;
      }
      .console-log h3 { margin-top:0; font-weight:400; }
      .console-log .log-line { margin-bottom:5px; }
      /* Streaming answer block styling */
      #streaming-answer {
          margin-top:20px; padding:10px; background:#eef;
          border:1px solid #ccc; border-radius:6px;
      }
      ::-webkit-scrollbar { width:8px; }
      ::-webkit-scrollbar-thumb { background:#ccc; border-radius:6px; }
    </style>
</head>
<body>
<header>
    <h1>Plan & Execute Pipeline (Optimized, Subtasks Display)</h1>
    <div class="powered">Powered by Payment Evolution</div>
</header>

<div class="container">
    <div class="left-panel">
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
            <form method="POST" action="{{ url_for('ask_question') }}" class="qa-form">
                <input type="text" name="user_question" placeholder="Ask a question..." required />
                <button type="submit">Send</button>
            </form>

            <form method="POST" action="{{ url_for('upload_file') }}" enctype="multipart/form-data" class="upload-form">
                <label for="fileInput">Attach PDF</label>
                <span class="file-name-display" id="fileNameDisplay"></span>
                <input type="file" name="uploaded_file" id="fileInput" accept=".pdf" required />
                <button type="submit">Upload</button>
            </form>

            <form method="POST" action="{{ url_for('toggle_files') }}">
                <button type="submit">
                    {% if show_files %}Hide{% else %}Show{% endif %} Files
                </button>
            </form>

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

        {% if user_subtasks %}
          <div class="user-subtasks">
            <h3>Provide answers for these USER subtasks:</h3>
            <form method="POST" action="{{ url_for('answer_user_subtasks') }}">
              {% for idx, prompt in user_subtasks %}
                <div style="margin-bottom:10px;">
                  <label><strong>Subtask #{{ idx+1 }}:</strong> {{ prompt }}</label><br>
                  <input type="text" name="answer_{{idx}}" style="width:80%%;" required />
                </div>
              {% endfor %}
              <button type="submit">Submit</button>
            </form>
          </div>
        {% endif %}

        {% if llm_executions %}
          <div class="executed-llm-subtasks">
            <h3>Executed LLM Subtasks</h3>
            <ul>
              {% for item in llm_executions %}
              <li>
                <strong>Subtask #{{ item.index+1 }} (LLM)</strong><br>
                Task: {{ item.task }}<br>
                Answer: {{ item.answer }}
              </li>
              {% endfor %}
            </ul>
          </div>
        {% endif %}

        {# Streaming block: if streaming==True, open an SSE connection to stream final answer #}
        {% if streaming and streaming_question %}
          <div id="streaming-answer"></div>
          <script>
            var evtSource = new EventSource("/stream_final_answer?question={{ streaming_question }}");
            var streamingDiv = document.getElementById("streaming-answer");
            evtSource.onmessage = function(e) {
              if(e.data === "[DONE]") {
                evtSource.close();
                return;
              }
              streamingDiv.textContent += e.data;
            };
          </script>
        {% endif %}
    </div>

    <div class="right-panel">
        <div class="console-log">
          <h3>Log Output</h3>
          {% if global_log %}
            {% for line in global_log %}
              <div class="log-line">{{ line }}</div>
            {% endfor %}
          {% else %}
            <p>No logs yet.</p>
          {% endif %}
        </div>
    </div>
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
</script>
</body>
</html>
"""

# ---------------------------------------------------------------
# 6) Flask Routes
# ---------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Main page."""
    return render_template_string(
        HTML_TEMPLATE,
        chat_history=GLOBAL_CHAT_LOG,
        show_files=SHOW_FILES,
        uploaded_files=UPLOADED_FILES,
        upload_message=UPLOAD_MESSAGE,
        global_log=GLOBAL_LOG,
        user_subtasks=None,
        streaming=False,
        streaming_question="",
        llm_executions=LLM_SUBTASK_EXECUTIONS
    )

@app.route("/ask_question", methods=["POST"])
def ask_question():
    """
    When the user asks a question:
      1. Run planning and partial execution.
      2. If USER subtasks are required, show them.
      3. Otherwise, redirect to a streaming page so the final answer is streamed.
    """
    GLOBAL_LOG.clear()  # Clear logs for new question
    user_q = request.form.get("user_question", "")
    log(f"== New question: {user_q} ==")

    # 1) Plan + partial executer
    state = run_workflow_partial_plan(UPLOADED_PDF_TEXT, user_q)
    # Gather user subtasks (if any)
    user_subtask_prompts = []
    subtasks = state["execution_result"].get("subtasks", [])
    for idx, st in enumerate(subtasks):
        if st.get("type", "").upper() == "USER":
            user_subtask_prompts.append((idx, st["task"]))

    app.config["PARTIAL_STATE"] = state

    if user_subtask_prompts:
        # Display user subtasks for manual input
        return render_template_string(
            HTML_TEMPLATE,
            chat_history=GLOBAL_CHAT_LOG,
            show_files=SHOW_FILES,
            uploaded_files=UPLOADED_FILES,
            upload_message=UPLOAD_MESSAGE,
            global_log=GLOBAL_LOG,
            user_subtasks=user_subtask_prompts,
            streaming=False,
            streaming_question="",
            llm_executions=LLM_SUBTASK_EXECUTIONS
        )
    else:
        # No USER subtasks required: stream the final answer.
        return redirect(url_for("streaming", question=user_q))

@app.route("/answer_user_subtasks", methods=["POST"])
def answer_user_subtasks():
    """
    Called when user provides answers for all USER subtasks.
    Then run LLM subtasks and generate the final answer synchronously.
    """
    state = app.config.get("PARTIAL_STATE", None)
    if not state:
        log("ERROR: No partial pipeline state found.")
        return redirect(url_for("index"))

    user_subtask_answers = {}
    subtasks = state["execution_result"]["subtasks"]
    for idx, st in enumerate(subtasks):
        if st.get("type", "").upper() == "USER":
            val = request.form.get(f"answer_{idx}", "")
            user_subtask_answers[idx] = val

    updated = run_workflow_llm_and_final(state, user_subtask_answers)
    final_ans = updated["final_answer"]
    user_q = updated["user_question"]

    GLOBAL_CHAT_LOG.append({"question": user_q, "answer": final_ans})
    app.config["PARTIAL_STATE"] = None
    return redirect(url_for("index"))

@app.route("/upload_file", methods=["POST"])
def upload_file():
    global UPLOADED_FILES, UPLOAD_MESSAGE, UPLOADED_PDF_TEXT
    uploaded_file = request.files.get("uploaded_file", None)
    if uploaded_file and uploaded_file.filename:
        filename = uploaded_file.filename
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pages_text = []
            for page in pdf_reader.pages:
                pages_text.append(page.extract_text())
            pdf_text = "\n".join(pages_text)
            UPLOADED_PDF_TEXT = pdf_text
            UPLOADED_FILES.append({"filename": filename})
            UPLOAD_MESSAGE = f"'{filename}' uploaded. Extracted text length: {len(pdf_text)}."
            log(f"[UPLOAD] Extracted {len(pdf_text)} chars from '{filename}'.")
        except Exception as e:
            UPLOAD_MESSAGE = f"Error reading PDF: {e}"
            log(f"[UPLOAD] ERROR reading PDF {filename}: {e}")
    else:
        UPLOAD_MESSAGE = "No file selected or empty filename."
        log("[UPLOAD] No valid file selected.")
    return redirect(url_for("index"))

@app.route("/toggle_files", methods=["POST"])
def toggle_files():
    global SHOW_FILES
    SHOW_FILES = not SHOW_FILES
    log(f"Show files => {SHOW_FILES}")
    return redirect(url_for("index"))

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    global GLOBAL_CHAT_LOG
    GLOBAL_CHAT_LOG.clear()
    log("Chat log cleared.")
    return redirect(url_for("index"))

@app.route("/delete_file/<filename>", methods=["POST"])
def delete_file(filename):
    global UPLOADED_FILES, UPLOADED_PDF_TEXT
    UPLOADED_FILES = [f for f in UPLOADED_FILES if f["filename"] != filename]
    if not UPLOADED_FILES:
        UPLOADED_PDF_TEXT = ""
        log("All PDF files removed. Clearing PDF text.")
    else:
        log(f"Deleted file: {filename}")
    return redirect(url_for("index"))

@app.route("/streaming", methods=["GET"])
def streaming():
    """
    Renders the main layout with an additional SSE streaming block.
    The 'question' query parameter is passed to the SSE endpoint.
    """
    question = request.args.get("question", "")
    return render_template_string(
        HTML_TEMPLATE,
        chat_history=GLOBAL_CHAT_LOG,
        show_files=SHOW_FILES,
        uploaded_files=UPLOADED_FILES,
        upload_message=UPLOAD_MESSAGE,
        global_log=GLOBAL_LOG,
        user_subtasks=None,
        streaming=True,
        streaming_question=question,
        llm_executions=LLM_SUBTASK_EXECUTIONS
    )

@app.route("/stream_final_answer", methods=["GET"])
def stream_final_answer():
    """
    SSE endpoint that streams the final answer token-by-token.
    It runs the partial workflow and then (if no USER subtasks are required)
    executes any LLM subtasks before streaming the final answer.
    """
    question = request.args.get("question", "")
    state = run_workflow_partial_plan(UPLOADED_PDF_TEXT, question)
    subtasks = state["execution_result"].get("subtasks", [])
    if any(st.get("type", "").upper() == "USER" for st in subtasks):
        # If USER subtasks exist, we cannot stream the final answer.
        def gen():
            yield "data: [USER subtasks required, cannot stream final answer]\n\n"
            yield "data: [DONE]\n\n"
        return Response(gen(), mimetype="text/event-stream")
    else:
        # Run LLM subtasks (if any)
        state = execute_llm_subtasks(state, {})
        def event_stream():
            for token in stream_final_answer_generator(state):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        return Response(event_stream(), mimetype="text/event-stream")

# ---------------------------------------------------------------
# 7) Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    # For best speed, run with debug=False, or in a production WSGI server
    app.run(debug=False)
