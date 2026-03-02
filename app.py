from flask import Flask, request, jsonify
import requests
import json
import os
import re
import math
from openai import OpenAI
from PyPDF2 import PdfReader

app = Flask(__name__)

APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")
MEMORY_FILE = "memory.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace("wfh", "work from home")
    text = text.replace("hr", "human resources")
    text = text.replace("it", "information technology")
    return text

def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def ask_ai(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful company assistant."},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def ask_rag(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a company assistant. Answer ONLY using the provided document context. If answer is not found in the context, say 'Not found in uploaded documents.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )
    return response.choices[0].message.content.strip()

# -------------------------
# LOAD MEMORY
# -------------------------

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)
else:
    memory = {"manual_memory": {}, "document_chunks": {}}

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

# -------------------------
# LARK AUTH
# -------------------------

def get_tenant_access_token():
    url = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
    payload = {"app_id": APP_ID, "app_secret": APP_SECRET}
    response = requests.post(url, json=payload)
    return response.json()["tenant_access_token"]

def send_message(chat_id, text):
    token = get_tenant_access_token()
    url = "https://open.larksuite.com/open-apis/im/v1/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text})
    }
    requests.post(url + "?receive_id_type=chat_id", headers=headers, json=payload)

# -------------------------
# WEBHOOK
# -------------------------

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json

    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    event = data.get("event", {})
    message = event.get("message", {})
    chat_id = message.get("chat_id")

    text_content = message.get("content", "{}")
    text_dict = json.loads(text_content)
    user_text = text_dict.get("text", "").strip()
    user_clean = clean_text(user_text)

    reply = None

    # -------------------------
    # TEACH (Manual Memory)
    # -------------------------
    if user_clean.startswith("teach"):
        try:
            content = user_text.replace("teach:", "").strip()
            question, answer = content.split("=")

            clean_question = clean_text(question.strip())
            embedding = get_embedding(clean_question)

            memory["manual_memory"][clean_question] = {
                "answer": answer.strip(),
                "embedding": embedding
            }

            save_memory()
            reply = "Learned successfully ✅"

        except:
            reply = "Format should be: teach: question = answer"

    # -------------------------
    # SEARCH (Manual Memory)
    # -------------------------
    else:
        user_embedding = get_embedding(user_clean)

        # 1️⃣ Check Manual Memory
        best_score = 0
        best_answer = None

        for question, data_item in memory["manual_memory"].items():
            score = cosine_similarity(user_embedding, data_item["embedding"])
            if score > best_score:
                best_score = score
                best_answer = data_item["answer"]

        if best_score > 0.75:
            reply = best_answer

        # 2️⃣ If Not Found → Search Document Chunks
        if not reply and memory["document_chunks"]:
            best_score = 0
            best_chunk = None

            for chunk_id, chunk_data in memory["document_chunks"].items():
                score = cosine_similarity(user_embedding, chunk_data["embedding"])
                if score > best_score:
                    best_score = score
                    best_chunk = chunk_data["text"]

            if best_score > 0.70:
                reply = ask_rag(user_text, best_chunk)

    # -------------------------
  # STRICT RAG MODE
if not reply:
    reply = "Not found in uploaded documents."

    send_message(chat_id, reply)
    return jsonify({"status": "ok"})

# -------------------------
# PDF UPLOAD ROUTE
# -------------------------

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"doc_chunk_{len(memory['document_chunks']) + i}"
            embedding = get_embedding(chunk)

            memory["document_chunks"][chunk_id] = {
                "text": chunk,
                "embedding": embedding
            }

        save_memory()

        return jsonify({
            "message": "Document uploaded and processed successfully",
            "chunks_created": len(chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)