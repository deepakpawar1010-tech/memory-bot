from flask import Flask, request, jsonify
import requests
import json
import os
import re
import math
from openai import OpenAI
from PyPDF2 import PdfReader
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)

APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")

DOC_THRESHOLD = 0.55
MANUAL_THRESHOLD = 0.55

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# DATABASE CONNECTION (SAFE)
# -------------------------

def get_db_connection():
    return psycopg2.connect(
        host="aws-1-ap-south-1.pooler.supabase.com",
        database="postgres",
        user="postgres.udmzdiywjanygdtrpnuf",
        password=os.getenv("SUPABASE_DB_PASSWORD"),
        port=5432,
        sslmode="require"
    )

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
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("wfh", "work from home")
    text = text.replace("hr", "human resources")
    text = text.replace("it", "information technology")
    return text


def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
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
            {"role": "system", "content": "You are a general knowledge assistant."},
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
                "content": "Answer ONLY using the provided document context. If not found, say 'Not found in uploaded documents.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )
    return response.choices[0].message.content.strip()

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
    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
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
        # TEACH
        # -------------------------
        if user_clean.startswith("teach"):
            content = user_text.replace("teach:", "").strip()
            question, answer = content.split("=")

            clean_question = clean_text(question.strip())
            embedding = get_embedding(clean_question)

            cursor.execute(
                "INSERT INTO manual_memory (question, answer, embedding) VALUES (%s, %s, %s)",
                (clean_question, answer.strip(), json.dumps(embedding))
            )

            reply = "Learned successfully ✅"

        else:
            user_embedding = get_embedding(user_clean)

            # Manual Memory Search
            cursor.execute("SELECT * FROM manual_memory")
            rows = cursor.fetchall()

            best_score = 0
            best_answer = None

            for row in rows:
                stored_embedding = row["embedding"]
                if isinstance(stored_embedding, str):
                    stored_embedding = json.loads(stored_embedding)

                score = cosine_similarity(user_embedding, stored_embedding)

                if score > best_score:
                    best_score = score
                    best_answer = row["answer"]

            if best_score > MANUAL_THRESHOLD:
                reply = best_answer

            # Document Search
            if not reply:
                cursor.execute("SELECT * FROM document_chunks")
                rows = cursor.fetchall()

                scored_chunks = []

                for row in rows:
                    stored_embedding = row["embedding"]
                    if isinstance(stored_embedding, str):
                        stored_embedding = json.loads(stored_embedding)

                    score = cosine_similarity(user_embedding, stored_embedding)
                    scored_chunks.append((score, row["text"]))

                scored_chunks.sort(key=lambda x: x[0], reverse=True)

                if scored_chunks:
                    top_chunks = scored_chunks[:3]
                    best_doc_score = top_chunks[0][0]

                    if best_doc_score > DOC_THRESHOLD:
                        combined_context = "\n\n".join(
                            [chunk for score, chunk in top_chunks]
                        )
                        reply = ask_rag(user_text, combined_context)

        if not reply:
            gpt_answer = ask_ai(user_text)
            reply = (
                "Not present in uploaded documents.\n\n"
                "GPT Answer:\n"
                f"{gpt_answer}"
            )

        send_message(chat_id, reply)
        return jsonify({"status": "ok"})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

# -------------------------
# PDF UPLOAD
# -------------------------

@app.route("/upload", methods=["POST"])
def upload_pdf():
    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = get_embedding(chunk)
            cursor.execute(
                "INSERT INTO document_chunks (text, embedding) VALUES (%s, %s)",
                (chunk, json.dumps(embedding))
            )

        return jsonify({
            "message": "Document uploaded successfully",
            "chunks_created": len(chunks)
        })

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)