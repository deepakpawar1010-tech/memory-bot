from flask import Flask, request, jsonify
import requests
import json
import os
import re
import math
from openai import OpenAI

app = Flask(__name__)

APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")

MEMORY_FILE = "memory.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# Clean Text
# -----------------------
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace("wfh", "work from home")
    text = text.replace("hr", "human resources")
    text = text.replace("it", "information technology")
    return text

# -----------------------
# Cosine Similarity (Pure Python)
# -----------------------
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot / (norm1 * norm2)

# -----------------------
# Load Memory
# -----------------------
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        qa_memory = json.load(f)
else:
    qa_memory = {}

def save_memory():
    with open(MEMORY_FILE, "w") as f:
        json.dump(qa_memory, f, indent=4)

# -----------------------
# Get OpenAI Embedding
# -----------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# -----------------------
# Ask AI
# -----------------------
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

# -----------------------
# Lark Auth
# -----------------------
def get_tenant_access_token():
    url = "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
    payload = {
        "app_id": APP_ID,
        "app_secret": APP_SECRET
    }
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

# -----------------------
# Webhook
# -----------------------
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

    # TEACH
    if user_clean.startswith("teach"):
        try:
            content = user_text.replace("teach:", "").strip()
            question, answer = content.split("=")

            clean_question = clean_text(question.strip())
            embedding = get_embedding(clean_question)

            qa_memory[clean_question] = {
                "answer": answer.strip(),
                "embedding": embedding
            }

            save_memory()
            reply = "Learned successfully ✅"
        except:
            reply = "Format should be: teach: question = answer"

    # SEARCH
    else:
        if qa_memory:
            user_embedding = get_embedding(user_clean)

            best_score = 0
            best_answer = None

            for question, data in qa_memory.items():
                score = cosine_similarity(user_embedding, data["embedding"])
                if score > best_score:
                    best_score = score
                    best_answer = data["answer"]

            if best_score > 0.75:
                reply = best_answer

    # FALLBACK
    if not reply:
        reply = ask_ai(user_text)

    send_message(chat_id, reply)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)