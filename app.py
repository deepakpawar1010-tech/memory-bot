from flask import Flask, request, jsonify
import requests
import json
import os
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

app = Flask(__name__)

APP_ID = "---"
APP_SECRET = "---"

MEMORY_FILE = "memory.json"

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
# Load Embedding Model
# -----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------
# OpenAI Client
# -----------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    # TEACHING
    if user_clean.startswith("teach"):
        try:
            content = user_text.replace("teach:", "").strip()
            question, answer = content.split("=")

            clean_question = clean_text(question.strip())
            qa_memory[clean_question] = answer.strip()
            save_memory()

            reply = "Learned successfully ✅"
        except:
            reply = "Format should be: teach: question = answer"

    # SEMANTIC SEARCH
    else:
        if qa_memory:
            questions = list(qa_memory.keys())
            question_embeddings = model.encode(questions)
            user_embedding = model.encode([user_clean])

            similarities = cosine_similarity(user_embedding, question_embeddings)[0]
            best_match_index = similarities.argmax()
            best_score = similarities[best_match_index]

            if best_score > 0.55:
                reply = qa_memory[questions[best_match_index]]

    # AI FALLBACK
    if not reply:
        reply = ask_ai(user_text)

    send_message(chat_id, reply)
    return jsonify({"status": "ok"})

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)