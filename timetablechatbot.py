import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


app = FastAPI()


class Query(BaseModel):
    message: str


df = pd.read_csv("documents/university_timetable.csv")


df["combined"] = df.apply(
    lambda row: f"Day: {row.day}, Time: {row.time}, Course: {row.course}, Teacher: {row.teacher}, Room: {row.room}",
    axis=1
)

documents = df["combined"].tolist()


embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")


doc_embeddings = embedder.encode(documents, normalize_embeddings=True)


os.environ["GROQ_API_KEY"] = "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"

client = Groq()


def retrieve(query, top_k=3):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [documents[i] for i in top_indices]


def ask_llm(context, question):
    prompt = f"""
You are a university timetable assistant.

Use ONLY the following context to answer.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful university timetable assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


@app.post("/chat")
def chat(query: Query):
    context_docs = retrieve(query.message)
    context = "\n".join(context_docs)
    answer = ask_llm(context, query.message)

    return {"response": answer}


@app.get("/", response_class=HTMLResponse)
def homepage():
    return """
<!DOCTYPE html>
<html>
<head>
<title>AI Timetable Assistant</title>
<style>
body{font-family:Arial;background:#0f172a;display:flex;justify-content:center;align-items:center;height:100vh;}
.chat-container{width:420px;height:600px;background:white;border-radius:12px;display:flex;flex-direction:column;padding:20px;}
#chatbox{flex:1;overflow-y:auto;margin-bottom:10px;}
.message{padding:10px;margin:6px;border-radius:6px;max-width:80%;}
.user{background:#3b82f6;color:white;margin-left:auto;}
.bot{background:#e5e7eb;}
.input-area{display:flex;}
input{flex:1;padding:10px;}
button{padding:10px;background:#3b82f6;color:white;border:none;cursor:pointer;}
</style>
</head>
<body>
<div class="chat-container">
<h2>🎓 AI Timetable Assistant</h2>
<div id="chatbox"></div>
<div class="input-area">
<input id="userInput" placeholder="Ask about your timetable">
<button onclick="sendMessage()">Send</button>
</div>
</div>

<script>
async function sendMessage(){
let input=document.getElementById("userInput")
let message=input.value
if(!message) return

addMessage(message,"user")
input.value=""

let response=await fetch("/chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({message:message})
})

let data=await response.json()

addMessage(data.response,"bot")
}

function addMessage(text,type){
let chatbox=document.getElementById("chatbox")
let div=document.createElement("div")
div.className="message "+type
div.innerText=text
chatbox.appendChild(div)
chatbox.scrollTop=chatbox.scrollHeight
}
</script>

</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)