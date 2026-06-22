import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic

app = FastAPI()
client = anthropic.Anthropic()  # automatically reads ANTHROPIC_API_KEY from the environment

SYSTEM_PROMPT = (
    "You are a warm, professional assistant for a central Maine real estate agent "
    "(The Home Shore). Your job is to chat with prospective home buyers and gently "
    "qualify them: ask about their budget, the towns or areas they're interested in, "
    "must-have features, their timeline for buying, and whether they're pre-approved "
    "for a mortgage. Ask one or two questions at a time and keep replies short and "
    "conversational. Never invent specific listings or prices."
)

class ChatRequest(BaseModel):
    messages: list  # [{"role": "user"/"assistant", "content": "..."}]

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    resp = client.messages.create(
        model="claude-sonnet-4-6",   # swap to claude-haiku-4-5 to cut cost/latency
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=req.messages,
    )
    reply = "".join(b.text for b in resp.content if b.type == "text")
    return {"reply": reply}
