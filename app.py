# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma:2b"

# In-memory session store
sessions = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Get or create session
    history = sessions.get(req.user_id, [])

    # Append user message
    history.append({"role": "user", "content": req.message})

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(OLLAMA_URL, json={
                "model": MODEL,
                "messages": history,
                "stream": False
            })

        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Ollama error")

        data = res.json()
        assistant_msg = data["message"]["content"]

        # Save assistant response
        history.append({"role": "assistant", "content": assistant_msg})
        sessions[req.user_id] = history

        return ChatResponse(response=assistant_msg)

    except httpx.RequestError:
        raise HTTPException(status_code=500, detail="Cannot reach Ollama")
