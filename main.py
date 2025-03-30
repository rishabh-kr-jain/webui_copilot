# main.py

import os
from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator import Orchestrator

# Paths to your persisted vectorstores
UN_VECTORSTORE_DIR = "un_food_index"
CLINICAL_VECTORSTORE_DIR = "clinical_index"

# Initialize orchestrator
orchestrator = Orchestrator(
    food_vectorstore_dir=UN_VECTORSTORE_DIR,
    clinical_vectorstore_dir=CLINICAL_VECTORSTORE_DIR
)

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    user_question = request.question
    answer = orchestrator.run(user_question)
    return ChatResponse(answer=answer)
