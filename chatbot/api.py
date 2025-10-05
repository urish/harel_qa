"""FastAPI application for the Harel chatbot."""
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot import graph


class ChatRequest(BaseModel):
    question: str
    category: str


class ChatResponse(BaseModel):
    answer: str
    category: str
    question: str


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Harel Insurance Chatbot API",
        description="RAG-based chatbot for Harel insurance products",
        version="1.0.0"
    )

    # Enable CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api")
    async def root():
        """Health check endpoint."""
        return {"status": "ok", "message": "Harel Chatbot API is running"}

    @app.get("/api/categories")
    async def get_categories() -> List[str]:
        """Get available insurance categories."""
        return ["apartment", "business", "car", "health", "life", "travel"]

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Process a chat question and return an answer."""
        try:
            response = graph.invoke({
                "question": request.question,
                "category": request.category,
            })

            return ChatResponse(
                answer=response["answer"],
                category=request.category,
                question=request.question
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

    return app