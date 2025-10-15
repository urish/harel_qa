"""FastAPI application for the Harel chatbot."""

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_query import query_rag
from source_utils import format_display_filename


class ChatRequest(BaseModel):
    question: str
    category: str


class Source(BaseModel):
    source_file: str
    page_number: str
    content: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    category: str
    question: str
    sources: List[Source]


def create_app(embeddings_function, llm) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Harel Insurance Chatbot API",
        description="RAG-based chatbot for Harel insurance products",
        version="1.0.0",
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
            response = query_rag(
                question=request.question,
                collection_name="documents",
                category=request.category,
                embedding_function=embeddings_function,
                llm=llm,
            )

            return ChatResponse(
                answer=response.answer,
                category=request.category,
                question=request.question,
                sources=[
                    Source(
                        source_file=format_display_filename(s["source_file"]),
                        page_number=s["page_number"],
                        content=s["content"],
                        metadata=s["metadata"],
                    )
                    for s in response.sources
                ],
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing question: {str(e)}"
            )

    return app
