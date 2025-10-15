import argparse
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, TypedDict

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from rag_query import setup_embeddings, query_rag
from report_generator import generate_html_report

print("🔧 Setting up RAG components...")
embedding_function = setup_embeddings()
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


def process_question(question_data: dict, question_num: int, total: int) -> dict:
    """Process a single question and return the result."""
    print(
        f"Processing question {question_num}/{total}: {question_data['question'][:50]}..."
    )

    # Ensure an asyncio event loop exists in this thread, as the Milvus client requires it.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    response = query_rag(
        question=question_data["question"],
        category=question_data["category"],
        collection_name="documents",
        embedding_function=embedding_function,
        llm=llm,
    )

    return {
        "question": question_data["question"],
        "category": question_data["category"],
        "expected_answer": question_data["answer"],
        "expected_citation": question_data["citation"],
        "actual_answer": response.answer,
    }


def run_eval_mode(evalset_path: str, output_path: str) -> None:
    """Run evaluation mode: process all questions and generate HTML report."""
    print(f"Loading evaluation set from {evalset_path}...")

    with open(evalset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Found {len(questions)} questions. Processing in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_question, q, i + 1, len(questions)): i
            for i, q in enumerate(questions)
        }

        for future in as_completed(futures):
            results.append(future.result())

    # Sort results to maintain original order
    results.sort(
        key=lambda x: questions.index(
            next(q for q in questions if q["question"] == x["question"])
        )
    )

    print(f"Generating HTML report to {output_path}...")
    generate_html_report(results, output_path)
    print(f"Report generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Chatbot with RAG capabilities")
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation mode using evalset.json"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the web server with API and web interface",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )

    args = parser.parse_args()

    if args.serve:
        from server import run_server
        from api import create_app

        app = create_app(embeddings_function=embedding_function, llm=llm)
        run_server(app, host=args.host, port=args.port)
    elif args.eval:
        evalset_path = Path(__file__).parent.parent / "eval" / "evalset.json"
        output_path = Path(__file__).parent.parent / "evalset-report.html"
        run_eval_mode(str(evalset_path), str(output_path))
    else:
        print("Application ready, asking a sample question...")
        response = query_rag(
            question="עבור ביטוח דירה, האם יש כיסוי למקרה נזק של מחשב נייד?",
            collection_name="documents",
            category="apartment",
            embedding_function=embedding_function,
            llm=llm,
        )
        print(response.answer)


if __name__ == "__main__":
    main()
