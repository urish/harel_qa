import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, TypedDict

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from rag import load_vector_store
from report_generator import generate_html_report

vector_store = load_vector_store()
print("Vector store loaded.")
prompt = hub.pull("rlm/rag-prompt")
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


class State(TypedDict):
    category: str
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    embedding = vector_store.embedding.embed_query(state["question"])
    results = vector_store.similarity_search_with_score_by_vector(
        embedding,
        k=7,
        filter=lambda doc: state["category"] in doc.metadata.get("category", ""),
    )
    retrieved_docs = [doc for doc, _ in results]
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


def process_question(question_data: dict, question_num: int, total: int) -> dict:
    """Process a single question and return the result."""
    print(f"Processing question {question_num}/{total}: {question_data['question'][:50]}...")

    response = graph.invoke({
        "question": question_data["question"],
        "category": question_data["category"],
    })

    return {
        "question": question_data["question"],
        "category": question_data["category"],
        "expected_answer": question_data["answer"],
        "expected_citation": question_data["citation"],
        "actual_answer": response["answer"],
    }


def run_eval_mode(evalset_path: str, output_path: str) -> None:
    """Run evaluation mode: process all questions and generate HTML report."""
    print(f"Loading evaluation set from {evalset_path}...")

    with open(evalset_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"Found {len(questions)} questions. Processing in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_question, q, i+1, len(questions)): i
            for i, q in enumerate(questions)
        }

        for future in as_completed(futures):
            results.append(future.result())

    # Sort results to maintain original order
    results.sort(key=lambda x: questions.index(
        next(q for q in questions if q["question"] == x["question"])
    ))

    print(f"Generating HTML report to {output_path}...")
    generate_html_report(results, output_path)
    print(f"Report generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Chatbot with RAG capabilities")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation mode using evalset.json"
    )

    args = parser.parse_args()

    if args.eval:
        evalset_path = Path(__file__).parent.parent / "eval" / "evalset.json"
        output_path = Path(__file__).parent.parent / "evalset-report.html"
        run_eval_mode(str(evalset_path), str(output_path))
    else:
        print("Application ready, asking a sample question...")
        response = graph.invoke(
            {
                "question": "עבור ביטוח דירה, האם יש כיסוי למקרה נזק של מחשב נייד?",
                "category": "apartment",
            }
        )
        print(response["answer"])


if __name__ == "__main__":
    main()
