from typing import List, TypedDict

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from rag import load_vector_store

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

print("Application ready, asking a sample question...")
response = graph.invoke(
    {
        "question": "עבור ביטוח דירה, האם יש כיסוי למקרה נזק של מחשב נייד?",
        "category": "apartment",
    }
)
print(response["answer"])
