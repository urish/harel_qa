from typing import List, TypedDict

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from rag import load_vector_store

vector_store = load_vector_store()
print("Vector store loaded.")
prompt = hub.pull("rlm/rag-prompt")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
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
    {"question": "עבור ביטוח דירה, האם יש כיסוי למקרה נזק של מחשב נייד?"}
)
print(response["answer"])
