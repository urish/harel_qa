import getpass
import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


VECTOR_DB_FILE = "harel_db.json"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def load_vector_store():
    """Load the vector store from disk into memory."""
    if os.path.exists(VECTOR_DB_FILE):
        print(f"Loading vector store from '{VECTOR_DB_FILE}'...")
        return InMemoryVectorStore.load(VECTOR_DB_FILE, embeddings)
    else:
        raise FileNotFoundError(
            f"Vector DB file '{VECTOR_DB_FILE}' not found. Please run index_docs.py first"
        ) from None
