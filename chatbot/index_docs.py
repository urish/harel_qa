import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import VECTOR_DB_FILE, embeddings


def load_documents_from_filesystem():
    """Load documents from the data directory using LangChain document loaders."""
    print("Loading documents from filesystem using LangChain loaders...")

    # Get the data directory path (go up one level from chatbot folder)
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)

    all_documents = []

    # Load markdown files from docs directories
    for category_dir in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category_dir)
        if os.path.isdir(category_path):
            docs_path = os.path.join(category_path, "docs")
            pages_path = os.path.join(category_path, "pages")

            # Load .md files from docs directory
            if os.path.exists(docs_path):
                try:
                    md_loader = DirectoryLoader(
                        docs_path,
                        glob="*.md",
                        loader_cls=UnstructuredMarkdownLoader,
                        show_progress=True,
                    )
                    md_docs = md_loader.load()
                    for doc in md_docs:
                        doc.metadata["category"] = category_dir
                    all_documents.extend(md_docs)
                except Exception as e:
                    print(f"Error loading markdown files from {docs_path}: {e}")

            # Load .txt files from pages directory
            if os.path.exists(pages_path):
                try:
                    txt_loader = DirectoryLoader(
                        pages_path,
                        glob="*.txt",
                        loader_cls=TextLoader,
                        loader_kwargs={"encoding": "utf-8"},
                        show_progress=True,
                    )
                    txt_docs = txt_loader.load()
                    for doc in txt_docs:
                        doc.metadata["category"] = category_dir
                    all_documents.extend(txt_docs)
                except Exception as e:
                    print(f"Error loading text files from {pages_path}: {e}")

    print(f"Loaded {len(all_documents)} documents using LangChain loaders")
    return all_documents


if __name__ == "__main__":
    print("Starting document indexing...")
    vector_store = InMemoryVectorStore(embeddings)
    docs = load_documents_from_filesystem()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"Split into {len(all_splits)} chunks of text")
    _ = vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(all_splits)} text chunks into the vector store.")
    vector_store.dump(VECTOR_DB_FILE)
    print(f"Vector store saved to '{VECTOR_DB_FILE}'")
    print("Document indexing completed.")
