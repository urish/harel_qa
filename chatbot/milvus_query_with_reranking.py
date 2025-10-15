#!/usr/bin/env python3

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import click
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_milvus import Milvus
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from pprint import pprint

from sentence_transformers import CrossEncoder

# Load the Cross-Encoder model once at system startup
# This model takes a (query, document) pair and outputs a relevance score
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
INITIAL_RETRIEVAL_K = 20
reranker_model = CrossEncoder(RERANKER_MODEL_NAME)

def rerank_documents(query: str, retrieved_docs: list[Document], top_n: int = 5) -> list[Document]:
    """
    Reranks a list of retrieved documents based on relevance to the query.
    """
    # 1. Create pairs of (query, document_content) for the cross-encoder
    sentences = [(query, doc.page_content) for doc in retrieved_docs]

    # 2. Score all pairs (higher score is better)
    scores = reranker_model.predict(sentences)

    # 3. Pair the original document with its new score
    doc_scores = list(zip(retrieved_docs, scores))

    # 4. Sort the documents by their score in descending order
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # 5. Return the content of the top_n documents
    reranked_docs = [doc for doc, score in doc_scores[:top_n]]

    return reranked_docs

def setup_embeddings(model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
    return SentenceTransformerEmbeddings(model_name=model_name)

def search_documents(collection_name: str, question: str, embedding_function, 
                    k: int = 5, milvus_host: str = "localhost", milvus_port: int = 19530) -> List[Document]:
    vector_store = Milvus(
        embedding_function=embedding_function,
        collection_name=collection_name
    )
    
    # Perform similarity search
    print(f"Searching collection '{collection_name}' for: '{question}'")
    results = vector_store.similarity_search(question, k=k)
    
    print(f"Found {len(results)} relevant documents")
    return results
        


@click.command()
@click.option("--question", "-q", required=True, help="Question to ask")
@click.option("--collection-name", "-c", default="documents", help="Milvus collection name")
@click.option("--k", default=5, help="Number of documents to retrieve")
@click.option("--model-name", default="paraphrase-multilingual-mpnet-base-v2", 
              help="Sentence transformer model for embeddings")
@click.option("--gemini-model", default="gemini-2.5-flash", help="Gemini model name")
@click.option("--api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
@click.option("--milvus-host", default="localhost", help="Milvus server host")
@click.option("--milvus-port", default=19530, help="Milvus server port")
@click.option("--show-sources/--no-sources", default=True, help="Show source documents")
@click.option("--show-context/--no-context", default=False, help="Show full context")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode for multiple queries")

def main(question: str, collection_name: str, k: int, model_name: str, gemini_model: str,
         api_key: Optional[str], milvus_host: str, milvus_port: int, 
         show_sources: bool, show_context: bool, interactive: bool):
    
    embedding_function = setup_embeddings(model_name)
    documents = search_documents(collection_name, question, embedding_function, INITIAL_RETRIEVAL_K, milvus_host, milvus_port)

    #pprint("Retrieved Documents:")    
    #pprint(documents)

    reranked_documents = rerank_documents(
        query=question,
        retrieved_docs=documents,
        top_n=k
    )

    print(f"Reranked {len(documents)} to {len(reranked_documents)} relevant documents")
    pprint(reranked_documents)



if __name__ == "__main__":
    main()
