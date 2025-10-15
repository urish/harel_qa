#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) query script using LangChain.

This script:
1. Takes a question and collection name
2. Searches the Milvus vector database for relevant documents
3. Creates context from retrieved documents
4. Sends question + context to Gemini LLM
5. Returns the answer with source citations
"""

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

@dataclass
class QueryResult:
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    context: str

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
    """Setup the embedding function for vector search."""
    return SentenceTransformerEmbeddings(model_name=model_name)

def search_documents(collection_name: str, question: str, embedding_function, 
                    k: int = 5, milvus_host: str = "localhost", milvus_port: int = 19530) -> List[Document]:
    """Search for relevant documents in Milvus collection."""
    try:
        # Connect to existing Milvus collection
        vector_store = Milvus(
            embedding_function=embedding_function,
            collection_name=collection_name,
            connection_args={"host": milvus_host, "port": milvus_port}
        )
        
        # Perform similarity search
        print(f"Searching collection '{collection_name}' for: '{question}'")
        results = vector_store.similarity_search(question, k=k)
        
        print(f"Found {len(results)} relevant documents")
        return results
        
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

def create_context(documents: List[Document]) -> str:
    """Create context string from retrieved documents."""
    if not documents:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        # Extract metadata for citation
        metadata = doc.metadata
        source_file = metadata.get("source_file", "Unknown")
        page_number = metadata.get("page_number", "Unknown")
        doc_type = metadata.get("doc_type", "Unknown")
        
        # Create citation
        citation = f"[{i}] Source: {source_file}, {doc_type}: {page_number}"
        
        # Add document content with citation
        context_parts.append(f"{citation}\n{doc.page_content}\n")
    
    return "\n".join(context_parts)

def setup_gemini_llm(api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
    """Setup Gemini LLM for question answering."""
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass --api-key")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=2048
    )

def create_prompt_template() -> ChatPromptTemplate:
    """Create the prompt template for RAG."""
    template = """You are a helpful assistant that answers questions based on the provided context documents.

Context Documents:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context documents
2. If the context doesn't contain enough information to answer the question, say so
3. Cite your sources using the format [1], [2], etc. where the numbers correspond to the document citations
4. Be concise but comprehensive
5. If the question is in Hebrew, answer in Hebrew. If in English, answer in English
6. For insurance-related questions, provide specific details from the documents when available

Answer:"""
    
    return ChatPromptTemplate.from_template(template)

def query_rag(question: str, collection_name: str, embedding_function, llm, 
             k: int = 5, milvus_host: str = "localhost", milvus_port: int = 19530) -> QueryResult:
    """Perform RAG query: search, create context, and generate answer."""
    
    # 1. Search for relevant documents
    documents = search_documents(collection_name, question, embedding_function, INITIAL_RETRIEVAL_K, milvus_host, milvus_port)
    
    reranked_documents = rerank_documents(
        query=question,
        retrieved_docs=documents,
        top_n=k
    )

    # 2. Create context from documents
    context = create_context(reranked_documents)
    
    # 3. Prepare sources for result
    sources = []
    for doc in reranked_documents:
        sources.append({
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
            "source_file": doc.metadata.get("source_file", "Unknown"),
            "page_number": doc.metadata.get("page_number", "Unknown")
        })
    
    # 4. Generate answer using LLM
    prompt_template = create_prompt_template()
    chain = prompt_template | llm
    
    try:
        response = chain.invoke({
            "context": context,
            "question": question
        })
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {e}"
    
    return QueryResult(
        question=question,
        answer=answer,
        sources=sources,
        context=context
    )

def print_results(result: QueryResult, show_sources: bool = True, show_context: bool = False):
    """Print the RAG query results in a formatted way."""
    print("\n" + "="*80)
    print("🤖 RAG QUERY RESULTS")
    print("="*80)
    
    print(f"\n❓ Question: {result.question}")
    print(f"\n💡 Answer:\n{result.answer}")
    
    if show_sources:
        print(f"\n📚 Sources ({len(result.sources)} documents):")
        print("-" * 40)
        for i, source in enumerate(result.sources, 1):
            print(f"[{i}] {source['source_file']} (Page: {source['page_number']})")
            print(f"    {source['content']}")
            print()
    
    if show_context:
        print(f"\n📄 Full Context:")
        print("-" * 40)
        print(result.context)

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
    """Query the RAG system with a question."""
    
    try:
        # Setup components
        print("🔧 Setting up RAG components...")
        embedding_function = setup_embeddings(model_name)
        llm = setup_gemini_llm(api_key, gemini_model)
        
        if interactive:
            print(f"\n🚀 Interactive RAG mode. Collection: '{collection_name}'")
            print("Type 'quit' or 'exit' to stop.\n")
            
            while True:
                try:
                    user_question = input("❓ Your question: ").strip()
                    if user_question.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not user_question:
                        continue
                    
                    # Perform RAG query
                    result = query_rag(user_question, collection_name, embedding_function, llm, 
                                     k, milvus_host, milvus_port)
                    print_results(result, show_sources, show_context)
                    print("\n" + "-"*80 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        else:
            # Single query mode
            print(f"🚀 Processing question: '{question}'")
            result = query_rag(question, collection_name, embedding_function, llm, 
                             k, milvus_host, milvus_port)
            print_results(result, show_sources, show_context)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
