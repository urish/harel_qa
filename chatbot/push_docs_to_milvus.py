#!/usr/bin/env python3
"""
Process parsed text files, chunk them, generate embeddings, and store in Milvus.

This script:
1. Traverses txt files (processed by index_docs_to_local_pages.py)
2. Parses documents with metadata extraction
3. Chunks documents using RecursiveCharacterTextSplitter
4. Generates embeddings using paraphrase-multilingual-mpnet-base-v2
5. Stores everything in Milvus database
"""

import os
import sys
import pathlib
import re
from uuid import uuid4
import click

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pprint import pprint

from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document structure
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]
    source_file: str
    id: str

def extract_metadata_from_header(header_line: str) -> Tuple[str, str, str]:
    """Extract metadata from the header format: <<<SOURCE:path | TYPE:identifier>>>"""
    # Pattern: <<<SOURCE:/path/to/file.pdf | PAGE:3>>>
    pattern = r"<<<SOURCE:(.+?)\s*\|\s*(\w+):(.+?)>>>"
    match = re.match(pattern, header_line.strip())
    if match:
        source_path, doc_type, identifier = match.groups()
        return source_path, doc_type, identifier
    return "", "", ""

def parse_txt_file(file_path: pathlib.Path) -> List[Document]:
    """Parse a single txt file and extract documents with metadata."""
    documents = []
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return documents
    
    sections = re.split(r'<<<SOURCE:.*?>>>', content)
    headers = re.findall(r'<<<SOURCE:(.*?)>>>', content)
    
    if len(sections) != len(headers) + 1:
        print(f"Warning: Mismatched sections/headers in {file_path}")
        return documents
    
    for i, (header, section) in enumerate(zip(headers, sections[1:]), 1):
        if not section.strip():
            continue
            
        source_path, doc_type, page_number = extract_metadata_from_header(f"<<<SOURCE:{header}>>>")
        
        doc = Document(
            metadata={
                "source_file": source_path,
                "doc_type": doc_type,
                "page_number": page_number,
                "file_name": file_path.name,
                "section_index": i
            },
            page_content=section.strip(),
            source_file=source_path,
            id=f"{file_path.stem}_{doc_type}_{page_number}_{i}"
        )
        documents.append(doc)
    
    return documents

def traverse_txt_files(input_dir: pathlib.Path) -> List[pathlib.Path]:
    """Find all .txt files in the input directory."""
    txt_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(pathlib.Path(root) / file)
    return txt_files

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Chunk documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
                source_file=doc.source_file,
                id=f"{doc.id}_chunk_{i}_{hash(chunk_text)}"
            )
            chunked_docs.append(chunk_doc)
    
    return chunked_docs


def generate_embeddings_from_texts(texts: List[str], model_name: str = "paraphrase-multilingual-mpnet-base-v2") -> List[List[float]]:
    model = SentenceTransformer(model_name)
    return model.encode(texts).tolist()

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return generate_embeddings_from_texts(texts, model_name=self.model_name)

    def embed_query(self, text: str) -> List[float]:
        return generate_embeddings_from_texts([text], model_name=self.model_name)[0]

@click.command()
@click.option("--input-dir", type=click.Path(path_type=pathlib.Path, exists=True, file_okay=False), 
              help="Directory containing parsed .txt files")
@click.option("--collection-name", default="documents", help="Milvus collection name")
@click.option("--chunk-size", default=1000, help="Chunk size for text splitting")
@click.option("--chunk-overlap", default=200, help="Overlap between chunks")
@click.option("--model-name", default="paraphrase-multilingual-mpnet-base-v2", 
              help="Sentence transformer model for embeddings")
@click.option("--milvus-host", default="localhost", help="Milvus server host")
@click.option("--milvus-port", default=19530, help="Milvus server port")

def main(input_dir: pathlib.Path, collection_name: str, chunk_size: int, chunk_overlap: int, 
         model_name: str, milvus_host: str, milvus_port: int):
    """Process parsed documents and store in Milvus with embeddings."""
    
    input_dir = input_dir.resolve()
    
    if not input_dir.exists():
        click.echo(f"Input directory not found: {input_dir}", err=True)
        sys.exit(1)
    
    print(f"Processing documents from: {input_dir}")
    
    # 1. Find all txt files
    txt_files = traverse_txt_files(input_dir)
    print(f"Found {len(txt_files)} .txt files")
    
    if not txt_files:
        click.echo("No .txt files found in input directory", err=True)
        sys.exit(1)
    
    # 2. Parse documents with metadata
    all_documents = []
    for txt_file in txt_files:
        print(f"Parsing: {txt_file}")
        documents = parse_txt_file(txt_file)
        all_documents.extend(documents)
        #print(f"  Extracted {len(documents)} documents")

    print(f"Total documents parsed: {len(all_documents)}")

    
    
    # 3. Chunk documents
    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    chunked_documents = chunk_documents(all_documents, chunk_size, chunk_overlap)
    print(f"Created {len(chunked_documents)} chunks")
        

    # 4. Generate embeddings, and store in Milvus
    my_embedding_object = CustomEmbeddings(model_name)

    print(f"len(my_embedding_object) = {len(chunked_documents)}")


    vector_store = Milvus.from_documents(
        documents=chunked_documents,
        embedding=my_embedding_object,
        collection_name=collection_name,
        connection_args={"host": milvus_host, "port": str(milvus_port)}
    )

    print("Done! Documents are now searchable in Milvus.")

if __name__ == "__main__":
    main()
