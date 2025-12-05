# this code will import a newer version of sqlite3 if the version is less than 3.35.X
# which is necessary for chromadb >= v0.4
import sqlite3
import sys

if (sqlite3.sqlite_version_info[0] < 3) or (
    (sqlite3.sqlite_version_info[0] == 3) and (sqlite3.sqlite_version_info[1] < 35)
):
    print("Upgrading sqlite3 version from " + sqlite3.sqlite_version)
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    import sqlite3

    print("New sqlite3 version is " + sqlite3.sqlite_version)
    import chromadb  # noqa: F401

import os
import shutil
import warnings
import argparse
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core._api import LangChainDeprecationWarning
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def clear_database():
    """From https://github.com/pixegami/rag-tutorial-v2"""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def load_documents(pdf_directory):
    """From https://github.com/pixegami/rag-tutorial-v2"""
    document_loader = PyPDFDirectoryLoader(pdf_directory)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """From https://github.com/pixegami/rag-tutorial-v2"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def get_embedding_function():
    """From https://github.com/pixegami/rag-tutorial-v2"""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


def calculate_chunk_ids(chunks):
    """From https://github.com/pixegami/rag-tutorial-v2

    This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks: list[Document]):
    """From https://github.com/pixegami/rag-tutorial-v2"""
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")


def query_rag(query_text: str, k=25, verbose=False):
    """Modified from https://github.com/pixegami/rag-tutorial-v2"""
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=k)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    if verbose:
        print(prompt)

    model = Ollama(model="mistral:7b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def main():
    # TODO: add this flag to docstring
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument(
        "--initialize", action="store_true", help="Initialize the database."
    )
    parser.add_argument("--query", action="store_true", help="Query the database.")
    parser.add_argument("query_text", type=str, help="The query text.", nargs="?")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()
    if args.initialize:
        # Create (or update) the data store.
        # TODO: make path a command line argument
        documents = load_documents("data")
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    if args.query:
        # Create CLI.
        query_text = args.query_text
        query_rag(query_text)


if __name__ == "__main__":
    main()