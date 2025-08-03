import os
import torch
import chromadb
import shutil
from paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_all_gutenberg_books


def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "gutenberg_books",
    delete_existing: bool = False,
) -> chromadb.Collection:
    """
    Initialize a ChromaDB instance and persist it to disk.

    Args:
        persist_directory (str): The directory where ChromaDB will persist data. Defaults to "./vector_db"
        collection_name (str): The name of the collection to create/get. Defaults to "gutenberg_books"
        delete_existing (bool): Whether to delete the existing database if it exists. Defaults to False
    Returns:
        chromadb.Collection: The ChromaDB collection instance
    """
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get a collection
    try:
        # Try to get existing collection first
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except Exception:
        # If collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            },  # Use cosine distance for semantic search
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

    return collection


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "gutenberg_books",
) -> chromadb.Collection:
    """
    Get a ChromaDB client instance.

    Args:
        persist_directory (str): The directory where ChromaDB persists data
        collection_name (str): The name of the collection to get

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance
    """
    return chromadb.PersistentClient(path=persist_directory).get_collection(
        name=collection_name
    )


def chunk_book(
    book: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Chunk the book into smaller documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(book)


def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    embeddings = model.embed_documents(documents)
    return embeddings


def insert_books(collection: chromadb.Collection, books: list[str]):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        documents (list[str]): The documents to insert

    Returns:
        None
    """
    next_id = collection.count()
    i=0

    for book in books:
        i=i+1
        chunked_book = chunk_book(book)
        embeddings = embed_documents(chunked_book)
        ids = list(range(next_id, next_id + len(chunked_book)))
        ids = [f"document_{id}" for id in ids]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_book,
        )
        print(f"book number {i} added in collection")
        next_id += len(chunked_book)


def main():
    collection = initialize_db(
        persist_directory=VECTOR_DB_DIR,
        collection_name="gutenberg_books",
        delete_existing=True,
    )
    books = load_all_gutenberg_books()
    insert_books(collection, books)

    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
