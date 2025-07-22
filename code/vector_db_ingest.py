import os
import torch
import chromadb
import shutil
from paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_all_books
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import unicodedata
import unicodedata

def safe_text(text: str) -> str:
    """
    Cleans text by handling potential Unicode encoding issues,
    prioritizing preservation of special characters and formulas.
    Adds an initial robust sanitization for deeply malformed strings.
    """
    if not isinstance(text, str):
        return str(text)

    try:
        # Initial robust scrub: Try to encode to a common format (like ASCII or UTF-8)
        # with error handling to remove truly unencodable/malformed characters.
        # This is the most critical change.
        # We try 'ascii' first for characters that definitely shouldn't be there,
        # then 'utf-8' with 'ignore' or 'replace' for anything else.
        
        # Option A: Strict ASCII check (might remove too much, but catches severe issues)
        # cleaned_text = text.encode('ascii', 'ignore').decode('ascii')
        
        # Option B: Direct UTF-8 with 'ignore' (more flexible, generally safer)
        # This will remove any bytes that can't be decoded as valid UTF-8.
        cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')

        # After this initial cleaning, proceed with normalization and surrogate handling
        # on the (now less likely to be fundamentally corrupted) string.

        # Step 1: Normalize the text (NFKC is good for compatibility)
        normalized_text = unicodedata.normalize('NFKC', cleaned_text)

        # Step 2: Re-encode to UTF-16 and decode with 'surrogatepass'.
        # This is still important for handling legitimate surrogate pairs that
        # represent characters outside the BMP. With 'cleaned_text' as input,
        # this step is less likely to fail due to "illegal UTF-16 surrogate"
        # because the initial scrub should have removed truly problematic bytes.
        temp_encoded = normalized_text.encode('utf-16', 'surrogatepass')
        temp_decoded = temp_encoded.decode('utf-16')

        # Step 3: Final encode to UTF-8. Crucially, use 'replace' only at this
        # final stage for characters that are *still* unencodable after
        # all prior processing (e.g., if PyPDFLoader introduced extremely rare
        # or non-standard characters).
        final_encoded = temp_decoded.encode('utf-8', 'replace')
        final_text = final_encoded.decode('utf-8')

        return final_text
    except Exception as e:
        # This fallback should now be rarely hit for the original error,
        # but it's good to keep as a safeguard for unforeseen issues.
        print(f"Warning: An error occurred during text sanitization (fallback): {e}.")
        # Fallback to the most basic and safe replacement if all else fails
        return text.encode('utf-8', 'replace').decode('utf-8')

# def initialize_db(
#     persist_directory: str = VECTOR_DB_DIR,
#     collection_name: str = "books",
#     delete_existing: bool = False,
# ) -> chromadb.Collection:

# def safe_text(text: str) -> str:
#     try:
#         # Convert valid surrogates to full characters
#         text = text.encode('utf-16', 'surrogatepass').decode('utf-16')
#     except UnicodeError:
#         pass
#     return text
    
def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "books",
    delete_existing: bool = False,
) -> Chroma:
    """
    Initialize a ChromaDB instance and persist it to disk.

    Args:
        persist_directory (str): The directory where ChromaDB will persist data. Defaults to "./vector_db"
        collection_name (str): The name of the collection to create/get. Defaults to "books"
        delete_existing (bool): Whether to delete the existing database if it exists. Defaults to False
    Returns:
        chromadb.Collection: The ChromaDB collection instance
    """
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    #client = chromadb.PersistentClient(path=persist_directory)

    # Create or get a collection
    # try:
    #     # Try to get existing collection first
    #     collection = client.get_collection(name=collection_name)
    #     print(f"Retrieved existing collection: {collection_name}")
    # except Exception:
        # If collection doesn't exist, create it


    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key= os.getenv('GOOGLE_API_KEY'))
        # collection = client.create_collection(
        #     name=collection_name,
        #     embedding_function=embeddings,
        #     metadata={
        #         "hnsw:space": "cosine",
        #         "hnsw:batch_size": 10000,
        #     },  # Use cosine distance for semantic search
        # )
        # print(f"Created new collection: {collection_name}")


    vector_store = Chroma(
        collection_name="books",
        embedding_function=embeddings,
        persist_directory=persist_directory,  # Where to save data locally, remove if not necessary
        collection_metadata={
               "hnsw:space": "cosine",
                 "hnsw:batch_size": 10000,
            },  # Use cosine distance for semantic search
         )
        #vector_store.

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

#     vector_store = Chroma(
#     collection_name="example_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )

    return vector_store


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "books",
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


# def chunk_book(
#     book: Document, chunk_size: int = 1000, chunk_overlap: int = 200
# ) -> list[Document]:
#     """
#     Chunk the book into smaller documents.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#     )
#     return text_splitter.split_documents(book)
#     #return text_splitter.split_text(book)


def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key= os.getenv('GOOGLE_API_KEY'))

    embedding_vector = embeddings.embed_documents(documents)
    return embedding_vector


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

#     embeddings = model.embed_documents(documents)
#     return embeddings


def insert_books(vector_store: Chroma, books: list[Document]):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        documents (list[str]): The documents to insert

    Returns:
        None
    """
    # next_id = collection.count()
    documents=[]
    # chunk_size: int = 1000, chunk_overlap: int = 200

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(books)
    fixed_docs = [
    Document(page_content=safe_text(doc.page_content), metadata=doc.metadata)
    for doc in all_splits
]

    # for book in books:
    #     chunked_book = chunk_book(book)
        #vector_store = get_db_collection()

        # embeddings = embed_documents(chunked_book)
        # ids = list(range(next_id, next_id + len(chunked_book)))
        # ids = [f"document_{id}" for id in ids]
        # collection.add(
        #     embeddings=embeddings,
        #     ids=ids,
        #     documents=chunked_book,
        # )
        # next_id += len(chunked_book)
    vector_store.add_documents(documents=fixed_docs)
    print(f"Total documents in collection: {len(all_splits)}")




def main():
    collection = initialize_db(
        persist_directory=VECTOR_DB_DIR,
        collection_name="books",
        delete_existing=True,
    )
    books = load_all_books()
    insert_books(collection, books)

    # print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
