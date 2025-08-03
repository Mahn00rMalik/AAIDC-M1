import logging
import re
import os
import streamlit as st # Keep streamlit here for st.cache_resource, and st.warning within functions if needed
from dotenv import load_dotenv # Still needed for initial loading if not handled globally
from langchain_groq import ChatGroq

# Import your custom modules
from utils import load_yaml_config
from prompt_builder import build_prompt_from_config
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR # Keep paths here as backend needs them
from vector_db_ingest import get_db_collection, embed_documents

# --- Setup Logging for RAG Core ---
logger = logging.getLogger(__name__)

def setup_logging_nutribot_logic():
    # Only add handler if not already present to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(OUTPUTS_DIR, "nutribot_rag_core.log"))
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.info("RAG Core logging set up.")

# --- Global ChromaDB Collection (or pass it explicitly) ---
# For caching, it's often best to keep st.cache_resource where st is imported
@st.cache_resource
def get_chroma_collection():
    """Caches the ChromaDB collection to avoid re-initializing on every rerun."""
    logger.info("Initializing ChromaDB collection.")
    return get_db_collection(collection_name="gutenberg_books")

# --- Utility function to clean LLM response ---
def clean_llm_response(text: str) -> str:
    """Removes leading Markdown heading hashes and ensures content starts on a new line."""
    match = re.match(r"^(#+)\s*(.*)", text, re.DOTALL)
    if match:
        heading_text = match.group(2).strip()
        return f"\n{heading_text}"
    else:
        return text.strip()

# --- Core RAG Functions ---

def retrieve_relevant_documents_streamlit(
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
    collection=None # Accept collection as an argument
) -> list[str]:
    """
    Query the ChromaDB database with a string query.
    Intermediate st.info/spinner messages are removed.
    """
    if collection is None:
        logger.error("ChromaDB collection not provided to retrieve_relevant_documents_streamlit.")
        return [] # Or raise an error

    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }

    query_embedding = embed_documents([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    if not relevant_results["documents"]:
        # We can keep this st.warning here as it's directly tied to retrieval outcome
        st.warning("No relevant documents found for your query with the current settings.")
        logger.warning(f"No documents found for query: {query}")
    return relevant_results["documents"]


def respond_to_query_streamlit(
    prompt_config: dict,
    query: str,
    llm_model_name: str,
    n_results: int = 5,
    threshold: float = 0.3,
    collection=None # Accept collection as an argument
) -> str:
    """
    Respond to a query using the ChromaDB database.
    """
    if collection is None:
        logger.error("ChromaDB collection not provided to respond_to_query_streamlit.")
        return "An internal error occurred: Database not accessible."

    relevant_documents = retrieve_relevant_documents_streamlit(
        query, n_results=n_results, threshold=threshold, collection=collection
    )

    if not relevant_documents:
        logger.info("No relevant documents, returning fallback message.")
        return "I could not find enough relevant information to answer your question."

    input_data = (
        f"Relevant documents:\n\n{' '.join(relevant_documents)}\n\nUser's question:\n\n{query}"
    )

    assistant_prompt = build_prompt_from_config(
        prompt_config, input_data=input_data
    )
    llm = ChatGroq(model=llm_model_name)
    response = llm.invoke(assistant_prompt)
    logger.info(f"LLM responded to query: {query}")

    return clean_llm_response(response.content)