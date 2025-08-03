import streamlit as st
import os
import logging
from dotenv import load_dotenv
from utils import load_yaml_config
from paths import APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, OUTPUTS_DIR

# Import the core RAG logic from the new file
from nutribot_logic import (
    get_chroma_collection,
    respond_to_query_streamlit, # Keep the streamlit feedback in this function for simplicity
    setup_logging_nutribot_logic # Renamed to avoid confusion with streamlit's own setup
)

# --- Setup Logging (for the Streamlit app specific logs, if any) ---
# Note: Most logging for RAG process will be handled within rag_core.py now
logger = logging.getLogger(__name__)

def setup_logging_app():
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

# --- Load Environment Variables and Global Configurations ---
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize ChromaDB collection (cached)
collection = get_chroma_collection() # This function is now in rag_core.py

# --- Streamlit UI ---

def main():
    setup_logging_app() # Set up logging for the Streamlit UI file
    setup_logging_nutribot_logic() # Set up logging for the RAG core logic

    st.set_page_config(page_title="NutriBot RAG App", layout="wide")
    st.title("📚 NutriBot RAG Assistant")

    st.markdown(
        """
        Welcome to NutriBot! This RAG (Retrieval Augmented Generation) assistant
        uses your query to retrieve relevant information from a document database
        (ChromaDB) and then uses a Large Language Model (Groq) to answer your questions
        based on the retrieved context.
        """
    )

    # Load configurations (cached to avoid reloading on every rerun)
    @st.cache_data
    def get_configs():
        return load_yaml_config(APP_CONFIG_FPATH), load_yaml_config(PROMPT_CONFIG_FPATH)

    app_config, prompt_config = get_configs()
    nutribot_system_prompt = prompt_config["nutribot_system_prompt"]
    llm_model = app_config["llm"]

    # --- Session State Initialization ---
    if "n_results" not in st.session_state:
        st.session_state.n_results = app_config["vectordb"]["n_results"]
    if "threshold" not in st.session_state:
        st.session_state.threshold = app_config["vectordb"]["threshold"]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input_value" not in st.session_state:
        st.session_state.user_input_value = ""

    st.sidebar.header("Configuration")
    st.sidebar.write("Adjust the parameters for document retrieval:")

    with st.sidebar.expander("Retrieval Parameters"):
        new_n_results = st.slider(
            "Number of Results (Top K):",
            min_value=1,
            max_value=20,
            value=st.session_state.n_results,
            key="n_results_slider",
            help="How many top similar documents to retrieve from the database."
        )
        if new_n_results != st.session_state.n_results:
            st.session_state.n_results = new_n_results

        new_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.05,
            key="threshold_slider",
            help="Documents with a similarity distance above this will be filtered out. Lower is more similar."
        )
        if new_threshold != st.session_state.threshold:
            st.session_state.threshold = new_threshold

    st.sidebar.info(f"Current LLM Model: `{llm_model}`")

    # --- Display Chat History (Continuous Feed) ---
    st.header("Conversation")

    chat_container = st.container()

    with chat_container:
        for chat_message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat_message['query'])
            with st.chat_message("assistant"):
                st.markdown(chat_message['response'])

    # --- Chat Input and Button ---
    def clear_text_input_on_change(): # This is just a dummy, the actual clear is in submit_query
        pass

    user_query_input = st.text_input(
        "Your question:",
        key="user_input",
        value=st.session_state.user_input_value,
        on_change=clear_text_input_on_change
    )

    def submit_query():
        query = st.session_state.user_input # Get the value from the keyed text_input
        if query:
            with st.spinner("Processing your request..."):
                try:
                    response_content = respond_to_query_streamlit(
                        prompt_config=nutribot_system_prompt,
                        query=query,
                        llm_model_name=llm_model,
                        n_results=st.session_state.n_results,
                        threshold=st.session_state.threshold,
                        # Pass the global collection object
                        collection=collection
                    )
                    st.session_state.chat_history.append({"query": query, "response": response_content})
                    st.session_state.user_input_value = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.info("Please check your API keys, configuration files, and ensure the ChromaDB is properly initialized.")
        else:
            st.warning("Please enter a question.")

    if st.button("Get Answer"):
        submit_query()

    # Clear history button in sidebar
    if st.session_state.chat_history:
        if st.sidebar.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()