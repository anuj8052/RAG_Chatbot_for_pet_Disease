# app.py

import streamlit as st
import config # Import configurations
from llm_services import (
    get_llm,
    load_and_get_retriever,
    build_rag_chain,
    get_direct_llm_response
)

# --- Page Configuration ---
st.set_page_config(page_title="Omelo Vet Chatbot", page_icon="🐶", layout="wide")
st.title("Omelo Vet Chatbot 🐶")
st.caption("Ask questions about your pet's health. Toggle RAG mode in the sidebar.")

# --- Sidebar for Mode Selection and Info ---
st.sidebar.header("Chatbot Mode")
rag_enabled_ui = st.sidebar.toggle(
    "Enable RAG (Retrieval Augmented Generation)",
    value=config.DEFAULT_RAG_ENABLED # Default from config
)

st.sidebar.header("About")
st.sidebar.info(
    "This chatbot can operate in two modes:\n"
    "1. **RAG Enabled**: Answers are based on custom veterinary documents.\n"
    "2. **RAG Disabled**: Answers come directly from the LLM's general knowledge.\n\n"
    "The information provided is not a substitute for professional veterinary advice."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Current Configuration")
st.sidebar.markdown(f"- **LLM Model**: `{config.LLM_MODEL_NAME}`")
if rag_enabled_ui:
    st.sidebar.markdown(f"- **Data Source**: `{config.DATA_PATH}`")
    st.sidebar.markdown(f"- **Embedding Model**: `{config.EMBEDDING_MODEL_NAME}`")
    st.sidebar.markdown(f"- **Retriever K**: `{config.RETRIEVER_K}`")


# --- Initialize LLM ---
# This is always needed, regardless of RAG mode
llm = get_llm()
if not llm:
    st.error("Failed to initialize the Language Model. Please check your Ollama setup and model name in config.")
    st.stop()

# --- Conditional RAG Setup ---
retriever = None
rag_chain = None

if rag_enabled_ui:
    with st.spinner("Loading knowledge base for RAG..."):
        retriever = load_and_get_retriever()
    if retriever:
        with st.spinner("Building RAG chain..."):
            rag_chain = build_rag_chain(llm, retriever) # Pass llm and retriever
        if not rag_chain:
            st.warning("Could not build RAG chain. Will use direct LLM interaction.")
            rag_enabled_ui = False # Fallback to direct LLM
    else:
        st.warning(f"Could not load retriever. RAG mode disabled. Check data in '{config.DATA_PATH}'.")
        rag_enabled_ui = False # Fallback to direct LLM

# # --- Chat UI and Logic ---
# if "messages" not in st.session_state:
#     st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your pet health questions today?"}]

# # Display chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User input
# if user_query := st.chat_input("Ask a question..."):
#     st.session_state.messages.append({"role": "user", "content": user_query})
#     with st.chat_message("user"):
#         st.markdown(user_query)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = ""
#             if rag_enabled_ui and rag_chain:
#                 st.write(f"Mode: RAG (Retrieval from `{config.DATA_PATH}`)") # Indicate mode
#                 if config.DEBUG_SHOW_RETRIEVED_DOCS and retriever:
#                     try:
#                         relevant_docs = retriever.invoke(user_query)
#                         with st.expander("Retrieved Context (for RAG debug)"):
#                             for i, doc in enumerate(relevant_docs):
#                                 st.text_area(f"Doc {i+1} (Source: {doc.metadata.get('source', 'N/A')})",
#                                              doc.page_content, height=100, key=f"doc_{i}")
#                     except Exception as e:
#                         st.warning(f"Could not retrieve documents for RAG debug: {e}")

#                 response = rag_chain.invoke(user_query)
#             else:
#                 st.write("Mode: Direct LLM Interaction") # Indicate mode
#                 # Pass only previous messages for history, not the current user query again
#                 history_for_direct_llm = st.session_state.messages[:-1] # Exclude current user query
#                 response = get_direct_llm_response(llm, user_query, history_for_direct_llm)

#             st.markdown(response)
#     st.session_state.messages.append({"role": "assistant", "content": response})

# --- Chat UI and Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your pet health questions today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # More generic spinner message
        with st.spinner("Processing your question..."): # Changed spinner message
            response = ""
            if rag_enabled_ui and rag_chain:
                st.write(f"Mode: RAG (Retrieval from `{config.DATA_PATH}`)") # Indicate mode

                # Only show RAG debug info if RAG is enabled and DEBUG_SHOW_RETRIEVED_DOCS is true
                if config.DEBUG_SHOW_RETRIEVED_DOCS and retriever:
                    try:
                        relevant_docs = retriever.invoke(user_query)
                        with st.expander("Retrieved Context (for RAG debug)"):
                            for i, doc in enumerate(relevant_docs):
                                st.text_area(f"Doc {i+1} (Source: {doc.metadata.get('source', 'N/A')})",
                                             doc.page_content, height=100, key=f"doc_{i}_rag_debug") # Added unique key suffix
                    except Exception as e:
                        st.warning(f"Could not retrieve documents for RAG debug: {e}")

                response = rag_chain.invoke(user_query)
            elif llm: # Ensure llm is available for direct mode
                st.write("Mode: Direct LLM Interaction") # Indicate mode
                # Pass only previous messages for history, not the current user query again
                history_for_direct_llm = st.session_state.messages[:-1] # Exclude current user query
                response = get_direct_llm_response(llm, user_query, history_for_direct_llm)
            else:
                response = "Sorry, the language model is not available right now."


            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

if not llm:
    st.warning("LLM not available. Chatbot cannot function.")