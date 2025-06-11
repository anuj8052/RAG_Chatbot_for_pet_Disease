# llm_services.py

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

import config # Import configurations

# --- LLM Initialization ---
@st.cache_resource
def get_llm():
    """Initializes and returns the LLM based on config."""
    print(f"Initializing LLM: {config.LLM_MODEL_NAME}")
    llm = ChatOllama(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE
    )
    return llm

# --- RAG Specific Functions ---
@st.cache_resource
def load_and_get_retriever():
    """
    Loads data from DATA_PATH, splits it, creates embeddings,
    stores them in ChromaDB, and returns a retriever object.
    Uses configurations from config.py.
    """
    print(f"Loading documents from: {config.DATA_PATH}")
    loader = DirectoryLoader(
        config.DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()

    if not documents:
        st.error(f"No documents found in '{config.DATA_PATH}'. RAG functionality will be limited.")
        return None

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}")
    embedding_function = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    print(f"Creating ChromaDB vector store at: {config.CHROMA_DB_PATH}")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedding_function,
        persist_directory=config.CHROMA_DB_PATH
    )
    print("Vector store created and data indexed.")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.RETRIEVER_K}
    )
    print(f"Retriever configured to fetch top {config.RETRIEVER_K} documents.")
    return retriever

@st.cache_resource
def build_rag_chain(_llm, _retriever): # Underscore to avoid conflict with global llm/retriever
    """Builds and returns the RAG chain."""
    if not _retriever:
        return None

    print("Building RAG chain...")
    prompt = ChatPromptTemplate.from_template(config.RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {"context": _retriever, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    print("RAG chain built.")
    return rag_chain

# --- Direct LLM Interaction Function ---
def get_direct_llm_response(llm, user_query, chat_history_messages):
    """
    Gets a response directly from the LLM, incorporating chat history.
    chat_history_messages should be a list of Langchain HumanMessage/AIMessage objects.
    """
    print("Getting direct LLM response...")
    prompt_messages = [
        ("system", config.DIRECT_LLM_SYSTEM_PROMPT)
    ]
    # Add existing history
    for msg in chat_history_messages:
        if msg["role"] == "user":
            prompt_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            prompt_messages.append(AIMessage(content=msg["content"]))

    # Add current user query
    prompt_messages.append(HumanMessage(content=user_query))

    formatted_prompt = ChatPromptTemplate.from_messages(prompt_messages)

    chain = formatted_prompt | llm | StrOutputParser()
    response = chain.invoke({}) # Invoke with empty dict as context is in messages
    print("Direct LLM response received.")
    return response