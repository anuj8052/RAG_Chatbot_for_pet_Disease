# config.py

# --- Data and Indexing Configuration ---
DATA_PATH = "vet_data"  # Path to your custom data files
CHROMA_DB_PATH = "vet_chroma_db_refactored"  # Directory to store ChromaDB
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2" # Embedding model for RAG

# --- Text Splitting Configuration for RAG ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- LLM Configuration ---
LLM_MODEL_NAME = "llama3.2" # Ensure this model is pulled in Ollama (changed from llama3.2 as 3.1 is more common now)
LLM_TEMPERATURE = 0.3

# --- RAG Retriever Configuration ---
RETRIEVER_K = 3  # Number of relevant documents to retrieve

# --- Prompt Templates ---
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for veterinary information.
Answer the question based ONLY on the following context.
If the information is not in the context, say "I don't have information on that topic in my current knowledge base."
Be concise and helpful.

Context:
{context}

Question: {question}

Answer:
"""

DIRECT_LLM_SYSTEM_PROMPT = """
You are a helpful veterinary assistant. Answer the user's questions about pet health.
If you don't know the answer, say so. Do not make up information.
Keep your answers concise and informative.
"""

# --- Application Behavior ---
# This can be overridden by UI, but sets the default
DEFAULT_RAG_ENABLED = True
DEBUG_SHOW_RETRIEVED_DOCS = True # Set to False for production