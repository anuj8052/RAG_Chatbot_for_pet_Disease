import streamlit as st 
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# configurations

DATA_PATH = "vet_data"
CHROMA_DB_PATH = "vet_chroma_db"
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
LLM_MODEL_NAME = "llama3.2"

# helper functions

@st.cache_resource
def load_and_index_data():
    """Loads data, splits it, creates embeddings, and stores them in ChromaDB."""
    print("Loading data...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls = TextLoader, show_progress=True, use_multithreading= True)
    documents = loader.load()
    if not documents:
        st.error("No documents found in the specified {DATA_PATH} directory. Please add some text files.")
        
        return None, None
    print(f"Loaded {len(documents)} documents.")
    
    print("Splitting documents..")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    
    print("initializing the embeddings")
    
    embedding_function = SentenceTransformerEmbeddings(model_name = EMBEDDING_MODEL_NAME)
    print("Creating a Chroma vector store....")
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedding_function,
        persist_directory=CHROMA_DB_PATH
    )
    print("Vector store created and data indexed.")
    return vectorstore, embedding_function

@st.cache_resource
def get_llm():
    """Return a configured LLM Model"""
    
    print("Loading LLM model..")
    llm = ChatOllama(
        model = LLM_MODEL_NAME,
        temperature= 0.3
    )
    
    return llm

# application logic

st.set_page_config(page_title = "Omelo Vet Chatbot", page_icon = "🐶", layout = "wide")
st.title("Omelo Vet Chatbot 🐶")
st.caption("Ask questions about your pet's health and get answers from our vet database.")

# load data and initialize vector store and LLM
vectorstore, embedding_function = load_and_index_data()
llm = get_llm()

if vectorstore and llm:
    retriever = vectorstore.as_retriever(
        search_kwargs = {
            "k": 5, # Number of documents to retrieve
            "filter": {"source": "vet_data"}  # Filter to ensure we only get relevant documents
        }
    )
    
    prompt_template = """
    You are a helpful assistant for veterinary information.
    Answer the question based ONLY on the following context.
    If the information is not in the context, say "I don't have information on that topic in my current knowledge base."
    Be concise and helpful.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    
    # create the rag chain
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser(keep_whitespace=True, keep_newlines=True, keep_emoji=True, keep_links=True)
    )
    
    # streamlit input and output
    
    if "message" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Hi! How can I help you with your pet health questions today?"}]

        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    # user input
    if user_query := st.chat_input("Ask a question about your pet's health..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query) 
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # For debugging: show retrieved context
                relevant_docs = retriever.invoke(user_query)
                st.write("Retrieved context:")
                for i, doc in enumerate(relevant_docs):
                    st.text_area(f"Doc {i+1}", doc.page_content, height=100, key=f"doc_{i}")
                    
                response = rag_chain.invoke(user_query)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Could not initialize the chatbot. Please check data path and Ollama setup.")

st.sidebar.header("About")
st.sidebar.info(
    "This chatbot uses Retrieval Augmented Generation (RAG) "
    "with Llama 3.1 8B to answer questions based on custom veterinary documents. "
    "The information provided is not a substitute for professional veterinary advice."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")
st.sidebar.markdown(f"Documents loaded from: `{DATA_PATH}`")
st.sidebar.subheader("Models Used")
st.sidebar.markdown(f"- LLM: `{LLM_MODEL_NAME}` (via Ollama)")
st.sidebar.markdown(f"- Embedding: `{EMBEDDING_MODEL_NAME}`")
