import os
import tempfile
import requests
import streamlit as st

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain


# Helper function to dynamically fetch active models from Groq API
def get_groq_models(api_key):
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            # Filter for text models only (excluding audio/guard models)
            models = [
                m["id"] for m in data.get("data", [])
                if not ("whisper" in m["id"] or "guard" in m["id"])
            ]
            if models:
                return sorted(models)
    except Exception:
        pass
    # Fallback options
    return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]


# Page Configuration
st.set_page_config(
    page_title="Multi-Document RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Document RAG Assistant")
st.write("Upload your PDFs in the sidebar, enter your Groq API Key, and ask questions based on their contents.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input("Enter Groq API Key (Optional if secret configured):", type="password")

    # Priority: User input key > Streamlit Secret key
    api_key = user_api_key or st.secrets.get("GROQ_API_KEY", "")

    selected_model = None
    if api_key:
        available_models = get_groq_models(api_key)
        selected_model = st.selectbox("Select Groq Model:", available_models)
    else:
        st.info("💡 Enter your Groq API Key above to load available models.")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )


# --- DOCUMENT PROCESSING ---
def process_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        documents.extend(docs)
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


# --- VECTOR STORE & EMBEDDINGS ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vector_store(splits):
    embeddings = get_embedding_model()
    return Chroma.from_documents(documents=splits, embedding=embeddings)


# --- DOCUMENT PROCESSING TRIGGER ---
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) ready to process.")

    if st.sidebar.button("Process Documents"):
        with st.spinner("Processing documents and generating embeddings..."):
            chunks = process_documents(uploaded_files)
            st.session_state.chunks = chunks

            vectorstore = create_vector_store(chunks)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            st.success(f"Successfully processed {len(chunks)} text chunks into ChromaDB!")


# --- QUESTION ANSWERING CHAIN ---
if "retriever" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask Questions About Your Documents")

    if not api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to ask questions.")
    else:
        user_question = st.text_input("Enter your question:")

        if user_question:
            with st.spinner("Searching documents & generating response..."):
                try:
                    # Initialize ChatGroq with selected active model
                    llm = ChatGroq(
                        groq_api_key=api_key,
                        model_name=selected_model,
                        temperature=0.2
                    )

                    # Prompt template
                    system_prompt = (
                        "You are a helpful and precise assistant for question-answering tasks.\n"
                        "Use the following pieces of retrieved context from uploaded documents to answer "
                        "the question accurately. If you don't know the answer, state clearly that "
                        "the answer is not in the provided documents. Do not invent information.\n\n"
                        "Context:\n{context}"
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    # Build Chains
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

                    # Generate Answer
                    response = rag_chain.invoke({"input": user_question})

                    st.markdown("### 💡 Answer:")
                    st.write(response["answer"])

                    with st.expander("📄 View Source Paragraphs Used for This Answer"):
                        for idx, doc in enumerate(response["context"]):
                            st.markdown(f"**Source Snippet {idx+1}** (Page {doc.metadata.get('page', 0) + 1}):")
                            st.info(doc.page_content)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")