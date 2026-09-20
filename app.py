import os
import tempfile
import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Import retrieval chain helpers
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# Function to fetch active models dynamically from Groq API
def get_groq_models(api_key):
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            # Filter text generation models and sort them
            models = [
                m["id"] for m in data.get("data", []) 
                if not ("whisper" in m["id"] or "guard" in m["id"])
            ]
            return sorted(models)
    except Exception:
        pass
    # Fallback default models if API call fails
    return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]


st.set_page_config(page_title="Multi-Document RAG Assistant", layout="wide")
st.title("📚 Multi-Document RAG Assistant")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    
    selected_model = None
    if api_key:
        available_models = get_groq_models(api_key)
        selected_model = st.selectbox("Select Active Model:", available_models)
    else:
        st.warning("Please enter your Groq API Key to load models.")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    process_btn = st.button("Process Documents")

# Initialize VectorStore Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process PDFs
if process_btn:
    if not api_key:
        st.error("Please enter a valid Groq API Key!")
    elif not uploaded_files:
        st.error("Please upload at least one PDF document!")
    else:
        with st.spinner("Processing uploaded PDFs..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                all_docs.extend(docs)
                os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = Chroma.from_documents(splits, embeddings)
            st.success("Documents processed successfully!")

# Q&A Interface
st.markdown("---")
st.subheader("💬 Ask Questions About Your Documents")
query = st.text_input("Enter your question:")

if query:
    if not api_key:
        st.error("Please provide a Groq API Key in the sidebar.")
    elif st.session_state.vector_store is None:
        st.error("Please upload and process documents first!")
    else:
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name=selected_model,
                temperature=0.2
            )

            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the provided context:
                
                <context>
                {context}
                </context>
                
                Question: {input}"""
            )

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

            with st.spinner("Generating answer..."):
                response = rag_chain.invoke({"input": query})
                st.markdown("### Answer:")
                st.write(response["answer"])

                with st.expander("View Source Context"):
                    for doc in response["context"]:
                        st.write(f"**Source Page:** {doc.metadata.get('page', 'N/A')}")
                        st.write(doc.page_content)
                        st.markdown("---")
        except Exception as e:
            st.error(f"An error occurred: {e}")