import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

import streamlit as st
import pkg_resources
import langchain
import faiss
import torch

st.sidebar.title("📦 Version Diagnostics")

def show_versions():
    st.sidebar.write("✅ **Installed Package Versions**")
    required = [
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-openai",
        "sentence-transformers",
        "faiss-cpu",
        "streamlit",
        "pydantic",
        "openai",
        "torch",
        "python-dotenv"
    ]
    for pkg in required:
        try:
            version = pkg_resources.get_distribution(pkg).version
            st.sidebar.write(f"- **{pkg}**: {version}")
        except Exception:
            st.sidebar.write(f"- **{pkg}**: ❌ Not Installed")

    st.sidebar.write("\n**FAISS Version:**", faiss.__version__)
    st.sidebar.write("**Torch Version:**", torch.__version__)
    st.sidebar.write("**LangChain Version:**", langchain.__version__)

show_versions()


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Medical AI Assistant")
st.title("🩺 Medical AI Assistant")
st.write("Describe your symptoms, and I’ll suggest possible conditions and treatments.")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

query = st.text_area("🩹 Enter your symptoms:")

if st.button("Diagnose"):
    if query.strip():
        with st.spinner("Analyzing symptoms..."):
            response = qa_chain.invoke({"query": query})
            st.success(response["result"])
    else:
        st.warning("Please enter symptoms first.")
