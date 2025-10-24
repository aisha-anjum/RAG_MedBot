import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA  # ✅ Correct import for v1.0+


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Optional: Check if key loaded
print("API Key loaded:", bool(os.getenv("OPENAI_API_KEY")))

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
)


# -----------------------------
# Streamlit UI setup
# -----------------------------
st.set_page_config(page_title="Medical AI Assistant", layout="centered")
st.title("🩺 Medical AI Assistant")
st.write("Describe your symptoms, and I’ll suggest possible conditions and treatments.")

# -----------------------------
# Load FAISS index (cached)
# -----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# -----------------------------
# Connect to LLM (OpenAI)
# -----------------------------
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

# -----------------------------
# Build the RAG chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

# -----------------------------
# User input and response
# -----------------------------
query = st.text_area("🩹 Enter your symptoms:")

if st.button("Diagnose"):
    if query.strip():
        with st.spinner("Analyzing symptoms..."):
            response = qa_chain.invoke({"query": query})
            st.success(response["result"])
    else:
        st.warning("Please describe your symptoms first.")
