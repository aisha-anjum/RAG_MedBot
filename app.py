import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ✅ Verify key loaded
if not api_key:
    st.error("OPENAI_API_KEY not found! Please set it in Streamlit Secrets.")
    st.stop()

# ✅ App UI
st.set_page_config(page_title="Medical AI Assistant", layout="centered")
st.title("🩺 Medical AI Assistant")
st.write("Describe your symptoms, and I’ll suggest possible conditions and treatments.")

# ✅ Load Chroma vector database
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ✅ Model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)

# ✅ RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ✅ User Input
query = st.text_area("🩹 Enter your symptoms:")

if st.button("Diagnose"):
    if query.strip():
        with st.spinner("Analyzing symptoms..."):
            response = qa_chain.invoke({"query": query})
            st.success(response["result"])
    else:
        st.warning("Please enter your symptoms.")


