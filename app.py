import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Medical AI Assistant")
st.title("🩺 Medical AI Assistant")
st.write("Describe your symptoms, and I’ll suggest possible conditions and treatments.")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)



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
