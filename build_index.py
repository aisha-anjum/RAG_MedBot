import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Paths
pdf_path = r"D:\Python\RAG_MedBot\standard-treatment-guidelines.pdf"
index_path = "faiss_index"

print("🔹 Loading new PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print("🔹 Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

print("🔹 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Check if FAISS index already exists
if os.path.exists(index_path):
    print("🔹 Loading existing FAISS index...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("🔹 Adding new data to existing index...")
    vectorstore.add_documents(texts)
else:
    print("🔹 No index found — creating a new one...")
    vectorstore = FAISS.from_documents(texts, embeddings)

print("🔹 Saving updated FAISS index...")
vectorstore.save_local(index_path)

print("✅ Update complete! Your knowledge base now contains old + new PDFs ✅")
