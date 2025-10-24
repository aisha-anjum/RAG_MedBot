import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# ✅ Paths
pdf_path = r"D:\Python\RAG_MedBot\Primary-Healthcare-Standard-Treatment-Guidelines-and-Essential-Medicines-List-8th-Edition-2024.pdf"
db_path = "chroma_db"

print("🔹 Loading PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print("🔹 Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

print("🔹 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("🔹 Creating / Updating Chroma DB...")
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=db_path)
vectorstore.persist()

print("✅ Chroma DB created & saved successfully!")
