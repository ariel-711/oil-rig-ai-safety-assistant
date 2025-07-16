import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DOCS_PATH = "Oil Rig AI Safety Assistant/safety_docs"
VECTOR_STORE_PATH = "Oil Rig AI Safety Assistant/vector_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A fast and effective model

def build_vector_store():
    """Loads docs, splits them, creates embeddings, and saves to a vector store."""
    print("🚀 Starting to build the vector store...")

    # 1. Load documents from the specified directory
    print(f"📂 Loading documents from '{DOCS_PATH}'...")
    loader = PyPDFDirectoryLoader(DOCS_PATH)
    documents = loader.load()
    if not documents:
        print("❌ No documents found. Please add your safety PDFs to the 'safety_docs' folder.")
        return

    # 2. Split documents into smaller chunks
    print("✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(docs_chunks)} document chunks.")

    # 3. Create embeddings for the document chunks
    print(f"🧠 Creating embeddings using '{EMBEDDING_MODEL}'...")
    # This will download the model from Hugging Face the first time it runs
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 4. Create a FAISS vector store and save it locally
    print("💾 Creating and saving the vector store...")
    vector_store = FAISS.from_documents(docs_chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

    print(f"✅ Vector store built successfully and saved at '{VECTOR_STORE_PATH}'.")

if __name__ == "__main__":
    build_vector_store()