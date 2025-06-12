import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def create_vector_store():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder_path = os.path.join(current_dir, "pdf")
    
    # Load and process documents
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    new_docs = splitter.split_documents(documents)
    
    # Create and return vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(new_docs, embeddings)
    return db.as_retriever(search_kwargs={"k": 3})