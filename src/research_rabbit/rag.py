# /src/research_rabbit/rag.py

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

class RAGManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Initialize vector store
        if os.path.exists(persist_directory):
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )

    def upload_directory(self, directory_path: str, recursive: bool = True):
        """Upload all supported documents from a directory to the vector store"""
        try:
            loader = DirectoryLoader(
                directory_path,
                recursive=recursive,
                show_progress=True
            )
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            self.vector_store.add_documents(splits)
            
            print(f"Successfully uploaded {len(splits)} document chunks from {directory_path}")
            return len(splits)
            
        except Exception as e:
            print(f"Error uploading directory: {str(e)}")
            return 0

    def retrieve_relevant_context(self, query: str, k: int = 4):
        """Retrieve relevant documents for a query"""
        return self.vector_store.similarity_search(query, k=k)