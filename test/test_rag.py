# test_rag.py
from research_rabbit.rag import RAGManager

def test_vector_store():
    # Initialize RAG manager
    rag = RAGManager()
    
    # Test uploading documents
    docs_added = rag.upload_directory("/Users/kishorereddy/Documents/amaravati_chamber_rag_store")
    print(f"Number of document chunks added: {docs_added}")
    
    # Test retrieval with a sample query
    if docs_added > 0:
        test_query = "Write a query related to your documents' content"
        results = rag.retrieve_relevant_context(test_query)
        
        print("\nRetrieved documents:")
        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(doc.page_content)

if __name__ == "__main__":
    test_vector_store()