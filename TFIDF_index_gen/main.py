from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import TFIDFRetriever

vector_store = Chroma(
    collection_name="ml-foundation",
    persist_directory="./chroma_db",
)

stored_data = vector_store.get()

docs = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(stored_data["documents"], stored_data["metadatas"])
]

tfidf_retriever = TFIDFRetriever.from_documents(docs, k=4)

tfidf_retriever.save_local("./tfidf_index")