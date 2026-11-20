from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
import re

def embedding_pdf(path):
    loader = PyMuPDFLoader(path)
    raw_docs = loader.load()

    for doc in raw_docs:
        text = doc.page_content
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        doc.page_content = text.strip()

    spilitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = spilitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ml-foundation",
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"}
    )

embedding_pdf("./data/ml-foundation.pdf")