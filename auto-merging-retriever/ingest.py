from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore

raw_docs = PyMuPDFLoader("./data/ml-foundation.pdf").load()

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = Chroma(
    collection_name="ml-child-chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

fs = LocalFileStore("./parent_docstore")
docstore = create_kv_docstore(fs)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 4},
)

retriever.add_documents(raw_docs)