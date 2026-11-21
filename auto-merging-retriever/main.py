from chromadb.api.types import Document
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from langchain_classic.storage._lc_store import create_kv_docstore
from distro.distro import TypedDict
from dotenv import load_dotenv
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import LocalFileStore

load_dotenv()

class AppState(TypedDict):
    question: str
    documents: list[Document]
    compressed_docs: list[Document]
    answer: str

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name="ml-child-chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

docstore = create_kv_docstore(LocalFileStore("./parent_docstore"))

auto_merge_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=docstore,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200),
    search_kwargs={"k": 4},
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

def retriever_node(state: AppState) -> dict:
    parent_docs = auto_merge_retriever.invoke(state["question"])
    return {"documents": parent_docs}

def generator_node(state: AppState):

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}

builder = StateGraph(AppState)
builder.add_node("retrieve", retriever_node)
builder.add_node("generate", generator_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
result = graph.invoke({"question": "Explain overfitting and regularization."})
print(result["answer"])