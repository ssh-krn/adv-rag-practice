from chromadb.api.types import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from flashrank import Ranker, RerankRequest
import re

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
    collection_name="ml-foundation",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./cache")

def retriever_node(state: AppState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

def generator_node(state: AppState):

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}

def context_compresser_node(state: AppState):
    compressed_docs = []

    for doc in state["documents"]:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", doc.page_content)
            if s.strip()
        ]
    
        if len(sentences) <= 2:
            compressed_docs.append(doc)
            continue
    
        passages = [{"id": i, "text": sentence} for i, sentence in enumerate(sentences)]

        rerank_request = RerankRequest(query=state["question"], passages=passages)
        results = ranker.rerank(rerank_request)

        top_two = sorted(results[:2], key=lambda item: item["id"])
        compressed_text = " ".join(item["text"] for item in top_two)

    return {"compressed_docs": compressed_text}

builder = StateGraph(AppState)
builder.add_node("retrieve", retriever_node)
builder.add_node("compress", context_compresser_node)
builder.add_node("generate", generator_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "compress")
builder.add_edge("compress", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
result = graph.invoke({"question": "Explain overfitting and regularization."})
print(result["answer"])