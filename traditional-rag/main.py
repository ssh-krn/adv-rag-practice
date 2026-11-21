from chromadb.api.types import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


load_dotenv()

class AppState(TypedDict):
    question: str
    documents: list[Document]
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

def retriever_node(state: AppState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}

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