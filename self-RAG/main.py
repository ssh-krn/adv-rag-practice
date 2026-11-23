from flashrank import Ranker, RerankRequest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import re


load_dotenv()

class BatchRelevance(BaseModel):
    reasoning: str = Field(
        description="Brief 1-sentence justification explaining which chunks are relevant to the question."
    )
    relevant_indices: list[int] = Field(
        description="List of 0-based indices of chunks that contain facts directly useful for answering the question. Return [] if none are relevant."
    )

class GradeHallucination(BaseModel):
    reasoning: str = Field(
        description="Brief 1-sentence explanation identifying any unsupported claims or confirming all facts appear in the retrieved context."
    )
    grade: Literal["grounded", "ungrounded"] = Field(
        description=(
            "Evaluate whether the generated answer is strictly supported by the retrieved context:\n"
            "- 'grounded': Every factual claim in the answer is directly supported by or inferred from the retrieved context.\n"
            "- 'ungrounded': The answer introduces external facts, assumptions, or claims not present in the retrieved context."
        )
    )

class QueryRewrite(BaseModel):
    reasoning: str = Field(description="Why the initial retrieval missed.")
    vector_query: str = Field(description="Keyword-rich query for textbook vector search.")

class AppState(TypedDict):
    question: str
    rewrite_reasoning: str
    documents: list[Document]
    doc_grade: str
    ans_grade: str
    retry_count: int
    gen_retry_count: int
    document_reasoning: str
    answer_reasoning: str
    vector_query: str
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

relevance_grader_llm = llm.with_structured_output(BatchRelevance)
hallucination_grader_llm = llm.with_structured_output(GradeHallucination)
query_rewriter = llm.with_structured_output(QueryRewrite)

def retriever_node(state: AppState):
    query = state.get("vector_query") or state["question"]
    docs = retriever.invoke(query)
    return {"documents": docs}

def reranker_node(state: AppState):
    docs = state["documents"]

    sentences = []
    for doc in state["documents"]:
        sentences.extend(
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", doc.page_content)
            if s.strip()
        )

    if not sentences:
        return {"documents": docs}

    passages = [{"id": i, "text": sentence} for i, sentence in enumerate(sentences)]

    rerank_request = RerankRequest(query=state["question"], passages=passages)
    results = ranker.rerank(rerank_request)

    top_two = sorted(results[:2], key=lambda item: item["id"])
    compressed_text = " ".join(item["text"] for item in top_two)

    return {"documents": [Document(page_content=compressed_text)]}

def grade_document_node(state: AppState):
    question = state["question"]
    docs = state["documents"]

    if not docs:
        return {"documents": [], "doc_grade": "irrelevant", "document_reasoning": "No documents retrieved."}

    numbered_chunks = "\n\n".join(
            f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)
        )

    result: BatchRelevance = relevance_grader_llm.invoke([
            SystemMessage(
                content=(
                    "You are a strict relevance grader. Review the numbered chunks and return the indices "
                    "of chunks that contain facts useful for answering the user's question."
                )
            ),
            HumanMessage(
                content=f"User Question: {question}\n\nRetrieved Chunks:\n{numbered_chunks}"
            ),
        ])

    filtered_docs = [
            docs[i] for i in result.relevant_indices if 0 <= i < len(docs)
        ]

    return {
        "documents": filtered_docs,
        "document_reasoning": result.reasoning,
        "doc_grade": "relevant" if filtered_docs else "irrelevant",
    }

def grade_answer_node(state: AppState):
    answer = state["answer"]
    context = "\n\n".join(doc.page_content for doc in state.get("documents", []))

    hallucination_res: GradeHallucination = hallucination_grader_llm.invoke([
            SystemMessage(
                content=(
                    "You are a strict hallucination grader. Verify whether every claim in the LLM's answer "
                    "is directly supported by the retrieved context. Classify as 'grounded' or 'ungrounded'."
                )
            ),
            HumanMessage(
                content=f"Retrieved Context:\n{context}\n\nLLM Answer:\n{answer}"
            ),
        ])

    if hallucination_res.grade == "ungrounded":
        return {
            "ans_grade": "ungrounded",
            "answer_reasoning": hallucination_res.reasoning,  
        }

    return {"ans_grade": hallucination_res.grade, "answer_reasoning": hallucination_res.reasoning}

def route_after_document_grading(state: AppState):
    grade = state["doc_grade"]
    retries = state.get("retry_count", 0)

    if grade == "relevant":
        return "generate"
    else:
        if retries < 2:
            return "rewrite_query"
        return "generate"

def route_after_answer_grading(state: AppState):
    if state["ans_grade"] == "ungrounded" and state.get("gen_retry_count", 0) < 2:
        return "generate"
    return "end"

def query_rewriter_node(state: AppState):
    messages = [
        SystemMessage(
            content=(
                "Convert the user's question into a concise, high-signal vector search query (3–8 words). "
                "Strip conversational filler, keep core technical terms, and ensure the query is completely standalone."
            )
        ),
        HumanMessage(content=f"User question: {state.get("vector_query") or state["question"]}"),
    ]

    result = query_rewriter.invoke(messages)
    return {
        "vector_query": result.vector_query,
        "rewrite_reasoning": result.reasoning,
        "retry_count": state.get("retry_count", 0) + 1
    }

def generator_node(state: AppState):

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content, "gen_retry_count": state.get("gen_retry_count", 0) + 1}

builder = StateGraph(AppState)
builder.add_node("retrieve", retriever_node)
builder.add_node("rerank", reranker_node)
builder.add_node("grade_documents", grade_document_node)
builder.add_node("grade_answer", grade_answer_node)
builder.add_node("rewrite_query", query_rewriter_node)
builder.add_node("generate", generator_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "grade_documents")

builder.add_conditional_edges(
    "grade_documents",
    route_after_document_grading,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
    },
)

builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate", "grade_answer")

builder.add_conditional_edges(
    "grade_answer",
    route_after_answer_grading,
    {"generate": "generate", "end": END},
)

graph = builder.compile()
result = graph.invoke({"question": "Explain overfitting and regularization."})
print(result["answer"])