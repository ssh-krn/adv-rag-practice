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
from langchain_tavily import TavilySearch


load_dotenv()

class GradeDocuments(BaseModel):
    reasoning: str = Field(
        description="Brief 1-sentence justification explaining why the retrieved context is relevant, partial/ambiguous, or irrelevant to the question."
    )
    grade: Literal["relevant", "ambiguous", "irrelevant"] = Field(
        description=(
            "Classify the retrieved documents:\n"
            "- 'relevant': Contains clear, direct facts needed to fully answer the question.\n"
            "- 'ambiguous': Partially related or lacks key details; needs both local context and supplementary web search.\n"
            "- 'irrelevant': Off-topic or does not contain information to answer the question."
        )
    )

class VectorRewrite(BaseModel):
    reasoning: str = Field(description="Why the initial retrieval missed.")
    vector_query: str = Field(description="Keyword-rich query for textbook vector search.")

class WebRewrite(BaseModel):
    web_query: str = Field(description="Concise 3-8 word web search query.")

class AppState(TypedDict):
    question: str
    documents: list[Document]
    doc_grade: str
    reasoning: str
    vector_query: str
    web_query: str
    retry_count: int
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

tavily_search = TavilySearch(
    max_results=3,
    topic="general",
    include_answer=False,
)

grader_llm = llm.with_structured_output(GradeDocuments)
vec_query_rewriter = llm.with_structured_output(VectorRewrite)
web_query_rewriter = llm.with_structured_output(WebRewrite)

def retriever_node(state: AppState):
    query = state.get("vector_query") or state["question"]
    docs = retriever.invoke(query)
    return {"documents": docs}

def grade_document_node(state: AppState):
    question = state["question"]
    docs = state["documents"]

    if not docs:
        return {"documents": [], "doc_grade": "irrelevant"}

    batch_messages = [
        [
            SystemMessage(
                content=(
                    "You are a strict relevance grader evaluating retrieved chunks against a user question. "
                    "Classify each chunk as 'relevant', 'ambiguous', or 'irrelevant'."
                )
            ),
            HumanMessage(
                content=f"User Question: {question}\n\nRetrieved Chunk:\n{doc.page_content}"
            ),
        ]
        for doc in docs
    ]

    grades: list[GradeDocuments] = grader_llm.batch(batch_messages)

    filtered_docs = []
    has_ambiguous = False

    for doc, result in zip(docs, grades):
        if result.grade == "relevant":
            filtered_docs.append(doc)
        elif result.grade == "ambiguous":
            filtered_docs.append(doc)
            has_ambiguous = True

    if not filtered_docs:
        overall_grade = "irrelevant"
    elif has_ambiguous:
        overall_grade = "ambiguous"
    else:
        overall_grade = "relevant"

    return {
        "documents": filtered_docs,
        "doc_grade": overall_grade,
    }

def route_after_grading(state: AppState):
    grade = state["doc_grade"]
    retries = state.get("retry_count", 0)

    if grade == "relevant":
        return "generate"
    elif grade == "ambiguous":
        return "rewrite_for_web"
    else: 
        if retries < 2:
            return "rewrite_for_vector"
        return "rewrite_for_web"

def rewrite_vec_node(state: AppState):
    messages = [
        SystemMessage(
            content=(
                "Convert the user's question into a concise, high-signal vector search query (3–8 words). "
                "Strip conversational filler, keep core technical terms, and ensure the query is completely standalone."
            )
        ),
        HumanMessage(content=f"User question: {state.get("vector_query") or state["question"]}"),
    ]

    result = vec_query_rewriter.invoke(messages)
    return {
        "vector_query": result.vector_query,
        "reasoning": result.reasoning,
        "retry_count": state.get("retry_count", 0) + 1,
    }

def rewrite_web_node(state: AppState):
    messages = [
        SystemMessage(
            content=(
                "Convert the user's question into a concise, high-signal web search query (3–8 words). "
                "Strip conversational filler, keep core technical terms, and ensure the query is completely standalone."
            )
        ),
        HumanMessage(content=f"User question: {state['question']}"),
    ]

    query = web_query_rewriter.invoke(messages)
    return {"web_query": query.web_query}

def web_search_node(state: AppState):
    query = state["web_query"]
    raw_response = tavily_search.invoke({"query": query})

    web_docs = [
        Document(
            page_content=item["content"],
            metadata={
                "source": item.get("url", "web"),
                "title": item.get("title", "Tavily Search Result"),
            },
        )
        for item in raw_response.get("results", [])
        if item.get("content")
    ]

    existing_docs = state.get("documents", [])
    return {"documents": existing_docs + web_docs}

def generator_node(state: AppState):

    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {state['question']}"
    response = llm.invoke(prompt)
    return {"answer": response.content}

builder = StateGraph(AppState)
builder.add_node("retrieve", retriever_node)
builder.add_node("grade_documents", grade_document_node)
builder.add_node("rewrite_for_vector", rewrite_vec_node)
builder.add_node("rewrite_for_web", rewrite_web_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generator_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")

builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite_for_vector": "rewrite_for_vector",
        "rewrite_for_web": "rewrite_for_web",
    },
)

builder.add_edge("rewrite_for_vector", "retrieve")
builder.add_edge("rewrite_for_web", "web_search")
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
result = graph.invoke({"question": "Explain overfitting and regularization."})
print(result["answer"])