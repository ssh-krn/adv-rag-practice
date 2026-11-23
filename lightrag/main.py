import json
import networkx as nx
from typing import Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from flashrank import Ranker, RerankRequest

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

GRAPH_PATH = "./lightrag_graph.gml"
CHROMA_DIR = "./chroma_db"


# ==========================================
# 1. Structured Output Schemas
# ==========================================
class DualKeywords(BaseModel):
    low_level_keywords: list[str] = Field(
        description="Specific entities, algorithms, formulas, or proper nouns from the query."
    )
    high_level_keywords: list[str] = Field(
        description="Abstract themes, mechanisms, comparisons, or broader relationships."
    )


class BatchRelevance(BaseModel):
    reasoning: str = Field(
        description="Brief 1-sentence justification explaining which context items are relevant."
    )
    relevant_indices: list[int] = Field(
        description="0-based indices of context blocks useful for answering the question. Return [] if none apply."
    )


class GradeHallucination(BaseModel):
    reasoning: str = Field(
        description="1-sentence explanation identifying any unsupported claims or confirming groundedness."
    )
    grade: Literal["grounded", "ungrounded"] = Field(
        description="'grounded' if every claim is supported by the context; 'ungrounded' otherwise."
    )


class QueryRewrite(BaseModel):
    reasoning: str = Field(description="Why the previous graph + vector retrieval missed.")
    rewritten_query: str = Field(description="Standalone, technical query (4-10 words) for retry.")


# ==========================================
# 2. Load Persisted Graph, Chroma & Models
# ==========================================
graph_db = nx.read_gml(GRAPH_PATH)
print(f"Loaded Knowledge Graph: {graph_db.number_of_nodes()} nodes, {graph_db.number_of_edges()} edges")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

entity_store = Chroma(
    collection_name="lightrag_entities",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
relation_store = Chroma(
    collection_name="lightrag_relations",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
chunk_store = Chroma(
    collection_name="lightrag_chunks",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./cache")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

keyword_llm = llm.with_structured_output(DualKeywords)
relevance_llm = llm.with_structured_output(BatchRelevance)
hallucination_llm = llm.with_structured_output(GradeHallucination)
rewriter_llm = llm.with_structured_output(QueryRewrite)


# ==========================================
# 3. LangGraph State Definition
# ==========================================
class HybridLightRAGState(TypedDict):
    question: str
    active_query: str
    low_level_keywords: list[str]
    high_level_keywords: list[str]
    context_blocks: list[Document]
    doc_grade: str
    document_reasoning: str
    ans_grade: str
    answer_reasoning: str
    retry_count: int
    gen_retry_count: int
    answer: str


# ==========================================
# 4. Graph Nodes
# ==========================================
def extract_keywords_node(state: HybridLightRAGState) -> dict:
    query = state.get("active_query") or state["question"]
    res: DualKeywords = keyword_llm.invoke([
        SystemMessage(
            content=(
                "Split the user's question into low-level keywords (specific entities, algorithms, terms) "
                "and high-level keywords (abstract themes, mechanisms, relationships)."
            )
        ),
        HumanMessage(content=query),
    ])
    return {
        "active_query": query,
        "low_level_keywords": res.low_level_keywords,
        "high_level_keywords": res.high_level_keywords,
    }


def hybrid_lightrag_retrieve_node(state: HybridLightRAGState) -> dict:
    """
    Executes LightRAG's dual-level retrieval (entities + 1-hop subgraph + relations)
    combined with FlashRank-reranked textbook chunks.
    """
    query = state["active_query"]

    # --- A. Low-Level Retrieval: Entity Vectors + Degree-Ranked 1-Hop Subgraph ---
    ll_query = ", ".join(state["low_level_keywords"]) or query
    matched_entities = entity_store.similarity_search(ll_query, k=5)

    # Sort matched entities by node degree (central textbook concepts first)
    ranked_entities = sorted(
        matched_entities,
        key=lambda d: graph_db.degree(d.metadata.get("entityName", ""))
        if graph_db.has_node(d.metadata.get("entityName", ""))
        else 0,
        reverse=True,
    )

    entity_lines = []
    linked_chunk_ids = set()

    for doc in ranked_entities[:4]:
        ent_name = doc.metadata.get("entityName", "")
        entity_lines.append(f"[Entity] {doc.page_content}")

        if graph_db.has_node(ent_name):
            raw_ids = graph_db.nodes[ent_name].get("chunkIds", "[]")
            linked_chunk_ids.update(json.loads(raw_ids))

            # Cap outgoing/incoming 1-hop neighbors to avoid hub explosion
            out_edges = list(graph_db.out_edges(ent_name, data=True))[:3]
            in_edges = list(graph_db.in_edges(ent_name, data=True))[:3]
            for u, v, data in out_edges + in_edges:
                entity_lines.append(f"  -> [1-Hop Edge] {u} -> {v}: {data.get('description', '')}")

    # --- B. High-Level Retrieval: Thematic Relationship Vectors ---
    hl_query = ", ".join(state["high_level_keywords"]) or query
    matched_relations = relation_store.similarity_search(hl_query, k=5)
    relation_lines = [f"[Relation] {doc.page_content}" for doc in matched_relations]

    # --- C. Grounding Chunks (Graph-Linked + Direct Similarity) + FlashRank ---
    candidate_texts = []
    if linked_chunk_ids:
        fetched = chunk_store.get(ids=list(linked_chunk_ids)[:8])
        candidate_texts.extend(fetched.get("documents", []))

    direct_chunks = chunk_store.similarity_search(query, k=5)
    candidate_texts.extend(doc.page_content for doc in direct_chunks)

    unique_candidates = list(dict.fromkeys(t for t in candidate_texts if t.strip()))
    reranked_chunks = []
    if unique_candidates:
        passages = [{"id": i, "text": txt} for i, txt in enumerate(unique_candidates)]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        reranked_chunks = [item["text"] for item in ranked[:3]]

    # Package into structured Document blocks for single-call BatchRelevance grading
    blocks: list[Document] = []
    if entity_lines:
        blocks.append(
            Document(
                page_content="Knowledge Graph Entities & 1-Hop Neighborhood:\n" + "\n".join(dict.fromkeys(entity_lines)),
                metadata={"source": "kg_entities"},
            )
        )
    if relation_lines:
        blocks.append(
            Document(
                page_content="Knowledge Graph High-Level Relationships:\n" + "\n".join(dict.fromkeys(relation_lines)),
                metadata={"source": "kg_relations"},
            )
        )
    for idx, chunk_txt in enumerate(reranked_chunks):
        blocks.append(
            Document(
                page_content=f"Textbook Passage {idx + 1}:\n{chunk_txt}",
                metadata={"source": "vector_chunk"},
            )
        )

    return {"context_blocks": blocks}


def grade_context_node(state: HybridLightRAGState) -> dict:
    blocks = state.get("context_blocks", [])
    if not blocks:
        return {"context_blocks": [], "doc_grade": "irrelevant", "document_reasoning": "No graph or chunk hits."}

    numbered = "\n\n".join(f"[{i}]\n{doc.page_content}" for i, doc in enumerate(blocks))
    res: BatchRelevance = relevance_llm.invoke([
        SystemMessage(
            content=(
                "You are a strict relevance grader. Return the 0-based indices of context blocks "
                "that contain facts useful for answering the user's question."
            )
        ),
        HumanMessage(content=f"Question: {state['question']}\n\nContext Blocks:\n{numbered}"),
    ])

    filtered = [blocks[i] for i in res.relevant_indices if 0 <= i < len(blocks)]
    return {
        "context_blocks": filtered,
        "document_reasoning": res.reasoning,
        "doc_grade": "relevant" if filtered else "irrelevant",
    }


def rewrite_query_node(state: HybridLightRAGState) -> dict:
    res: QueryRewrite = rewriter_llm.invoke([
        SystemMessage(
            content=(
                "Rewrite the query into a concise, technical search query (4-10 words) using "
                "canonical textbook terminology to improve Knowledge Graph and vector recall."
            )
        ),
        HumanMessage(content=f"Original question: {state['question']}\nFailed query: {state['active_query']}"),
    ])
    return {
        "active_query": res.rewritten_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def generate_node(state: HybridLightRAGState) -> dict:
    context = "\n\n---\n\n".join(doc.page_content for doc in state["context_blocks"])
    prompt = (
        "Synthesize a clear, technically accurate answer using ONLY the Knowledge Graph entities, "
        "relationships, and textbook passages below.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    )

    if state.get("ans_grade") == "ungrounded" and state.get("answer_reasoning"):
        prompt += (
            f"\n\nCritique of previous draft: {state['answer_reasoning']}. "
            "Rewrite strictly using ONLY facts present in the context."
        )

    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "gen_retry_count": state.get("gen_retry_count", 0) + 1,
    }


def grade_answer_node(state: HybridLightRAGState) -> dict:
    context = "\n\n".join(doc.page_content for doc in state.get("context_blocks", []))
    res: GradeHallucination = hallucination_llm.invoke([
        SystemMessage(
            content=(
                "Verify whether every factual claim in the answer is directly supported by the "
                "retrieved graph and textbook context. Classify as 'grounded' or 'ungrounded'."
            )
        ),
        HumanMessage(content=f"Context:\n{context}\n\nAnswer:\n{state['answer']}"),
    ])
    return {
        "ans_grade": res.grade,
        "answer_reasoning": res.reasoning,
    }


# ==========================================
# 5. Conditional Routers & Graph Wiring
# ==========================================
def route_after_context_grading(state: HybridLightRAGState) -> str:
    if state["doc_grade"] == "relevant":
        return "generate"
    if state.get("retry_count", 0) < 2:
        return "rewrite_query"
    return "generate"


def route_after_answer_grading(state: HybridLightRAGState) -> str:
    if state["ans_grade"] == "ungrounded" and state.get("gen_retry_count", 0) < 2:
        return "generate"
    return "end"


builder = StateGraph(HybridLightRAGState)
builder.add_node("extract_keywords", extract_keywords_node)
builder.add_node("hybrid_retrieve", hybrid_lightrag_retrieve_node)
builder.add_node("grade_context", grade_context_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("generate", generate_node)
builder.add_node("grade_answer", grade_answer_node)

builder.add_edge(START, "extract_keywords")
builder.add_edge("extract_keywords", "hybrid_retrieve")
builder.add_edge("hybrid_retrieve", "grade_context")

builder.add_conditional_edges(
    "grade_context",
    route_after_context_grading,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
builder.add_edge("rewrite_query", "extract_keywords")
builder.add_edge("generate", "grade_answer")

builder.add_conditional_edges(
    "grade_answer",
    route_after_answer_grading,
    {"generate": "generate", "end": END},
)

lightrag_app = builder.compile()


# ==========================================
# 6. Run & Inspect Node Execution
# ==========================================
if __name__ == "__main__":
    question = "How do L1 and L2 regularization address overfitting differently?"

    for event in lightrag_app.stream({"question": question}, stream_mode="updates"):
        for node_name, output in event.items():
            print(f"\n--- Node: {node_name} ---")
            if node_name == "extract_keywords":
                print("Low-level:", output["low_level_keywords"])
                print("High-level:", output["high_level_keywords"])
            elif node_name == "grade_context":
                print(f"Grade: {output['doc_grade']} | Kept blocks: {len(output['context_blocks'])}")
                print(f"Reasoning: {output['document_reasoning']}")
            elif node_name == "grade_answer":
                print(f"Groundedness: {output['ans_grade']} | {output['answer_reasoning']}")
            elif node_name == "generate":
                print("\nFinal Answer:\n", output["answer"])