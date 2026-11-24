import re
import difflib
import networkx as nx
from typing import Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

GRAPH_PATH = "./lightrag_graph.gml"

class QueryEntities(BaseModel):
    entities: list[str] = Field(
        description="Key technical concepts, algorithms, or terms from the question to look up in the Knowledge Graph."
    )


class BatchRelevance(BaseModel):
    reasoning: str = Field(
        description="Brief 1-sentence justification explaining which graph subgraphs are relevant."
    )
    relevant_indices: list[int] = Field(
        description="0-based indices of context blocks useful for answering the question. Return [] if none apply."
    )


class GradeHallucination(BaseModel):
    reasoning: str = Field(
        description="1-sentence explanation identifying any unsupported claims or confirming groundedness."
    )
    grade: Literal["grounded", "ungrounded"] = Field(
        description="'grounded' if every claim is supported by the graph context; 'ungrounded' otherwise."
    )


class QueryRewrite(BaseModel):
    reasoning: str = Field(description="Why the previous graph node lookup missed.")
    rewritten_query: str = Field(description="Alternative canonical technical terms (4-10 words) for graph retry.")

graph_db = nx.read_gml(GRAPH_PATH)
print(f"Loaded Pure Knowledge Graph: {graph_db.number_of_nodes()} nodes, {graph_db.number_of_edges()} edges")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

entity_extractor_llm = llm.with_structured_output(QueryEntities)
relevance_llm = llm.with_structured_output(BatchRelevance)
hallucination_llm = llm.with_structured_output(GradeHallucination)
rewriter_llm = llm.with_structured_output(QueryRewrite)

class GraphOnlyRAGState(TypedDict):
    question: str
    active_query: str
    target_entities: list[str]
    context_blocks: list[Document]
    doc_grade: str
    document_reasoning: str
    ans_grade: str
    answer_reasoning: str
    retry_count: int
    gen_retry_count: int
    answer: str


# ==========================================
# 4. Graph-Only Nodes
# ==========================================
def extract_query_entities_node(state: GraphOnlyRAGState) -> dict:
    query = state.get("active_query") or state["question"]
    res: QueryEntities = entity_extractor_llm.invoke([
        SystemMessage(
            content=(
                "Extract 2 to 6 canonical technical entities, algorithms, or concepts from the query "
                "to match against nodes in a Machine Learning Knowledge Graph."
            )
        ),
        HumanMessage(content=query),
    ])
    return {
        "active_query": query,
        "target_entities": res.entities,
    }


def _match_graph_nodes(terms: list[str], top_n: int = 5) -> list[str]:
    """
    Finds entry nodes in NetworkX using exact match, substring containment,
    and fuzzy string similarity (no vector embeddings required).
    """
    all_nodes = list(graph_db.nodes)
    node_lower_map = {n.lower(): n for n in all_nodes}
    scored_nodes: dict[str, float] = {}

    for term in terms:
        t_clean = term.strip().lower()
        if not t_clean:
            continue

        # 1. Exact match
        if t_clean in node_lower_map:
            real_name = node_lower_map[t_clean]
            scored_nodes[real_name] = scored_nodes.get(real_name, 0.0) + 10.0

        # 2. Substring match on node names or descriptions
        for node in all_nodes:
            n_low = node.lower()
            desc_low = graph_db.nodes[node].get("description", "").lower()
            if t_clean in n_low or n_low in t_low_guard(t_clean):
                scored_nodes[node] = scored_nodes.get(node, 0.0) + 5.0
            elif re.search(rf"\b{re.escape(t_clean)}\b", desc_low):
                scored_nodes[node] = scored_nodes.get(node, 0.0) + 2.5

        # 3. Fuzzy match fallback for slight naming variations (e.g., "L2 penalty" vs "L2 Regularization")
        close = difflib.get_close_matches(t_clean, list(node_lower_map.keys()), n=2, cutoff=0.72)
        for match_low in close:
            real_name = node_lower_map[match_low]
            scored_nodes[real_name] = scored_nodes.get(real_name, 0.0) + 3.0

    # Break ties using node degree (more connected textbook hubs rank higher)
    ranked = sorted(
        scored_nodes.items(),
        key=lambda item: (item[1], graph_db.degree(item[0])),
        reverse=True,
    )
    return [node for node, _ in ranked[:top_n]]


def t_low_guard(t: str) -> str:
    return t if len(t) >= 4 else ""


def pure_graph_retrieve_node(state: GraphOnlyRAGState) -> dict:
    """
    Traverses 1-hop and 2-hop neighborhoods in lightrag_graph.gml starting from
    the matched entry nodes—zero vector DB or keyword search used.
    """
    terms = state.get("target_entities") or [state["active_query"]]
    entry_nodes = _match_graph_nodes(terms, top_n=5)

    blocks: list[Document] = []

    for node in entry_nodes:
        node_data = graph_db.nodes[node]
        lines = [
            f"[Central Entity] {node} ({node_data.get('entityType', 'Concept')}): "
            f"{node_data.get('description', 'No description.')}"
        ]

        # 1-Hop Outgoing & Incoming Edges
        out_edges = list(graph_db.out_edges(node, data=True))[:4]
        in_edges = list(graph_db.in_edges(node, data=True))[:4]

        hop1_neighbors = set()
        for u, v, data in out_edges + in_edges:
            kw = data.get("keywords", "")
            desc = data.get("description", "")
            lines.append(f"  -> [1-Hop] {u} --({kw})--> {v}: {desc}")
            hop1_neighbors.add(v if u == node else u)

        # 2-Hop Expansion (follow edges between matched entry nodes or top neighbor)
        for neighbor in list(hop1_neighbors)[:2]:
            if graph_db.has_node(neighbor):
                for u2, v2, data2 in list(graph_db.out_edges(neighbor, data=True))[:2]:
                    if v2 != node:
                        lines.append(
                            f"    => [2-Hop via {neighbor}] {u2} --({data2.get('keywords', '')})--> {v2}: "
                            f"{data2.get('description', '')}"
                        )

        blocks.append(
            Document(
                page_content="\n".join(dict.fromkeys(lines)),
                metadata={"source": "networkx_subgraph", "root_node": node},
            )
        )

    return {"context_blocks": blocks}


def grade_context_node(state: GraphOnlyRAGState) -> dict:
    blocks = state.get("context_blocks", [])
    if not blocks:
        return {"context_blocks": [], "doc_grade": "irrelevant", "document_reasoning": "No matching graph nodes found."}

    numbered = "\n\n".join(f"[{i}]\n{doc.page_content}" for i, doc in enumerate(blocks))
    res: BatchRelevance = relevance_llm.invoke([
        SystemMessage(
            content=(
                "You are a strict relevance grader. Return the 0-based indices of Knowledge Graph subgraphs "
                "that contain facts useful for answering the user's question."
            )
        ),
        HumanMessage(content=f"Question: {state['question']}\n\nGraph Subgraphs:\n{numbered}"),
    ])

    filtered = [blocks[i] for i in res.relevant_indices if 0 <= i < len(blocks)]
    return {
        "context_blocks": filtered,
        "document_reasoning": res.reasoning,
        "doc_grade": "relevant" if filtered else "irrelevant",
    }


def rewrite_query_node(state: GraphOnlyRAGState) -> dict:
    res: QueryRewrite = rewriter_llm.invoke([
        SystemMessage(
            content=(
                "Rewrite the query using canonical, textbook entity names (4-10 words) "
                "so exact or fuzzy node matching in the Knowledge Graph succeeds."
            )
        ),
        HumanMessage(content=f"Original question: {state['question']}\nFailed entities: {state.get('target_entities', [])}"),
    ])
    return {
        "active_query": res.rewritten_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def generate_node(state: GraphOnlyRAGState) -> dict:
    context = "\n\n---\n\n".join(doc.page_content for doc in state["context_blocks"])
    prompt = (
        "Synthesize a clear, technically accurate answer using ONLY the Knowledge Graph entities "
        "and multi-hop relationships below.\n\n"
        f"Knowledge Graph Context:\n{context}\n\nQuestion: {state['question']}"
    )

    if state.get("ans_grade") == "ungrounded" and state.get("answer_reasoning"):
        prompt += (
            f"\n\nCritique of previous draft: {state['answer_reasoning']}. "
            "Rewrite strictly using ONLY facts present in the graph context."
        )

    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "gen_retry_count": state.get("gen_retry_count", 0) + 1,
    }


def grade_answer_node(state: GraphOnlyRAGState) -> dict:
    context = "\n\n".join(doc.page_content for doc in state.get("context_blocks", []))
    res: GradeHallucination = hallucination_llm.invoke([
        SystemMessage(
            content=(
                "Verify whether every factual claim in the answer is directly supported by the "
                "Knowledge Graph context. Classify as 'grounded' or 'ungrounded'."
            )
        ),
        HumanMessage(content=f"Graph Context:\n{context}\n\nAnswer:\n{state['answer']}"),
    ])
    return {
        "ans_grade": res.grade,
        "answer_reasoning": res.reasoning,
    }

def route_after_context_grading(state: GraphOnlyRAGState) -> str:
    if state["doc_grade"] == "relevant":
        return "generate"
    if state.get("retry_count", 0) < 2:
        return "rewrite_query"
    return "generate"


def route_after_answer_grading(state: GraphOnlyRAGState) -> str:
    if state["ans_grade"] == "ungrounded" and state.get("gen_retry_count", 0) < 2:
        return "generate"
    return "end"


builder = StateGraph(GraphOnlyRAGState)
builder.add_node("extract_entities", extract_query_entities_node)
builder.add_node("graph_retrieve", pure_graph_retrieve_node)
builder.add_node("grade_context", grade_context_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("generate", generate_node)
builder.add_node("grade_answer", grade_answer_node)

builder.add_edge(START, "extract_entities")
builder.add_edge("extract_entities", "graph_retrieve")
builder.add_edge("graph_retrieve", "grade_context")

builder.add_conditional_edges(
    "grade_context",
    route_after_context_grading,
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)
builder.add_edge("rewrite_query", "extract_entities")
builder.add_edge("generate", "grade_answer")

builder.add_conditional_edges(
    "grade_answer",
    route_after_answer_grading,
    {"generate": "generate", "end": END},
)

graph_only_app = builder.compile()


if __name__ == "__main__":
    question = "What is Machine Learning?"

    for event in graph_only_app.stream({"question": question}, stream_mode="updates"):
        for node_name, output in event.items():
            print(f"\n--- Node: {node_name} ---")
            if node_name == "extract_entities":
                print("Target Entities:", output["target_entities"])
            elif node_name == "grade_context":
                print(f"Grade: {output['doc_grade']} | Kept subgraphs: {len(output['context_blocks'])}")
                print(f"Reasoning: {output['document_reasoning']}")
            elif node_name == "grade_answer":
                print(f"Groundedness: {output['ans_grade']} | {output['answer_reasoning']}")
            elif node_name == "generate":
                print("\nFinal Answer:\n", output["answer"])