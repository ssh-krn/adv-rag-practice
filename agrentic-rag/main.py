import operator
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from flashrank import Ranker, RerankRequest

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "lightrag_chunks"  # Or "ml-foundation"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
    encode_kwargs={"normalize_embeddings": True},
)

chunk_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./cache")
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

@tool
def search_textbook_chunks(query: str) -> str:
    """
    Search the local ML textbook ChromaDB vector store and rerank passages
    with FlashRank. Use this FIRST for foundational concepts, definitions,
    mathematical derivations, and algorithms.
    """
    docs = chunk_store.similarity_search(query, k=8)
    unique_texts = list(dict.fromkeys(d.page_content.strip() for d in docs if d.page_content.strip()))

    if not unique_texts:
        return "NO_RELEVANT_TEXTBOOK_DATA_FOUND"

    passages = [{"id": i, "text": txt} for i, txt in enumerate(unique_texts)]
    ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    top_chunks = [item["text"] for item in ranked[:3]]

    return "\n\n".join(f"[Textbook Passage {i+1}]\n{c}" for i, c in enumerate(top_chunks))


tavily_client = TavilySearch(max_results=3, topic="general")


@tool
def search_web_tavily(query: str) -> str:
    """
    Search the live web using Tavily for recent developments, modern libraries,
    current benchmarks, or supplementary explanations not found in the local textbook.
    """
    results = tavily_client.invoke({"query": query})
    items = results.get("results", []) if isinstance(results, dict) else results
    if not items:
        return "NO_WEB_RESULTS_FOUND"

    formatted = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            title = item.get("title", "Web Source")
            url = item.get("url", "")
            content = item.get("content", "")
            formatted.append(f"[Web Source {i+1}: {title} ({url})]\n{content}")
        else:
            formatted.append(f"[Web Source {i+1}]\n{item}")

    return "\n\n".join(formatted)


TOOLS = [search_textbook_chunks, search_web_tavily]
agent_llm = llm.bind_tools(TOOLS)

class GradeHallucination(BaseModel):
    reasoning: str = Field(
        description="1-sentence explanation identifying any unsupported claims or confirming all claims appear in tool outputs."
    )
    grade: Literal["grounded", "ungrounded"] = Field(
        description="'grounded' if every factual claim is supported by the gathered tool evidence; 'ungrounded' otherwise."
    )


hallucination_grader_llm = llm.with_structured_output(GradeHallucination)

class AgenticRAGState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], operator.add]
    tool_calls_count: int
    ans_grade: str
    answer_reasoning: str
    gen_retry_count: int
    final_answer: str


AGENT_SYSTEM_PROMPT = """You are an autonomous research assistant backed by two retrieval tools:
1. `search_textbook_chunks`: Searches a local Machine Learning textbook vector store.
2. `search_web_tavily`: Searches the live web via Tavily for recent updates, modern tools, or missing context.

Strategy:
- Always check `search_textbook_chunks` first for foundational concepts, math, or algorithms.
- Call `search_web_tavily` whenever the question involves recent advancements, real-world libraries/frameworks, or when the textbook lacks sufficient detail.
- You may call both tools if a question bridges classical theory and modern practice.
- Once you have sufficient evidence, synthesize a thorough, well-structured answer citing whether facts come from [Textbook Passage] or [Web Source]. Rely strictly on tool outputs."""


def agent_reasoner_node(state: AgenticRAGState) -> dict:
    history = state.get("messages", [])
    if not history:
        history = [HumanMessage(content=state["question"])]

    system_msg = SystemMessage(content=AGENT_SYSTEM_PROMPT)

    if state.get("ans_grade") == "ungrounded" and state.get("answer_reasoning"):
        system_msg = SystemMessage(
            content=(
                AGENT_SYSTEM_PROMPT
                + f"\n\nCRITICAL CORRECTION: Your previous draft failed groundedness verification ({state['answer_reasoning']}). "
                "Rewrite your answer strictly using ONLY facts explicitly present in the tool outputs above."
            )
        )

    response = agent_llm.invoke([system_msg] + history)
    new_calls = len(response.tool_calls) if getattr(response, "tool_calls", None) else 0

    return {
        "messages": [response],
        "tool_calls_count": state.get("tool_calls_count", 0) + new_calls,
        "final_answer": response.content if not new_calls else state.get("final_answer", ""),
        "gen_retry_count": state.get("gen_retry_count", 0) + (1 if not new_calls else 0),
    }


tool_executor_node = ToolNode(TOOLS)


def grade_hallucination_node(state: AgenticRAGState) -> dict:
    evidence = "\n\n---\n\n".join(
        msg.content for msg in state["messages"] if isinstance(msg, ToolMessage)
    )
    answer = state["final_answer"]

    if not evidence:
        return {
            "ans_grade": "ungrounded",
            "answer_reasoning": "No retrieval tools were called to ground the answer.",
        }

    res: GradeHallucination = hallucination_grader_llm.invoke([
        SystemMessage(
            content=(
                "You are a strict hallucination auditor. Verify whether every factual claim in the "
                "Agent Answer is directly supported by the Retrieved Tool Evidence. Classify as 'grounded' or 'ungrounded'."
            )
        ),
        HumanMessage(
            content=f"Retrieved Tool Evidence:\n{evidence}\n\nAgent Answer:\n{answer}"
        ),
    ])
    return {
        "ans_grade": res.grade,
        "answer_reasoning": res.reasoning,
    }

def route_agent_step(state: AgenticRAGState) -> str:
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None) and state.get("tool_calls_count", 0) <= 4:
        return "tools"
    return "grade_answer"


def route_after_audit(state: AgenticRAGState) -> str:
    if state["ans_grade"] == "ungrounded" and state.get("gen_retry_count", 0) < 2:
        return "agent"
    return "end"


builder = StateGraph(AgenticRAGState)
builder.add_node("agent", agent_reasoner_node)
builder.add_node("tools", tool_executor_node)
builder.add_node("grade_answer", grade_hallucination_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_agent_step,
    {"tools": "tools", "grade_answer": "grade_answer"},
)
builder.add_edge("tools", "agent")
builder.add_conditional_edges(
    "grade_answer",
    route_after_audit,
    {"agent": "agent", "end": END},
)

agentic_rag_app = builder.compile()

if __name__ == "__main__":
    question = (
        "Explain the theoretical difference between L1 and L2 regularization from the textbook, "
        "and how PyTorch's AdamW optimizer handles weight decay differently from standard L2 regularization."
    )

    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
    }

    final_state = {}
    for event in agentic_rag_app.stream(initial_state, stream_mode="updates"):
        for node_name, output in event.items():
            final_state.update(output)
            print(f"\n=== Node: {node_name} ===")
            if node_name == "agent":
                last_msg = output["messages"][-1]
                if getattr(last_msg, "tool_calls", None):
                    for tc in last_msg.tool_calls:
                        print(f"  -> Calling Tool: {tc['name']}({tc['args']})")
                else:
                    print("  -> Synthesized Draft Answer")
            elif node_name == "tools":
                for msg in output["messages"]:
                    print(f"  <- Tool '{msg.name}' returned {len(msg.content)} chars")
            elif node_name == "grade_answer":
                print(f"  -> Groundedness: {output['ans_grade']} | {output['answer_reasoning']}")

    print("\n================ FINAL VERIFIED ANSWER ================\n")
    print(final_state.get("final_answer", ""))