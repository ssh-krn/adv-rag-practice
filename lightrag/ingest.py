import os
import json
import hashlib
import networkx as nx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

PDF_PATH = "./data/ml-foundation.pdf"       # <-- Keep your exact PDF path
GRAPH_PATH = "./lightrag_graph.gml"    # <-- Loads your existing graph checkpoint
CHROMA_DIR = "./chroma_db"             # <-- Reuses your existing Chroma collections

# Optional: If your first Groq run used random UUIDs before hitting TPD,
# set this to the last completed chunk index printed in your terminal (e.g. 245)
# so you don't re-process chunks that didn't use deterministic hash IDs.
START_CHUNK_INDEX = 0


# 1. Structured Extraction Schemas
class ExtractedEntity(BaseModel):
    name: str = Field(description="Canonical name of the concept, algorithm, or entity.")
    entity_type: str = Field(description="Category such as Algorithm, Metric, Technique, or Concept.")
    description: str = Field(description="Concise 1-2 sentence definition based on the text.")


class ExtractedRelation(BaseModel):
    source: str = Field(description="Exact name of the source entity.")
    target: str = Field(description="Exact name of the target entity.")
    keywords: str = Field(description="2-4 comma-separated theme keywords describing the relationship.")
    description: str = Field(description="1 sentence explaining how source and target relate.")


class GraphExtraction(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


# 2. Initialize Local Embeddings & Existing Chroma Collections
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

chunk_store = Chroma(
    collection_name="lightrag_chunks",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
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

# 3. Local Ollama Extractor (Native JSON Schema enforcement, zero rate limits)
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_ctx=4096,
    num_predict=600,        # Hard-stops any infinite whitespace/JSON loop at 600 tokens
    repeat_penalty=1.15,    # Prevents the model from repeating identical JSON keys/newlines
    client_kwargs={"timeout": 90.0},  # Raises an exception if a chunk ever takes >90s
)

extractor_llm = llm.with_structured_output(GraphExtraction, method="json_schema")

# Load existing NetworkX graph checkpoint if present
if os.path.exists(GRAPH_PATH):
    graph_db = nx.read_gml(GRAPH_PATH)
    print(f"Resumed existing graph: {graph_db.number_of_nodes()} nodes, {graph_db.number_of_edges()} edges.")
else:
    graph_db = nx.DiGraph()


# 4. Canonical Entity Deduplication
def resolve_canonical_entity(name: str, description: str, threshold: float = 0.90) -> str:
    normalized = name.strip().lower()
    for node_id in graph_db.nodes:
        if node_id.lower() == normalized:
            return node_id

    existing = entity_store.similarity_search_with_relevance_scores(
        f"{name}: {description}", k=1
    )
    if existing and existing[0][1] >= threshold:
        return existing[0][0].metadata["entityName"]

    return name.strip()


def make_chunk_id(text: str) -> str:
    """Deterministic SHA-256 ID so re-runs automatically recognize already-ingested chunks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


# 5. Resume PDF Ingestion into NetworkX + ChromaDB
def ingest_pdf(pdf_path: str) -> None:
    pages = PyMuPDFLoader(pdf_path).load()
    # Keep exact same chunk_size & overlap as your initial run
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    # Fetch already-stored deterministic chunk IDs from Chroma for automatic resume
    existing_ids = set(chunk_store.get(include=[])["ids"])
    print(f"Total chunks: {len(chunks)} | Starting at index {START_CHUNK_INDEX}...")

    for idx, doc in enumerate(chunks):
        if idx < START_CHUNK_INDEX:
            continue

        text = doc.page_content.strip()
        if len(text) < 100:
            continue

        chunk_id = make_chunk_id(text)
        if chunk_id in existing_ids:
            continue

        page_num = doc.metadata.get("page", 0)

        try:
            extraction: GraphExtraction = extractor_llm.invoke([
                SystemMessage(
                    content=(
                        "Extract key technical entities and directed relationships from the text. "
                        "If the text has no technical concepts (e.g. cover page or TOC), return empty lists. "
                        "Ensure every relation's source and target match an extracted entity name."
                    )
                ),
                HumanMessage(content=text),
            ])
        except Exception as e:
            print(f"[{idx + 1}/{len(chunks)}] Extraction error, skipping chunk: {e}")
            continue

        # Save raw chunk with deterministic ID
        chunk_store.add_documents(
            [Document(page_content=text, metadata={"chunkId": chunk_id, "page": page_num})],
            ids=[chunk_id],
        )
        existing_ids.add(chunk_id)

        name_map: dict[str, str] = {}

        # Merge deduplicated entities into NetworkX + Chroma
        for ent in extraction.entities:
            canonical = resolve_canonical_entity(ent.name, ent.description)
            name_map[ent.name] = canonical

            if graph_db.has_node(canonical):
                old_desc = graph_db.nodes[canonical].get("description", "")
                if ent.description not in old_desc:
                    graph_db.nodes[canonical]["description"] = f"{old_desc} | {ent.description}".strip(" |")
                c_ids = json.loads(graph_db.nodes[canonical].get("chunkIds", "[]"))
                if chunk_id not in c_ids:
                    c_ids.append(chunk_id)
                graph_db.nodes[canonical]["chunkIds"] = json.dumps(c_ids)
            else:
                graph_db.add_node(
                    canonical,
                    entityType=ent.entity_type,
                    description=ent.description,
                    chunkIds=json.dumps([chunk_id]),
                )

            entity_store.add_documents(
                [
                    Document(
                        page_content=f"{canonical} ({ent.entity_type}): {graph_db.nodes[canonical]['description']}",
                        metadata={"entityName": canonical},
                    )
                ],
                ids=[canonical.lower().replace(" ", "_")],
            )

        # Merge directed relations into NetworkX + Chroma
        for rel in extraction.relations:
            src = name_map.get(rel.source, resolve_canonical_entity(rel.source, rel.description))
            tgt = name_map.get(rel.target, resolve_canonical_entity(rel.target, rel.description))

            for node in (src, tgt):
                if not graph_db.has_node(node):
                    graph_db.add_node(node, entityType="Concept", description="", chunkIds=json.dumps([chunk_id]))

            if graph_db.has_edge(src, tgt):
                old_rel = graph_db.edges[src, tgt].get("description", "")
                if rel.description not in old_rel:
                    graph_db.edges[src, tgt]["description"] = f"{old_rel} | {rel.description}".strip(" |")
            else:
                graph_db.add_edge(src, tgt, keywords=rel.keywords, description=rel.description)

            rel_id = f"{src}__{tgt}".lower().replace(" ", "_")
            relation_store.add_documents(
                [
                    Document(
                        page_content=f"{src} -> {tgt} [{rel.keywords}]: {graph_db.edges[src, tgt]['description']}",
                        metadata={"source": src, "target": tgt},
                    )
                ],
                ids=[rel_id],
            )

        if (idx + 1) % 5 == 0:
            nx.write_gml(graph_db, GRAPH_PATH)
            print(f"[{idx + 1}/{len(chunks)}] Nodes: {graph_db.number_of_nodes()} | Edges: {graph_db.number_of_edges()}")

    nx.write_gml(graph_db, GRAPH_PATH)
    print(f"Complete! Saved {graph_db.number_of_nodes()} nodes and {graph_db.number_of_edges()} edges to {GRAPH_PATH}.")


if __name__ == "__main__":
    ingest_pdf(PDF_PATH)