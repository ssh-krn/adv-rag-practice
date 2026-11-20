from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from typing import TypedDict


load_dotenv()

class AppState(TypedDict):
    user_query: str
    retrieved_context: str
    llm_response: str

def retriver(state: AppState):
    user_query = state["user_query"]
