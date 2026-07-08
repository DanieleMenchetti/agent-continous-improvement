"""LangGraph agent that answers customer questions.

Graph:  START -> generate -> END
  generate: ask Gemini 2.5 Flash to answer, guided by the configured Agent Soul.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from .config import settings

SYSTEM_PROMPT = """
Answer the customer's question as helpfully and accurately as you can.
Be clear when you are uncertain.

Answer concisely and helpfully."""


class AgentState(TypedDict):
    question: str
    answer: str
    soul_prompt: str


_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )
    return _llm


def _generate(state: AgentState) -> AgentState:
    soul = (state.get("soul_prompt") or "").strip()
    system_content = f"{soul}\n\n{SYSTEM_PROMPT}" if soul else SYSTEM_PROMPT
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=state["question"]),
    ]
    response = _get_llm().invoke(messages)
    state["answer"] = response.content
    return state


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("generate", _generate)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_app = None


def get_agent():
    global _app
    if _app is None:
        _app = _build_graph()
    return _app


def answer_question(question: str, soul_prompt: str = "") -> str:
    """Run the agent and return its answer."""
    result = get_agent().invoke(
        {"question": question, "answer": "", "soul_prompt": soul_prompt}
    )
    return result["answer"]
