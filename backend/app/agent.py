"""LangGraph agent that answers customer questions using RAG over the wiki.

Graph:  START -> retrieve -> generate -> END
  retrieve: pull the most relevant wiki knowledge from the vector store
  generate: ask Gemini 2.5 Flash to answer, grounded in that knowledge and guided
            by the configured Agent Soul.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from . import vectorstore
from .config import settings

SYSTEM_PROMPT = """
You will be given RELEVANT KNOWLEDGE retrieved from an internal wiki. Treat this
knowledge as authoritative and prefer it over your own prior assumptions. If it
corrects a common mistake, make sure your answer reflects the correction. If no
knowledge is relevant, answer to the best of your ability and be clear when you
are uncertain.

You may also be given the CONVERSATION SO FAR (a summary of earlier messages plus
the most recent ones). Use it to resolve references like "it" or "that" and to stay
consistent with what was already said. Do not repeat earlier answers unless asked.

Answer concisely and helpfully."""


class AgentState(TypedDict):
    question: str
    retrieved: list[dict]
    answer: str
    soul_prompt: str
    history: str


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


def _retrieve(state: AgentState) -> AgentState:
    state["retrieved"] = vectorstore.search(state["question"])
    return state


def _format_context(hits: list[dict]) -> str:
    if not hits:
        return "(no relevant knowledge found for this question)"
    lines = []
    for i, h in enumerate(hits, 1):
        meta = h["metadata"]
        title = meta.get("title", "")
        lines.append(f"[{i}] {title}\n{h['document']}")
    return "\n\n".join(lines)


def _generate(state: AgentState) -> AgentState:
    context = _format_context(state["retrieved"])
    history = (state.get("history") or "").strip()
    history_block = f"CONVERSATION SO FAR:\n{history}\n\n" if history else ""
    user_content = (
        f"{history_block}"
        f"RELEVANT KNOWLEDGE FROM THE WIKI:\n{context}\n\n"
        f"CUSTOMER QUESTION:\n{state['question']}"
    )
    soul = (state.get("soul_prompt") or "").strip()
    system_content = f"{soul}\n\n{SYSTEM_PROMPT}" if soul else SYSTEM_PROMPT
    messages = [SystemMessage(content=system_content), HumanMessage(content=user_content)]
    response = _get_llm().invoke(messages)
    state["answer"] = response.content
    return state


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("generate", _generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_app = None


def get_agent():
    global _app
    if _app is None:
        _app = _build_graph()
    return _app


def answer_question(
    question: str, soul_prompt: str = "", history: str = ""
) -> tuple[str, list[dict]]:
    """Run the agent. Returns (answer, retrieved_knowledge).

    ``history`` is the conversation-history block built by :mod:`.history` (recent
    turns verbatim + a rolling summary of older ones); empty for a fresh session.
    """
    result = get_agent().invoke(
        {
            "question": question,
            "retrieved": [],
            "answer": "",
            "soul_prompt": soul_prompt,
            "history": history,
        }
    )
    return result["answer"], result["retrieved"]
