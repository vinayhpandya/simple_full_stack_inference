"""
LangGraph ReAct agent that sits in front of the inference gateway.

Architecture:
  Client → agent_launcher (port 8090)
              ↓
         LangGraph ReAct loop
              ↓  ↑
           Tools (web search, calculator)
              ↓
         gateway_launcher (port 8080)
              ↓
         Anyscale DeepSeek MoE / Modal SGLang / local

The agent intercepts every chat request, decides whether tools are needed,
calls them, and then forwards the enriched context to the LLM via the
gateway. The gateway URL is the only required config — no API keys needed
since the gateway doesn't require one.

Run:
  uv run python agent_launcher.py

Environment variables:
  GATEWAY_URL    — base URL of the gateway (default: http://localhost:8080)
  GATEWAY_MODEL  — model name to send in requests (default: deepseek)
  AGENT_PORT     — port this agent server listens on (default: 8090)
"""

import ast
import json
import operator
import os
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")
GATEWAY_MODEL = os.environ.get("GATEWAY_MODEL", "deepseek")
AGENT_PORT = int(os.environ.get("AGENT_PORT", "8090"))

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information. Use this when the user asks
    about recent events, current facts, or anything the model may not know."""
    try:
        return _search.run(query)
    except Exception as exc:
        return f"Search failed: {exc}"


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression and return the result.
    Supports standard arithmetic, exponentiation, and parentheses.
    Example: '(3 + 4) * 2 ** 3'"""
    try:
        # Restrict to safe AST nodes — no builtins, no attribute access.
        _SAFE_OPS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BinOp):
                op = _SAFE_OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {node.op}")
                return op(_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp):
                op = _SAFE_OPS.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {node.op}")
                return op(_eval(node.operand))
            raise ValueError(f"Unsupported expression node: {type(node)}")

        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"


TOOLS = [web_search, calculator]

# ---------------------------------------------------------------------------
# LLM — points at the gateway, which exposes an OpenAI-compatible API
# ---------------------------------------------------------------------------

llm = ChatOpenAI(
    base_url=f"{GATEWAY_URL}/v1",
    api_key="not-needed",
    model=GATEWAY_MODEL,
    temperature=0.7,
)

llm_with_tools = llm.bind_tools(TOOLS)

# ---------------------------------------------------------------------------
# LangGraph ReAct graph
#
# agent ──(tool calls?)──► tools ──► agent ──► END
#         (no tool calls)──────────────────────►
# ---------------------------------------------------------------------------


def call_agent(state: MessagesState) -> dict:
    """Invoke the LLM with tool bindings. May produce tool call requests."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)

_builder = StateGraph(MessagesState)
_builder.add_node("agent", call_agent)
_builder.add_node("tools", tool_node)
_builder.set_entry_point("agent")
_builder.add_conditional_edges("agent", tools_condition)
_builder.add_edge("tools", "agent")

graph = _builder.compile()

# ---------------------------------------------------------------------------
# Request / response schema (OpenAI-compatible subset)
# ---------------------------------------------------------------------------


class _Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[_Message]
    model: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LangGraph Agent",
    description="ReAct agent with web search and calculator, backed by the inference gateway.",
)


def _to_lc_messages(messages: List[_Message]):
    """Convert OpenAI-style dicts to LangChain message objects."""
    result = []
    for m in messages:
        if m.role == "system":
            result.append(SystemMessage(content=m.content))
        elif m.role == "user":
            result.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            result.append(AIMessage(content=m.content))
    return result


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    lc_messages = _to_lc_messages(req.messages)
    if not lc_messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    try:
        result = await graph.ainvoke({"messages": lc_messages})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}") from exc

    # The last message is always the final AI response (after all tool rounds).
    final = result["messages"][-1]
    content = final.content if hasattr(final, "content") else str(final)

    # Count tool calls made during this run for observability.
    tool_rounds = sum(
        1 for m in result["messages"] if isinstance(m, ToolMessage)
    )
    if tool_rounds:
        print(f"[agent] {tool_rounds} tool round(s) before final answer")

    return JSONResponse({
        "id": f"agent-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "model": req.model or GATEWAY_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [{"id": GATEWAY_MODEL, "object": "model", "owned_by": "agent"}],
    })


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "gateway": GATEWAY_URL, "model": GATEWAY_MODEL})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[agent] Starting LangGraph agent on port {AGENT_PORT}")
    print(f"[agent] Gateway: {GATEWAY_URL}  Model: {GATEWAY_MODEL}")
    print(f"[agent] Tools: {[t.name for t in TOOLS]}")
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
