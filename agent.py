from __future__ import annotations
 
import ast
import hashlib
import json
import operator  # noqa: F401  (kept for easy extension of safe math)
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, TypedDict
 
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
 
from catalog import KnowledgeBase, ToolRetriever, ToolSpec
 
load_dotenv()
 
MAX_STEPS = 6                 # max LLM calls per user turn
MAX_TOOL_CHARS = 2500         # trim big API responses before they reach the LLM
APPROVAL_LIMIT = float(os.getenv("APPROVAL_LIMIT", "1000"))
 
 
def make_llm():
    from langchain_openai import AzureChatOpenAI
    kw = {}
    if os.getenv("LLM_TEMPERATURE"):          # some reasoning models reject temperature
        kw["temperature"] = float(os.environ["LLM_TEMPERATURE"])
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"), **kw)
 
 
# ------------------------------------------------------- PII masking + audit
_EMAIL = re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_LONGNUM = re.compile(r"\b\d{9,}\b")
 
 
def mask(s: str) -> str:
    return _LONGNUM.sub("****", _EMAIL.sub(r"\1***@\2", s))
 
 
def mask_obj(o):
    if isinstance(o, str):
        return mask(o)
    if isinstance(o, dict):
        return {k: mask_obj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [mask_obj(v) for v in o]
    return o
 
 
class AuditLog:
    """Append-only JSONL. Feeds the System Search tool ('status of my last request?')."""
 
    def __init__(self, path="data/audit_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
 
    def write(self, **entry):
        row = {"ts": datetime.now().isoformat(timespec="seconds"), **mask_obj(entry)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
 
    def recent(self, thread_id: str, n: int = 5, kind: str = "api") -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows = [r for r in rows if r.get("thread_id") == thread_id and r.get("kind") == kind]
        return rows[-n:][::-1]
 
 
# ------------------------------------------------------------- mock API
class TransientAPIError(Exception):
    pass
 
 
class MockPayPal:
    """Stand-in for real HTTP calls. Validates nothing itself; returns realistic fakes.
    Honours idempotency keys; can inject failures via MOCK_FAILURE_RATE."""
 
    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate
        self.seen: dict[str, dict] = {}
 
    def call(self, spec: ToolSpec, req: dict, key: str) -> dict:
        if key in self.seen:
            return {**self.seen[key], "idempotent_replay": True}
        result = self._respond(spec, req)
        if spec.risk != "read":
            self.seen[key] = result            # side effect "happened"
        if self.failure_rate and random.random() < self.failure_rate:
            raise TransientAPIError("429 Too Many Requests / response lost")
        return result
 
    def _respond(self, spec: ToolSpec, req: dict) -> dict:
        n = spec.name
        seed = json.dumps(req, sort_keys=True, default=str)
        rng = random.Random(hashlib.md5(seed.encode()).hexdigest())
        if "transaction" in n and spec.risk == "read":
            start = req["query"].get("start_date", "2026-01-01T00:00:00Z")
            txs = [{"transaction_info": {
                "transaction_id": f"TX{rng.randint(10**8, 10**9)}",
                "transaction_amount": {"currency_code": "USD", "value": f"{rng.uniform(5, 400):.2f}"},
                "transaction_initiation_date": start}} for _ in range(8)]
            return {"transaction_details": txs, "total_items": 8, "page": 1,
                    "total_pages": 1, "_mock": True}
        if "dispute" in n:
            if spec.risk == "read":
                buyer = "user_123" if "user_123" in seed else f"user_{rng.randint(100, 999)}"
                return {"items": [{"dispute_id": f"PP-D-{rng.randint(1000, 9999)}",
                                   "status": "WAITING_FOR_SELLER_RESPONSE",
                                   "reason": "MERCHANDISE_OR_SERVICE_NOT_RECEIVED",
                                   "buyer_ref": buyer,
                                   "dispute_amount": {"currency_code": "USD", "value": "42.00"}}],
                        "_mock": True}
            return {"dispute_id": "PP-D-1234", "status": "UPDATED", "_mock": True}
        if "invoice" in n and spec.risk != "read":
            return {"id": f"INV2-{rng.randint(1000, 9999)}",
                    "status": "SENT" if "send" in n else "DRAFT", "_mock": True}
        if spec.risk == "read":
            return {"items": [{"id": f"MOCK-{rng.randint(100, 999)}", "status": "ACTIVE"}],
                    "total_items": 1, "_mock": True}
        return {"id": f"MOCK-{rng.randint(1000, 9999)}", "status": "SUCCESS", "_mock": True}
 
 
_PH = re.compile(r"^\$\{(\w+)(:str)?\}$")
_MISSING = object()
 
 
def fill_template(node, values: dict):
    """Fill '${name}' placeholders; drop anything unresolved; ':str' coerces to string."""
    if isinstance(node, str):
        m = _PH.match(node)
        if not m:
            return node
        v = values.get(m.group(1))
        if v is None or v == "":
            return _MISSING
        if m.group(2):
            v = str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
        return v
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            r = fill_template(v, values)
            if r is not _MISSING and r != {} and r != []:
                out[k] = r
        return out
    if isinstance(node, list):
        items = (fill_template(v, values) for v in node)
        return [r for r in items if r is not _MISSING and r != {} and r != []]
    return node
 
 
def build_request(spec: ToolSpec, args: dict, idem_key: str = "") -> dict:
    url = spec.url.split("?")[0]
    query, body = {}, {}
    for p in spec.params:
        v = args.get(p.name, p.default)
        if v is None:
            continue
        if p.location == "path":
            url = url.replace(f":{p.key}", str(v))
        elif p.location == "query":
            query[p.key] = v
        elif p.location == "body" and not spec.body_template:
            cur = body
            parts = p.key.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = v
    if spec.body_template:
        values = {p.name: args.get(p.name, p.default) for p in spec.params}
        values["idempotency_key"] = idem_key
        body = fill_template(spec.body_template, values)
 
    return {"method": spec.method, "url": url, "query": query, "body": body}
 
 
 
def validate_args(spec: ToolSpec, args: dict) -> str | None:
    known = {p.name: p for p in spec.params}
    errs = [f"unknown parameter '{a}'" for a in args if a not in known]
    for p in spec.params:
        v = args.get(p.name)
        if p.required and (v is None or v == ""):
            errs.append(f"missing required parameter '{p.name}' ({p.description or p.key})")
        elif v is not None:
            if p.type == "number" and (isinstance(v, bool) or not isinstance(v, (int, float))):
                errs.append(f"'{p.name}' must be a number")
            if p.type == "boolean" and not isinstance(v, bool):
                errs.append(f"'{p.name}' must be a boolean")
    return "; ".join(errs) or None
 
 
def execute_with_retry(api, spec, req, key, attempts=3, base=0.5):
    for i in range(1, attempts + 1):
        try:
            return api.call(spec, req, key)
        except TransientAPIError:
            if i == attempts:
                raise
            time.sleep(base * 2 ** (i - 1) + random.random() * 0.1)   # backoff + jitter
 
 
def find_amount(o) -> float:
    best = 0.0
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                best = max(best, find_amount(v))
            elif any(w in k.lower() for w in ("amount", "value", "total")):
                try:
                    best = max(best, float(v))
                except (TypeError, ValueError):
                    pass
    elif isinstance(o, list):
        best = max([find_amount(x) for x in o] + [0.0])
    return best
 
 
# --------------------------------------------------- deterministic utilities
def date_range(period: str, now: datetime | None = None) -> dict:
    t = (now or datetime.now()).replace(hour=0, minute=0, second=0, microsecond=0)
    d = timedelta(days=1)
    if period == "today":
        s = e = t
    elif period == "yesterday":
        s = e = t - d
    elif period == "last_7_days":
        s, e = t - 6 * d, t
    elif period == "last_30_days":
        s, e = t - 29 * d, t
    elif period == "this_month":
        s, e = t.replace(day=1), t
    elif period == "last_month":
        e = t.replace(day=1) - d
        s = e.replace(day=1)
    elif period == "this_year":
        s, e = t.replace(month=1, day=1), t
    elif period == "last_year":
        s, e = datetime(t.year - 1, 1, 1), datetime(t.year - 1, 12, 31)
    else:
        raise ValueError(f"unknown period '{period}'")
    return {"period": period, "start_date": s.strftime("%Y-%m-%dT00:00:00Z"),
            "end_date": e.strftime("%Y-%m-%dT23:59:59Z")}
 
 
def sum_numbers(numbers) -> dict:
    vals = [float(x) for x in numbers]
    return {"sum": round(sum(vals), 2), "count": len(vals)}
 
 
# ------------------------------------------------------------- meta-tools
def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}
 
 
META_SCHEMAS = [
    _fn("knowledge_search", "RAG over product docs/guides/policies. Use for 'how does X work' questions.",
        {"question": {"type": "string"}}, ["question"]),
    _fn("system_search", "Search the system itself: target='tools' lists available tools/capabilities; "
        "target='requests' shows the user's recent request log/status.",
        {"target": {"type": "string", "enum": ["tools", "requests"]},
         "query": {"type": "string", "description": "search text (for tools)"}}, ["target"]),
    _fn("find_tools", "Use when none of the visible tools fit: describe the capability needed "
        "and matching tools will be made available.",
        {"capability": {"type": "string"}}, ["capability"]),
    _fn("get_date_range", "Deterministic date range for relative periods. Never compute dates yourself.",
        {"period": {"type": "string", "enum": ["today", "yesterday", "last_7_days", "last_30_days",
                                               "this_month", "last_month", "this_year", "last_year"]}},
        ["period"]),
    _fn("sum_numbers", "Add up a list of numbers exactly.",
        {"numbers": {"type": "array", "items": {"type": "number"}}}, ["numbers"]),
]
META_NAMES = {s["function"]["name"] for s in META_SCHEMAS}
 
SYSTEM = """You are a payments assistant on a fintech platform.
- Each turn you only see a small subset of the available tools (chosen by retrieval).
  If none fits, call find_tools. Never invent tool names.
- Knowledge questions (how something works, policies) -> knowledge_search.
- Questions about the system itself (what tools exist, status of earlier requests) -> system_search.
- NEVER guess parameter values (emails, ids, amounts, dates). If a required value is
  missing, ask the user for it instead of calling the tool.
- Never ask for optional parameters (names, notes, quantity, currency). If all REQUIRED
  parameters are known, call the tool right away.
- Relative dates ("last month") -> get_date_range. Arithmetic over lists -> sum_numbers.
- Write/money actions are confirmed with the user by the system; call the tool once you
  have every parameter. If the user declines, do not retry.
- If a list/search tool needs no required parameters, call it and filter the results yourself
  instead of asking the user (e.g. "dispute from user_123" -> list disputes, then look for that buyer).
- Ask only for REQUIRED parameters. Never ask about optional ones that have defaults (quantity, currency).
- For "what tools exist" questions, call system_search and list everything it returns.
- Report only facts from tool results. Be concise.
Today is {today}."""
 
 
# ------------------------------------------------------------------ graph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    candidates: list[str]      # names of tools shown to the LLM this turn
    verdicts: dict             # tool_call_id -> "ok" | error text
    steps: int
 
 
def _trim(content) -> str:
    s = content if isinstance(content, str) else json.dumps(content, default=str)
    return s if len(s) <= MAX_TOOL_CHARS else s[:MAX_TOOL_CHARS] + "...[truncated]"
 
 
def _window(msgs: list, n: int = 20) -> list:
    tail = msgs[-n:]
    while tail and tail[0].type != "human":   # never start mid tool-call pair
        tail = tail[1:]
    return tail or msgs
 
 
def build_graph(retriever: ToolRetriever, kb: KnowledgeBase, llm, api: MockPayPal,
                audit: AuditLog, k: int = 6):
    by_name = retriever.by_name
 
 
    def retrieve(state: AgentState):
        humans = [m for m in state["messages"] if m.type == "human"]
        last = str(humans[-1].content)
        prev = state.get("candidates") or []
        names = [t.name for t, _ in retriever.search(last, k)]
        if len(last.split()) < 6 and prev:
            # short follow-up (e.g. an email address): keep the earlier tools first
            keep = prev[:4]
            merged = keep + [n for n in names if n not in keep]
        else:
            merged = names + [n for n in prev if n not in names]
        return {"candidates": merged[: k + 2], "steps": 0, "verdicts": {}}
   
    def agent_node(state: AgentState):
        schemas = [by_name[n].to_openai() for n in state["candidates"] if n in by_name]
        bound = llm.bind_tools(schemas + META_SCHEMAS)
        sys = SystemMessage(SYSTEM.format(today=datetime.now().strftime("%A %Y-%m-%d")))
        resp = bound.invoke([sys] + _window(state["messages"]))
        return {"messages": [resp], "steps": state.get("steps", 0) + 1}
 
    def route_agent(state: AgentState):
        return "gate" if getattr(state["messages"][-1], "tool_calls", None) else END
 
    def gate(state: AgentState):
        """Validation + human approval. Pure (no side effects): safe to re-run on resume."""
        verdicts, risky = {}, []
        for tc in state["messages"][-1].tool_calls:
            cid, name, args = tc["id"], tc["name"], tc["args"]
            if name in META_NAMES:
                verdicts[cid] = "ok"
                continue
            spec = by_name.get(name)
            if spec is None or name not in state["candidates"]:
                verdicts[cid] = (f"ERROR: tool '{name}' is not available. "
                                 "Call find_tools to discover the right tool.")
                continue
            err = validate_args(spec, args)
            if err:
                verdicts[cid] = (f"VALIDATION_ERROR: {err}. If a value is missing, "
                                 "ask the user; never guess.")
                continue
            verdicts[cid] = "ok"
            if spec.risk != "read":
                amt = find_amount(args)
                risky.append({"call_id": cid, "tool": name, "risk": spec.risk, "args": args,
                              "amount": amt, "elevated": amt > APPROVAL_LIMIT})
        if risky:
            decision = interrupt({"type": "approval", "actions": risky})
            ok = decision is True or str(decision).strip().lower() in ("y", "yes", "approve", "approved")
            if not ok:
                for r in risky:
                    verdicts[r["call_id"]] = "DECLINED_BY_USER: the user did not approve. Do not retry."
        return {"verdicts": verdicts}
 
    def run_meta(name, args, thread_id, cands):
        if name == "knowledge_search":
            hits = kb.search(args["question"], 3)
            ctx = "\n\n".join(f"[{s}] {t}" for s, t, _ in hits)
            r = llm.invoke([SystemMessage("Answer ONLY from the context. If it is insufficient, say so. "
                                          "Cite sources in [brackets]."),
                            HumanMessage(f"Context:\n{ctx}\n\nQuestion: {args['question']}")])
            return {"answer": r.content, "sources": sorted({s for s, _, _ in hits})}
        if name == "system_search":
            if args.get("target") == "requests":
                rows = audit.recent(thread_id, 5)
                return {"recent_requests": [{k2: r.get(k2) for k2 in
                        ("ts", "request_id", "tool", "risk", "status", "error")} for r in rows]
                        } if rows else {"recent_requests": [], "note": "No requests yet in this session."}
            hits = retriever.search(args.get("query") or "", 20)
            return {"tools": [{"name": t.name, "service": t.service, "risk": t.risk,
                               "description": t.description[:120]} for t, _ in hits],
                    "total_tools_in_catalog": len(retriever.tools)}
        if name == "find_tools":
            hits = retriever.search(args["capability"], k)
            for t, _ in hits[::-1]:
                if t.name in cands:
                    cands.remove(t.name)
                cands.insert(0, t.name)
            del cands[k + 4:]
            return {"now_available": [{"name": t.name, "description": t.description[:100]}
                                      for t, _ in hits]}
        if name == "get_date_range":
            return date_range(args["period"])
        if name == "sum_numbers":
            return sum_numbers(args["numbers"])
        raise KeyError(name)
 
    def execute(state: AgentState, config: RunnableConfig):
        thread_id = config["configurable"]["thread_id"]
        cands = list(state["candidates"])
        msgs = []
        for tc in state["messages"][-1].tool_calls:
            cid, name, args = tc["id"], tc["name"], tc["args"]
            verdict = state["verdicts"].get(cid, "ERROR: no verdict")
            rid = uuid.uuid4().hex[:8]
            spec = by_name.get(name)
            kind = "meta" if name in META_NAMES else "api"
            status, error = "ok", None
            try:
                if verdict != "ok":
                    status, content = "blocked", verdict
                elif kind == "meta":
                    content = run_meta(name, args, thread_id, cands)
                else:
                    key = hashlib.sha256(f"{thread_id}|{cid}|{name}|"
                                         f"{json.dumps(args, sort_keys=True)}".encode()).hexdigest()[:32]
                    content = execute_with_retry(api, spec, build_request(spec, args, key), key)
            except TransientAPIError as e:
                status, error = "failed", str(e)
                content = (f"API_ERROR: service unavailable after retries ({e}). "
                           "Tell the user; do not retry blindly.")
            except Exception as e:  # noqa: BLE001 - never crash the graph on a tool error
                status, error = "failed", f"{type(e).__name__}: {e}"
                content = f"ERROR: {error}"
            audit.write(kind=kind, thread_id=thread_id, request_id=rid, tool=name,
                        risk=spec.risk if spec else "meta", args=args, status=status, error=error)
            msgs.append(ToolMessage(content=_trim(content), tool_call_id=cid, name=name))
        return {"messages": msgs, "candidates": cands}
 
    def route_execute(state: AgentState):
        return "giveup" if state.get("steps", 0) >= MAX_STEPS else "agent"
 
    def giveup(state: AgentState):
        return {"messages": [AIMessage(content="I couldn't complete this request within my step "
                                               "limit. Please rephrase or give more detail.")]}
 
    g = StateGraph(AgentState)
    for n, f in [("retrieve", retrieve), ("agent", agent_node), ("gate", gate),
                 ("execute", execute), ("giveup", giveup)]:
        g.add_node(n, f)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_agent, {"gate": "gate", END: END})
    g.add_edge("gate", "execute")
    g.add_conditional_edges("execute", route_execute, {"agent": "agent", "giveup": "giveup"})
    g.add_edge("giveup", END)
    return g.compile(checkpointer=MemorySaver())