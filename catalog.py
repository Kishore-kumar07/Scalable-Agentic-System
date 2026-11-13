from __future__ import annotations
 
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Literal
 
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import html
 
load_dotenv()
DATA = Path("data")
 
RiskTier = Literal["read", "write", "money_movement"]
 
 
# ----------------------------------------------------------------- models
class Param(BaseModel):
    name: str                      # LLM-facing identifier (no dots)
    key: str                       # original key (dotted for nested body fields)
    location: Literal["path", "query", "body", "header"]
    required: bool = False
    type: str = "string"
    description: str = ""
    default: Any = None
 
    def to_schema(self) -> dict:
        s = {"type": self.type, "description": (self.description or self.key)[:200]}
        if self.type == "array":
            s["items"] = {"type": "object"}
        if self.default is not None:
            s["default"] = self.default
        return s
 
 
class ToolSpec(BaseModel):
    name: str
    service: str = "paypal"
    category: str = "general"
    description: str
    method: str = "GET"
    url: str = ""
    params: list[Param] = Field(default_factory=list)
    risk: RiskTier = "read"
    body_template: dict | None = None
 
    def embed_text(self) -> str:
        pn = ", ".join(p.key for p in self.params)
        return (f"{self.service} {self.category.replace('_', ' ')} | "
                f"{self.name.replace('_', ' ')}. {self.description} "
                f"[{self.method}] params: {pn}")
 
    def to_openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name,
            "description": f"[{self.service}/{self.risk}] {self.description}"[:900],
            "parameters": {
                "type": "object",
                "properties": {p.name: p.to_schema() for p in self.params},
                "required": [p.name for p in self.params if p.required],
            }}}
 
 
# --------------------------------------------------------- risk classifier
READ_WORDS = {"search", "list", "get", "show", "find", "lookup", "retrieve",
              "fetch", "view", "query", "check", "download"}
MONEY_WORDS = {"payout", "payouts", "refund", "capture", "transfer", "payment",
               "payments", "authorize", "charge", "pay", "order", "withdraw"}
 
 
def classify_risk(method: str, title: str) -> str:
    """Rule-based. REVIEW the output of `python run.py catalog` and fix by hand."""
    tokens = set(re.findall(r"[a-z]+", title.lower()))
    if method.upper() == "GET" or tokens & READ_WORDS:
        return "read"
    if tokens & MONEY_WORDS:
        return "money_movement"
    return "write"
 
 
# --------------------------------------------------------- Postman parser
def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
 
 
def _text(d) -> str:
    if isinstance(d, dict):
        d = d.get("content", "")
    return re.sub(r"\s+", " ", str(d or "")).strip()
 
 
def _type_of(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "string"
 
 
def _flatten(obj, prefix="", depth=0, max_depth=3):
    out = []
    for k, v in obj.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v and depth < max_depth:
            out.extend(_flatten(v, full, depth + 1, max_depth))
        else:
            out.append((full, v))
    return out
 
 
def _extract_params(req: dict) -> list[Param]:
    params: list[Param] = []
    seen: set[str] = set()
 
    def add(key, loc, required=False, typ="string", desc=""):
        name = re.sub(r"[^a-zA-Z0-9_]+", "__", key).strip("_") or "param"
        while name in seen:
            name += "_"
        seen.add(name)
        params.append(Param(name=name, key=key, location=loc, required=required,
                            type=typ, description=desc))
 
    url = req.get("url", {})
    if isinstance(url, str):
        url = {"raw": url}
    for seg in url.get("path", []) or []:
        if isinstance(seg, str) and seg.startswith(":"):
            add(seg[1:], "path", required=True)
    for q in url.get("query", []) or []:
        if q.get("key"):
            add(q["key"], "query", desc=_text(q.get("description")))
 
    body = req.get("body") or {}
    if body.get("mode") == "raw" and body.get("raw"):
        raw = re.sub(r'(?<!")\{\{[^}]*\}\}(?!")', '"__VAR__"', body["raw"])
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key, val in _flatten(parsed):
                    # heuristic: placeholder / empty sample value => required
                    req_flag = isinstance(val, str) and (
                        val.strip() == "" or "__VAR__" in val or "{{" in val)
                    add(key, "body", required=req_flag, typ=_type_of(val))
        except json.JSONDecodeError:
            pass
    elif body.get("mode") == "urlencoded":
        for f in body.get("urlencoded", []):
            if f.get("key"):
                add(f["key"], "body", desc=_text(f.get("description")))
    return params
 
 
def parse_postman(path: str | Path, service: str = "paypal") -> list[ToolSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[ToolSpec] = []
    used: set[str] = set()
 
    def walk(items, trail):
        for it in items:
            if "item" in it:
                walk(it["item"], trail + [it.get("name", "")])
                continue
            req = it.get("request")
            if not req:
                continue
            if isinstance(req, str):
                req = {"method": "GET", "url": req}
            title = it.get("name", "unnamed")
            method = req.get("method", "GET").upper()
            url = req.get("url", "")
            raw_url = url.get("raw", "") if isinstance(url, dict) else url
            desc = _text(req.get("description")) or _text(it.get("description")) or title
            category = slugify(trail[-1]) if trail else "general"
            name = slugify(f"{category}_{title}")[:60] or "tool"
            base, i = name, 2
            while name in used:
                name = f"{base[:56]}_{i}"
                i += 1
            used.add(name)
            out.append(ToolSpec(
                name=name, service=service, category=category,
                description=desc[:600], method=method, url=raw_url,
                params=_extract_params(req), risk=classify_risk(method, title)))
 
    walk(data.get("item", []), [])
    return out
 
 
def save_tools(tools: list[ToolSpec], path=DATA / "tools.json"):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    Path(path).write_text(json.dumps([t.model_dump() for t in tools], indent=2),
                          encoding="utf-8")
 
 
def clean_text(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s or ""))
    return re.sub(r"\s+", " ", s).strip()
 
 
def apply_overrides(tools: list[ToolSpec], path=DATA / "overrides.json") -> list[ToolSpec]:
    p = Path(path)
    ov = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    names = {t.name for t in tools}
    drop = set(ov.get("delete", []))
    patches = ov.get("tools", {})
    for n in (drop | set(patches)) - names:
        print(f"[overrides] WARNING: unknown tool name '{n}' (check tools.json)")
    out = []
    for t in tools:
        if t.name in drop:
            continue
        t.description = clean_text(t.description)
        o = patches.get(t.name)
        if o:
            if "risk" in o:
                t.risk = o["risk"]
            if "description" in o:
                t.description = o["description"]
            if "body_template" in o:
                t.body_template = o["body_template"]
            if "params" in o:
                t.params = [Param(**{"key": x["name"], "location": "body", **x})
                            for x in o["params"]]
            for pname, patch in o.get("param_patch", {}).items():
                hit = [prm for prm in t.params if prm.name == pname]
                if not hit:
                    print(f"[overrides] WARNING: {t.name} has no param '{pname}'")
                for prm in hit:
                    for k, v in patch.items():
                        setattr(prm, k, v)
        out.append(t)
    return out
 
 
def load_tools(path=DATA / "tools.json") -> list[ToolSpec]:
    p = Path(path)
    if not p.exists():
        tools = parse_postman(DATA / "paypal_collection.json")
        save_tools(tools, p)
    else:
        tools = [ToolSpec(**r) for r in json.loads(p.read_text(encoding="utf-8"))]
    return apply_overrides(tools)
 
 
# ------------------------------------------------------------- embeddings
class Embedder:
    """Azure OpenAI embeddings if a deployment is set, else local MiniLM."""
 
    def __init__(self):
        self.cache: dict[str, np.ndarray] = {}
        dep = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
        if dep:
            from langchain_openai import AzureOpenAIEmbeddings
            self.mode = "azure"
            self._emb = AzureOpenAIEmbeddings(
                azure_deployment=dep,
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))
        else:
            from sentence_transformers import SentenceTransformer
            self.mode = "local"
            self._st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 
    def encode(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            if self.mode == "azure":
                vecs = []
                for i in range(0, len(todo), 16):      # small batches: safe on Azure
                    vecs.extend(self._emb.embed_documents(todo[i:i + 16]))
                arr = np.array(vecs, dtype="float32")
            else:
                arr = self._st.encode(todo, show_progress_bar=False).astype("float32")
            arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(1e-9)
            for t, v in zip(todo, arr):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]).astype("float32")
 
 
# -------------------------------------------------------- hybrid retriever
STOP = {"the", "and", "for", "with", "from", "that", "this", "what", "was", "were",
        "any", "can", "you", "all", "send", "show", "get", "please", "need", "want",
        "has", "have", "are", "there", "which", "last", "my", "our"}
 
 
def _stem(t: str) -> str:
    return t[:-1] if t.endswith("s") and len(t) > 3 else t
 
 
def _toks(s: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", s.lower())
            if len(t) > 2 and t not in STOP}
 
 
class ToolRetriever:
    """FAISS semantic search + lexical overlap, fused. Supports incremental add."""
 
    def __init__(self, tools: list[ToolSpec], embedder: Embedder | None = None,
                 alpha: float = 0.8):
        self.embedder = embedder or Embedder()
        self.alpha = alpha
        self.tools: list[ToolSpec] = []
        self.by_name: dict[str, ToolSpec] = {}
        self._tok: list[set[str]] = []
        self.index = None
        self.add_tools(tools)
 
    def add_tools(self, new: list[ToolSpec]):
        import faiss
        new = [t for t in new if t.name not in self.by_name]
        if not new:
            return
        vecs = self.embedder.encode([t.embed_text() for t in new])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        for t in new:
            self.tools.append(t)
            self.by_name[t.name] = t
            self._tok.append(_toks(f"{t.name} {t.category} {t.description}"))
 
    def search(self, query: str, k: int = 6) -> list[tuple[ToolSpec, float]]:
        q = self.embedder.encode([query])
        n = min(len(self.tools), max(k * 5, 30))
        scores, ids = self.index.search(q, n)
        qt = _toks(query)
        ranked = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            lex = len(qt & self._tok[i]) / max(len(qt), 1)
            ranked.append((self.alpha * float(s) + (1 - self.alpha) * lex, self.tools[i]))
        ranked.sort(key=lambda x: -x[0])
        return [(t, sc) for sc, t in ranked[:k]]
 
 
# --------------------------------------------------------------- RAG KB
SAMPLE_DOCS = {  # SYNTHETIC sample docs, not official PayPal documentation.
    "refunds.md": "A refund returns money from a completed payment to the buyer. "
                  "Full and partial refunds are supported. A refund needs the id of "
                  "the original capture or transaction. Refunds are money-movement "
                  "actions and always require user approval in this assistant.",
    "disputes.md": "A dispute is raised by a buyer about a transaction. Sellers should "
                   "respond before the deadline shown on the dispute. Typical statuses "
                   "include open, waiting for seller response, under review and resolved. "
                   "Evidence such as tracking numbers can be provided to contest a dispute.",
    "invoicing.md": "Invoices are created as drafts, then sent to the recipient. A draft "
                    "needs a recipient email, currency and at least one line item. Sent "
                    "invoices can be reminded, cancelled or marked as paid. Sending is a "
                    "write action and needs approval.",
    "reporting.md": "Transaction search returns transactions between a start and end date. "
                    "Date ranges are limited in length, so long periods must be split. "
                    "Total sales volume is the sum of completed transaction amounts.",
    "assistant_policy.md": "Safety policy of this assistant: read actions run directly. "
                           "Write and money-movement actions pause for explicit user approval. "
                           "Amounts above the approval limit need the user to type APPROVE. "
                           "Retries never duplicate a payment because every action carries "
                           "an idempotency key. Emails and account ids are masked in logs.",
}
 
 
class KnowledgeBase:
    def __init__(self, embedder: Embedder, docs_dir=DATA / "docs"):
        import faiss
        docs = dict(SAMPLE_DOCS)
        d = Path(docs_dir)
        if d.exists():
            for f in list(d.glob("*.md")) + list(d.glob("*.txt")):
                docs[f.name] = f.read_text(encoding="utf-8")
        self.embedder = embedder
        self.chunks: list[tuple[str, str]] = []
        for src, text in docs.items():
            buf = ""
            for para in re.split(r"\n\s*\n", text):
                if len(buf) + len(para) < 600:
                    buf = f"{buf}\n{para}".strip()
                else:
                    if buf:
                        self.chunks.append((src, buf))
                    buf = para.strip()
            if buf:
                self.chunks.append((src, buf))
        vecs = embedder.encode([c[1] for c in self.chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
 
    def search(self, q: str, k: int = 3) -> list[tuple[str, str, float]]:
        s, ids = self.index.search(self.embedder.encode([q]), min(k, len(self.chunks)))
        return [(self.chunks[i][0], self.chunks[i][1], float(sc))
                for sc, i in zip(s[0], ids[0]) if i >= 0]
 
 
# ------------------------------------------------- synthetic distractors
_SERVICES = {   # stripe/quickbooks/square deliberately overlap with PayPal (hard mode)
    "stripe": ["invoice", "payment_intent", "refund", "dispute", "payout", "customer",
               "subscription", "balance_transaction", "charge", "payment_link"],
    "quickbooks": ["invoice", "bill", "customer", "vendor", "expense", "sales_receipt",
                   "payment", "estimate", "report"],
    "square": ["invoice", "payment", "refund", "dispute", "customer", "payout",
               "order", "catalog_item"],
    "market_data": ["quote", "order_book", "ohlc_bar", "dividend", "earnings_event",
                    "index_constituent", "fx_rate", "bond_yield"],
    "portfolio": ["holding", "position", "pnl_report", "allocation", "benchmark",
                  "risk_exposure", "rebalance_order"],
    "kyc": ["identity_check", "sanctions_screening", "document", "risk_score",
            "beneficial_owner"],
    "settlement": ["trade", "confirmation", "allocation", "fail", "custody_instruction",
                   "margin_call"],
    "crm": ["contact", "account", "opportunity", "activity", "note"],
}
_VERBS = [("list", "GET", "List all {r}s in {s}"), ("get", "GET", "Retrieve one {r} from {s} by id"),
          ("create", "POST", "Create a new {r} in {s}"), ("update", "PATCH", "Update an existing {r} in {s}"),
          ("delete", "DELETE", "Delete a {r} in {s}"), ("search", "POST", "Search {r}s in {s} with filters"),
          ("export", "GET", "Export {r}s from {s} as a file"), ("cancel", "POST", "Cancel a {r} in {s}")]
 
 
def make_distractors(n: int = 450, seed: int = 7) -> list[ToolSpec]:
    combos = [(s, r, v) for s, rs in _SERVICES.items() for r in rs for v in _VERBS]
    random.Random(seed).shuffle(combos)
    out = []
    for s, r, (verb, method, tmpl) in combos[:n]:
        rn = r.replace("_", " ")
        ps = []
        if verb in ("get", "update", "delete", "cancel"):
            ps.append(Param(name=f"{r}_id", key=f"{r}_id", location="path", required=True))
        if verb in ("list", "search", "export"):
            ps += [Param(name="limit", key="limit", location="query", type="number"),
                   Param(name="status", key="status", location="query")]
        if verb in ("create", "update"):
            ps += [Param(name="currency", key="currency", location="body"),
                   Param(name="description", key="description", location="body")]
        out.append(ToolSpec(
            name=f"{s}_{verb}_{r}"[:64], service=s, category=r,
            description=tmpl.format(r=rn, s=s), method=method,
            url=f"https://api.{s}.example.com/v1/{r}s", params=ps,
            risk=classify_risk(method, f"{verb} {r}")))
    return out
