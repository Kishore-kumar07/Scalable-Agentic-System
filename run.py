"""run.py - CLI: catalog | chat | demo | eval
 
  python run.py catalog
  python run.py chat [--distractors 450]
  python run.py demo
  python run.py eval [--distractors 450] [--k 6] [--show] [--llm]
"""
from __future__ import annotations
 
import argparse
import json
import os
import uuid
from collections import Counter
from pathlib import Path
 
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
 
from agent import AuditLog, MockPayPal, build_graph, make_llm
from catalog import (DATA, Embedder, KnowledgeBase, ToolRetriever, load_tools,
                     make_distractors, parse_postman, save_tools)
 
 
# ---------------------------------------------------------------- catalog
def cmd_catalog(_):
    tools = parse_postman(DATA / "paypal_collection.json")
    save_tools(tools)
    print(f"Parsed {len(tools)} tools -> data/tools.json\n")
    print("Risk tiers:", dict(Counter(t.risk for t in tools)))
    print("Categories:", dict(Counter(t.category for t in tools)), "\n")
    for t in tools:
        req = [p.name for p in t.params if p.required]
        print(f"[{t.risk:14}] {t.name:60} {t.method:6} required={req}")
    print("\nREVIEW the risk tiers and required params above; fix tools.json by hand if needed.")
 
 
# ------------------------------------------------------------------ chat
def build_all(n_distractors=0, k=6):
    tools = load_tools()
    emb = Embedder()
    retr = ToolRetriever(tools, emb)
    if n_distractors:
        retr.add_tools(make_distractors(n_distractors))
    kb = KnowledgeBase(emb)
    api = MockPayPal(float(os.getenv("MOCK_FAILURE_RATE", "0")))
    graph = build_graph(retr, kb, make_llm(), api, AuditLog(), k=k)
    print(f"[catalog size: {len(retr.tools)} tools | embeddings: {emb.mode}]")
    return graph
 
 
def run_turn(graph, cfg, text, ask=input):
    inp = {"messages": [HumanMessage(text)]}
    while True:
        graph.invoke(inp, cfg)
        snap = graph.get_state(cfg)
        if not snap.next:
            break
        payload = snap.tasks[0].interrupts[0].value
        approved = True
        print("\n=== APPROVAL REQUIRED ===")
        for a in payload["actions"]:
            print(f"  {a['tool']}  risk={a['risk']}  amount={a['amount']}")
            print(f"  args: {json.dumps(a['args'])}")
        if any(a["elevated"] for a in payload["actions"]):
            approved = ask("  Above approval limit. Type APPROVE to confirm: ").strip() == "APPROVE"
        else:
            approved = ask("  Confirm? (y/n): ").strip().lower() in ("y", "yes")
        inp = Command(resume=approved)
    vals = graph.get_state(cfg).values
    print(f"  [tools shown to LLM: {vals.get('candidates')}]")
    print("agent>", vals["messages"][-1].content)
 
 
def new_cfg():
    t = f"cli-{uuid.uuid4().hex[:6]}"
    return {"configurable": {"thread_id": t}, "recursion_limit": 40,
            "tags": ["fintech-agent"], "metadata": {"thread_id": t}}
 
 
def cmd_chat(a):
    graph, cfg = build_all(a.distractors, a.k), new_cfg()
    print("Type 'exit' to quit.")
    while True:
        text = input("\nyou> ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if text:
            run_turn(graph, cfg, text)
 
 
def cmd_demo(a):
    graph, cfg = build_all(a.distractors, a.k), new_cfg()
    for q in ["Send an invoice for $50", "john@example.com, for consulting services",
              "What was my total sales volume last month?",
              "Is there a dispute open from user_123?",
              "What tools are available for managing invoices?",
              "How do refunds work?",
              "What's the status of my last request?"]:
        print(f"\nyou> {q}")
        run_turn(graph, cfg, q)
 
 
# ------------------------------------------------------------------- eval
DEFAULT_CASES = [  # 'expect' = substrings that must ALL appear in the correct PayPal tool name
    {"q": "Send an invoice for $50 to john@example.com", "expect": ["invoice", "send"]},
    {"q": "Create a draft invoice for 3 hours of consulting", "expect": ["invoice", "create"]},
    {"q": "Cancel the invoice INV2-1234", "expect": ["invoice", "cancel"]},
    {"q": "List all my invoices", "expect": ["invoice", "list"]},
    {"q": "What was my total sales volume last month?", "expect": ["transaction"]},
    {"q": "Show me transactions from last week", "expect": ["transaction"]},
    {"q": "Is there a dispute open from user_123?", "expect": ["dispute", "list"]},
    {"q": "Accept the claim for dispute PP-D-1234", "expect": ["dispute", "accept"]},
    {"q": "Refund the capture 5O190127TN364715T", "expect": ["refund"]},
    {"q": "Send a payout of $20 to alice@example.com", "expect": ["payout"]},
    {"q": "Show the details of order 5O190127", "expect": ["order"]},
    {"q": "What are my account balances?", "expect": ["balance"]},
]
# These substrings are GUESSES. Run `python run.py eval --show`, then edit eval/cases.json
# to ~25-30 cases that match the real tool names in your collection.
 
 
def hit_rank(hits, expect):
    for i, (t, _) in enumerate(hits, 1):
        if t.service == "paypal" and t.name in expect:
            return i
    return None
 
 
def retrieval_metrics(retr, cases, k, show=False):
    ranks = []
    for c in cases:
        hits = retr.search(c["q"], k)
        r = hit_rank(hits, c["expect"])
        ranks.append(r)
        if show:
            print(f"\nQ: {c['q']}  (expect {c['expect']}) -> rank {r}")
            for t, s in hits:
                print(f"   {s:.2f} [{t.service}/{t.risk}] {t.name}")
    n = len(cases)
    return {"recall@k": sum(r is not None for r in ranks) / n,
            "top1": sum(r == 1 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r) / n}
 
 
def selection_accuracy(retr, llm, cases, k, use_all):
    if use_all and len(retr.tools) > 128:
        return None                                  # OpenAI-style APIs cap tools per request (~128)
    correct = 0
    err_seen = []
    for c in cases:
        specs = retr.tools if use_all else [t for t, _ in retr.search(c["q"], k)]
        bound = llm.bind_tools([t.to_openai() for t in specs], tool_choice="required")
        try:
            r = bound.invoke([SystemMessage("Pick the single best tool for the user's request."),
                              HumanMessage(c["q"])])
            name = r.tool_calls[0]["name"] if r.tool_calls else None
        except Exception as e:
            name = None
            if not err_seen:
                print("API error:", str(e)[:300])
                err_seen.append(1)
        spec = retr.by_name.get(name)
        if spec and spec.service == "paypal" and spec.name in c["expect"]:
            correct += 1
    return correct / len(cases)
 
def schema_tokens(specs):
    """Rough estimate: 1 token is about 4 characters."""
    return sum(len(json.dumps(t.to_openai())) for t in specs) // 4
 
def cmd_eval(a):
    p = Path("eval/cases.json")
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_CASES, indent=2), encoding="utf-8")
    cases = json.loads(p.read_text(encoding="utf-8"))
 
    tools = load_tools()
    retr = ToolRetriever(tools, Embedder())
    llm = make_llm() if a.llm else None
    rows = []
 
    def measure(label):
        m = retrieval_metrics(retr, cases, a.k, show=a.show)
        row = {"catalog": label, "tools": len(retr.tools), **m}
        if llm:
            row["LLM acc (top-k bound)"] = selection_accuracy(retr, llm, cases, a.k, False)
            row["LLM acc (ALL bound)"] = selection_accuracy(retr, llm, cases, a.k, True)
 
        row["tokens ALL tools"] = schema_tokens(retr.tools)
        row["tokens top-k"] = schema_tokens([t for t, _ in retr.search(cases[0]["q"], a.k)])
        rows.append(row)
 
    measure("PayPal only")
    retr.add_tools(make_distractors(a.distractors))      # incremental index update
    measure(f"+{a.distractors} distractors")
 
    def fmt(v):
        return "n/a (>128 tools)" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
 
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    out = "\n".join(md)
    print("\n" + out)
    Path("eval/results.md").write_text(out + "\n", encoding="utf-8")
 
 
# ------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("catalog").set_defaults(fn=cmd_catalog)
    for name, fn in [("chat", cmd_chat), ("demo", cmd_demo), ("eval", cmd_eval)]:
        sp = sub.add_parser(name)
        sp.add_argument("--distractors", type=int, default=450 if name == "eval" else 0)
        sp.add_argument("--k", type=int, default=6)
        if name == "eval":
            sp.add_argument("--show", action="store_true")
            sp.add_argument("--llm", action="store_true")
        sp.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)