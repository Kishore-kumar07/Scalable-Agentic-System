# Scalable Agentic System: PayPal Tools

An agent that uses 111 PayPal API tools from a chat interface. It does not show the LLM all the tools. For each message it searches a tool index and gives the LLM only the 6-10 most relevant ones. Write and money actions need human approval.

**API execution is mocked.** All data the agent returns (transactions, disputes, invoice IDs) is fake. This project tests the routing and safety design, not a live PayPal integration.

## How it works

```text
user message -> retrieve top-k tools -> agent (LLM) -> gate (validate + approval) -> execute -> reply
```

- **Retrieval:** Azure OpenAI embeddings in FAISS, plus a keyword-match score.
- **Agent:** a single LangGraph graph with checkpointed sessions.
- **Gate:** validates parameters and asks the user to approve any write or money action. Amounts above the limit need the user to type `APPROVE`.
- **Extra tools:** `knowledge_search` (RAG over sample docs) and `system_search` (tool catalog and request log).

## Requirements

- Python 3.10 or newer (tested on 3.13)
- An Azure OpenAI resource with one chat deployment and one embedding deployment

## Setup (Windows, Command Prompt)

```cmd
py -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and fill in your values:

```text
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your key>
AZURE_OPENAI_API_VERSION=<api version>
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat deployment name>
AZURE_OPENAI_EMBED_DEPLOYMENT=<embedding deployment name>
```

Use the deployment **names** from Azure, not the model names.

Optional settings:

| Variable | Default | Meaning |
|---|---|---|
| `APPROVAL_LIMIT` | 1000 | Amounts above this need the user to type `APPROVE` |
| `MOCK_FAILURE_RATE` | 0.0 | Chance (0 to 1) that a mock API call fails, to test retries |
| `LLM_TEMPERATURE` | unset | Set only if your model rejects the default temperature |
| `LANGSMITH_TRACING` | false | Set to true with a valid `LANGSMITH_API_KEY` to enable tracing |

## Data files

- `data/tools.json`: the parsed tool catalog (included).
- `data/overrides.json`: manual fixes (risk tiers, required flags, simplified invoice and payout schemas).
- `data/paypal_collection.json`: the PayPal Postman collection. It is only needed to rebuild `tools.json`. To get it, open the PayPal collection in Postman (web), choose **Export**, pick **Collection v2.1**, and save it as `data/paypal_collection.json`.

Note: `python run.py catalog` re-parses the Postman file and overwrites `data/tools.json`. Manual fixes live in `overrides.json`, which is applied every time tools are loaded.

## Run

```cmd
python run.py catalog
python run.py chat
python run.py demo
python run.py eval --show
python run.py eval --llm
```

| Command | What it does |
|---|---|
| `catalog` | Parses the Postman collection and prints each tool with its risk tier |
| `chat` | Interactive chat. Type `exit` to quit |
| `demo` | Scripted conversation covering the main scenarios |
| `eval --show` | Retrieval results for each test query (embeddings only, no chat model) |
| `eval --llm` | Retrieval metrics plus LLM tool-selection accuracy, with and without retrieval |

Options: `--k 6` sets the number of retrieved tools, and `--distractors 450` sets the number of synthetic tools added for the scale test. `chat` and `demo` also accept `--distractors` to run the agent against a larger catalog.

## Example session

```text
you> Send an invoice for $50

agent> I need the recipient's email address to create and send the invoice.

you> john@example.com, for consulting services

=== APPROVAL REQUIRED ===

  invoices_create_draft_invoice  risk=write  amount=50.0

  Confirm? (y/n): y

=== APPROVAL REQUIRED ===

  invoices_send_invoice  risk=write  amount=0.0

  Confirm? (y/n): y

agent> Invoice sent to john@example.com for consulting services, amount $50.
```

The full output is in `demo_transcript.txt`.

## Evaluation results

26 test queries, each with one correct tool. The 561-tool catalog is the 111 PayPal tools plus 450 synthetic tools (including Stripe, Square and QuickBooks tools that overlap with PayPal).

| Catalog | Recall@6 | Top-1 | LLM accuracy (top-6 shown) | LLM accuracy (all tools shown) | Prompt tokens (all tools) | Prompt tokens (top-6) |
|---|---|---|---|---|---|---|
| 111 tools | 0.92 | 0.85 | 0.92 | 1.00 | ~23,400 | ~1,500 |
| 561 tools | 0.92 | 0.77 | 0.73-0.81 | rejected by API | ~59,500 | ~1,100 |

Token counts are estimates (characters / 4). LLM accuracy varies between runs. Azure rejects a request with more than 128 tools, so showing all tools is not possible at 561. Results are in `eval/results.md`, and the queries are in `eval/cases.json`.

## Safety features

- Every tool is tagged `read`, `write` or `money_movement`.
- Human approval before any write or money action.
- `APPROVE` must be typed for amounts above the limit.
- Idempotency key on every action, so a retry cannot repeat a payment.
- Retry with backoff on API failures, capped at 3 attempts and 6 LLM steps per turn.
- Emails and long numbers are masked in the audit log (`data/audit_log.jsonl`).
- Access-token and card-setup tools are removed from the catalog.

## Project files

| File | Contents |
|---|---|
| `catalog.py` | Postman parser, overrides, embeddings, retriever, RAG knowledge base, synthetic tools |
| `agent.py` | LangGraph agent, approval gate, mock API, audit log |
| `run.py` | Command-line interface and evaluation |
| `data/` | Tool catalog, overrides, and (optionally) the Postman export |
| `eval/` | Test queries and results |

## Limitations

- The API is mocked. Nothing was tested against live PayPal.
- The 26 test queries were written by me and the system was tuned while looking at them, so the numbers are optimistic.
- Only the invoice and payout tools have simplified schemas. Other write tools still expose raw parameters.
- The RAG documents are sample text I wrote, not official PayPal documentation.
- The approval prompt for sending an invoice shows only the invoice ID, not the recipient or amount.
- Session state is in memory and is lost on restart.
- Not built: two-stage routing, a reranker, sub-agents, MCP, and a persistent checkpointer.
- LangSmith tracing was configured, but my API key returned a 403, so there is no trace.
