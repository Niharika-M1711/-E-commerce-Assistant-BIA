# E-Commerce Support Assistant (RAG + Tool-Calling)

An AI customer support assistant built for the BIA Capstone Project: **E-Commerce Customer Support Assistant**.

It resolves common e-commerce support queries — orders, returns, payments, shipping, and product questions — by combining **Retrieval-Augmented Generation (RAG)** over FAQ/policy documents with **LLM tool-calling** against structured data sources, instead of answering from memory.

## Problem Scope

E-commerce platforms receive high volumes of repetitive queries. This assistant automates the most common ones:

| Query type | How it's handled |
|---|---|
| Policy / FAQ questions (returns, shipping, payments, cancellations) | RAG search over `FAQ.md` / `Policies.md` |
| Product spec questions | Tool call to the product catalog (`Products.json`) |
| Order status questions | Tool call to a mock order database, asks for an order ID if missing |
| Off-topic questions | Politely declines and redirects to supported topics |

**Out of scope for this version:** live payment processing, real order/inventory systems, multi-language support, human-agent handoff (see Limitations).

## Architecture

```
User question
     |
     v
Groq LLM (Llama 3.3 70B) decides which tool to call, if any:
  - search_knowledge_base(query)   -> TF-IDF retrieval over FAQ.md / Policies.md
  - get_product_info(product_name) -> Products.json lookup
  - get_order_status(order_id)     -> mock order database lookup
     |
     v
Tool result -> natural language answer
(FAQ/Policy answers are shown using the retrieved text directly, not
 re-typed by the LLM, so prices/dates/IDs are always exact)
```

**Why TF-IDF instead of embeddings/vector DB?** The knowledge base is small (~25 chunks). A full embedding pipeline adds cost and latency without improving retrieval quality at this scale — TF-IDF cosine similarity is fast, free, and accurate enough for keyword-heavy support questions.

## Files

| File | Purpose |
|---|---|
| `retriever.py` | Chunks FAQ/Policies and does TF-IDF similarity search |
| `tools.py` | Product lookup + order status functions, and their tool schemas |
| `agent.py` | Core Groq tool-calling loop that ties everything together |
| `app.py` | Streamlit chat UI |
| `eval.py` | Scores the agent against `data/customer_queries_sample.csv` |
| `data/` | FAQ, Policies, Products, and sample queries (provided dataset) |

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get a free Groq API key: https://console.groq.com/keys

3. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```

## Run

**Test the core logic in terminal:**
```
python agent.py
```
Runs 7 sample questions and prints the bot's answers plus which tools it called.

**Score against the evaluation dataset:**
```
python eval.py
```
Prints a table of expected vs. actual tool used per query, plus an accuracy percentage.

**Launch the chat UI:**
```
streamlit run app.py
```
Opens an interactive chat interface with sample-query buttons in the sidebar.

## Evaluation

`eval.py` checks whether the assistant routed each sample query to the correct tool (or correctly asked a clarifying question when required information, like an order ID, was missing).

**Current result: 6/6 (100%) tool-routing accuracy** on the provided sample query set.

## Key Design Decisions

- **Exact numbers, never retyped:** Early testing showed the LLM could mistype a price when paraphrasing (₹999 → ₹99). Fix: FAQ/policy answers are shown using the retrieved text directly instead of letting the LLM rephrase them.
- **Ask, don't assume:** If a customer asks about order status without providing an order ID, the assistant asks for it rather than calling the tool with a guessed value.
- **Low temperature (0) + retry:** Reduces occasional malformed tool-call generation from the LLM.

## Limitations & Future Work

- Order and product data are mocked (`MOCK_ORDERS` in `tools.py`, sample `Products.json`) rather than connected to a live backend.
- No `create_return_request` tool yet — return questions are currently answered via policy text only, not an actionable return flow.
- No explicit "escalate to a human agent" path for queries the assistant can't resolve.
- No multi-language support.
- Evaluation is limited to the 6 sample queries provided in the dataset; a larger labeled test set would give a more robust accuracy estimate.

## Author

Niharika Mittapelly — niharikamittapelly@gmail.com — github.com/Niharika-M1711
```

