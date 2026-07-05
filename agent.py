"""
agent.py
The core assistant. One Groq tool-calling loop that can:
  - search_knowledge_base  -> RAG over FAQ.md / Policies.md
  - get_product_info       -> product catalog lookup
  - get_order_status       -> mock order lookup

Design notes:
- temperature=0 keeps tool-call formatting reliable (less random glitches).
- FAQ/Policy answers are shown using the retrieved text directly rather than
  having the LLM retype them — guarantees prices/dates/numbers are always
  exact, since LLMs can occasionally mis-type digits when paraphrasing.
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

from retriever import KnowledgeBase
from tools import TOOL_SCHEMAS, KB_TOOL_SCHEMA, AVAILABLE_FUNCTIONS

load_dotenv()  # reads GROQ_API_KEY from .env

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a helpful e-commerce customer support assistant.

Rules:
- Always use the provided tools to get facts (policies, product specs, order status). Never invent policy details, prices, specs, or order info from memory.
- CRITICAL: Any number, price, date, or ID from a tool result must be copied character-by-character, exactly as written. Do not shorten, round, or retype it from memory.
- For order status: NEVER call get_order_status unless the customer's message contains an actual order ID (e.g. a code like "ORD1001"). If no order ID is present in their message, do not call any tool — just ask them for their order ID in plain text.
- For returns: if the customer is asking about the return policy in general, use search_knowledge_base. Only call create_return_request if they've given a specific order ID AND a reason for the return. If they've given an order ID but no reason, ask for the reason first.
- If any tool returns an error (product/order not found, return not eligible, etc.), explain the error clearly to the customer in plain language.
- Escalate to a human: if a tool result is an error you cannot resolve any other way, or the customer explicitly asks for a human/agent, or the same issue remains unresolved after you've already tried to help once, call escalate_to_human with a short summary of the issue.
- Keep answers short, friendly, and directly useful (2-4 sentences typically).
- If the question is outside e-commerce support entirely, politely say you can only help with orders, products, shipping, returns, payments, and related support topics.
"""


class SupportAgent:
    def __init__(self, data_dir="data"):
        self.kb = KnowledgeBase(data_dir=data_dir)
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

        def search_knowledge_base(query: str):
            results = self.kb.search(query, top_k=2)
            if not results:
                return {"error": "No relevant policy/FAQ information found."}
            return {"results": [chunk for chunk, score in results]}

        self.functions = {
            **AVAILABLE_FUNCTIONS,
            "search_knowledge_base": search_knowledge_base,
        }
        self.tool_schemas = TOOL_SCHEMAS + [KB_TOOL_SCHEMA]

    def _call_with_tools(self, messages):
        """Call Groq with tools, retrying once if the model garbles a tool call
        (a known occasional quirk, not a logic bug)."""
        last_error = None
        for attempt in range(2):
            try:
                return self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    temperature=0,
                )
            except Exception as e:
                last_error = e
                continue
        raise last_error

    def _format_kb_answer(self, kb_result: dict) -> str:
        """Turn a raw FAQ/Policy chunk into a clean answer, showing the original
        text as-is so numbers/prices/dates can never get mis-typed."""
        chunk = kb_result["results"][0]
        if "\nA: " in chunk:
            answer = chunk.split("\nA: ", 1)[1].strip()
        else:
            lines = chunk.split("\n", 1)
            answer = lines[1].strip() if len(lines) > 1 else chunk.strip()
        return answer

    def ask(self, user_message: str, history: list = None, verbose: bool = False) -> dict:
        """
        Send a user message through the tool-calling loop.
        Returns dict: {"answer": str, "tool_calls": [...], "history": updated_history}
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tool_call_log = []

        response = self._call_with_tools(messages)
        response_message = response.choices[0].message

        if response_message.tool_calls:
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                fn = self.functions.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"Unknown tool {fn_name}"}

                tool_call_log.append({"tool": fn_name, "args": fn_args, "result": result})
                if verbose:
                    print(f"  [tool call] {fn_name}({fn_args}) -> {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": json.dumps(result),
                })

            # If the only tool used was the knowledge base search and it succeeded,
            # show the retrieved answer directly instead of letting the LLM retype it.
            if (
                len(tool_call_log) == 1
                and tool_call_log[0]["tool"] == "search_knowledge_base"
                and "results" in tool_call_log[0]["result"]
            ):
                answer = self._format_kb_answer(tool_call_log[0]["result"])
            else:
                final_response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0,
                )
                answer = final_response.choices[0].message.content

            messages.append({"role": "assistant", "content": answer})
        else:
            answer = response_message.content
            messages.append({"role": "assistant", "content": answer})

        updated_history = messages[1:]
        return {"answer": answer, "tool_calls": tool_call_log, "history": updated_history}


if __name__ == "__main__":
    agent = SupportAgent(data_dir="data")

    test_queries = [
        "Where is my order?",
        "How do I return a product?",
        "My payment failed but money was deducted.",
        "Do you charge for shipping?",
        "How do I contact customer support?",
        "Does the 24-inch monitor have HDMI?",
        "What's the status of order ORD1001?",
        "I want to return order ORD1003, the item arrived damaged",
        "Can I return order ORD1001? I changed my mind",
        "This isn't helping, I want to talk to a real person",
    ]
    for q in test_queries:
        print(f"\nUSER: {q}")
        result = agent.ask(q, verbose=True)
        print(f"BOT: {result['answer']}")