"""
eval.py
Runs the agent against customer_queries_sample.csv and checks whether it routed
to the correct tool (or correctly asked a clarifying question when info was missing).
Prints an accuracy score — useful evidence for a capstone submission/demo.

Run with: python eval.py
"""

import csv
from agent import SupportAgent

# Map each dataset intent -> the tool we expect the agent to call.
INTENT_TO_TOOL = {
    "order_status": "get_order_status",
    "return_process": "search_knowledge_base",
    "payment_failure": "search_knowledge_base",
    "shipping_charges": "search_knowledge_base",
    "contact_support": "search_knowledge_base",
    "product_spec_query": "get_product_info",
}


def run_eval(csv_path="data/customer_queries_sample.csv"):
    agent = SupportAgent(data_dir="data")

    correct = 0
    total = 0
    rows_out = []

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            query = row["query"]
            intent = row["intent"]
            notes = row.get("notes", "")

            result = agent.ask(query)
            tools_called = [tc["tool"] for tc in result["tool_calls"]]
            expected_tool = INTENT_TO_TOOL.get(intent)

            # order_status with no order ID given -> expect NO tool call (should ask for it)
            no_id_given = "no order id" in notes.lower()
            if intent == "order_status" and no_id_given:
                is_correct = len(tools_called) == 0
            else:
                is_correct = expected_tool in tools_called

            correct += int(is_correct)
            rows_out.append({
                "query": query,
                "intent": intent,
                "expected_tool": expected_tool if not no_id_given else "(ask for order ID)",
                "actual_tools": tools_called,
                "correct": is_correct,
                "answer": result["answer"],
            })

    print(f"{'QUERY':45} {'EXPECTED':22} {'ACTUAL':30} {'OK'}")
    print("-" * 105)
    for r in rows_out:
        print(f"{r['query'][:44]:45} {str(r['expected_tool'])[:21]:22} {str(r['actual_tools'])[:29]:30} {'✅' if r['correct'] else '❌'}")

    accuracy = correct / total * 100 if total else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")
    return rows_out, accuracy


if __name__ == "__main__":
    run_eval()