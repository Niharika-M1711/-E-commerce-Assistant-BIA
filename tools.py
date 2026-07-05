"""
tools.py
Functions the assistant can call when a question needs real data instead of
document text — product specs, order status, return requests, and escalation.
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

with open(os.path.join(DATA_DIR, "Products.json"), encoding="utf-8") as f:
    PRODUCTS = json.load(f)

# Mock order database: order_id -> order info
MOCK_ORDERS = {
    "ORD1001": {"status": "Shipped", "eta": "2026-07-05", "items": ["Wireless Earbuds"]},
    "ORD1002": {"status": "Processing", "eta": "2026-07-08", "items": ["24-inch Monitor (Full HD)"]},
    "ORD1003": {"status": "Delivered", "eta": "2026-06-28", "items": ["65W GaN Charger"]},
}

# Mock return requests database (starts empty, fills up as customers request returns)
RETURN_REQUESTS = {}
_next_return_id = 1
_next_ticket_id = 1


def get_product_info(product_name: str) -> dict:
    """Find a product by (partial, case-insensitive) name match and return its full details."""
    product_name_lower = product_name.lower()
    for p in PRODUCTS:
        if product_name_lower in p["name"].lower():
            return p
    return {"error": f"No product found matching '{product_name}'"}


def get_order_status(order_id: str) -> dict:
    """Look up order status by order ID."""
    order = MOCK_ORDERS.get(order_id.upper())
    if not order:
        return {"error": f"No order found with ID '{order_id}'"}
    return {"order_id": order_id.upper(), **order}


def create_return_request(order_id: str, reason: str) -> dict:
    """Create a return request for a delivered order. Returns a request ID and status."""
    global _next_return_id

    order_id = order_id.upper()
    order = MOCK_ORDERS.get(order_id)
    if not order:
        return {"error": f"Cannot create a return: no order found with ID '{order_id}'"}

    if order["status"] != "Delivered":
        return {
            "error": f"Order '{order_id}' has status '{order['status']}'. "
                     f"Returns can only be requested for delivered orders."
        }

    request_id = f"RET{1000 + _next_return_id}"
    _next_return_id += 1

    RETURN_REQUESTS[request_id] = {
        "order_id": order_id,
        "reason": reason,
        "status": "Pending pickup",
        "items": order["items"],
    }

    return {
        "request_id": request_id,
        "order_id": order_id,
        "status": "Pending pickup",
        "reason": reason,
        "message": "Return request created. A courier pickup will be scheduled within 2 business days.",
    }


def escalate_to_human(issue_summary: str) -> dict:
    """Create a support ticket to escalate an issue to a human agent."""
    global _next_ticket_id

    ticket_id = f"TICKET{1000 + _next_ticket_id}"
    _next_ticket_id += 1

    return {
        "ticket_id": ticket_id,
        "status": "Escalated",
        "issue_summary": issue_summary,
        "message": f"This has been escalated to our support team (ticket {ticket_id}). "
                    f"They'll follow up via support@example.com within 24 hours.",
    }


# ---- Tool schemas for Groq function-calling ----
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_info",
            "description": "Get details/specs/price/stock for a product by name. Use this for any question about a specific product's features, price, or availability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name or a key part of it, e.g. 'monitor' or 'earbuds'",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status of a customer's order using their order ID. Only call this if the customer has provided an order ID; otherwise ask them for it first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. 'ORD1001'",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return_request",
            "description": "Create a return request for a delivered order. Only call this if the customer has clearly stated they want to return a SPECIFIC order (has given an order ID) and a reason. If they're just asking about the return policy in general, use search_knowledge_base instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to return, e.g. 'ORD1003'",
                    },
                    "reason": {
                        "type": "string",
                        "description": "The customer's stated reason for the return",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the conversation to a human support agent. Use this when the customer's issue cannot be resolved with the other tools, when they explicitly ask for a human, or when the same problem remains unresolved after you've already tried to help.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_summary": {
                        "type": "string",
                        "description": "A short summary of the customer's unresolved issue, for the human agent.",
                    }
                },
                "required": ["issue_summary"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_product_info": get_product_info,
    "get_order_status": get_order_status,
    "create_return_request": create_return_request,
    "escalate_to_human": escalate_to_human,
}

KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search company FAQ and policy documents (shipping, returns, refunds, cancellations, payments, warranty, privacy, support contact). Use this for any general policy/how-to question that isn't about a specific product's specs, a specific order's status, or creating an actual return.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question, used to search the FAQ/policy documents.",
                }
            },
            "required": ["query"],
        },
    },
}


if __name__ == "__main__":
    print(get_product_info("monitor"))
    print(get_product_info("earbuds"))
    print(get_product_info("drone"))
    print(get_order_status("ord1001"))
    print(get_order_status("ORD9999"))
    print(create_return_request("ORD1003", "Item arrived damaged"))
    print(create_return_request("ORD1001", "Changed my mind"))
    print(escalate_to_human("Customer wants a refund method we don't support"))