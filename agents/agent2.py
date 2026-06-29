#RAG, Guardrails and HITL
import logging

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentspan.agents import (
    Agent,
    AgentRuntime,
    ConversationMemory,
    EventType,
    Guardrail,
    GuardrailResult,
    OnFail,
    Position,
    guardrail,
    start,
    tool,
)

load_dotenv(override=True)
logging.basicConfig(level=logging.WARNING, force=True)
logging.disable(logging.INFO)


MOCK_DB = {
    "orders": {"A100": {"status": "delivered", "total": 49.99}},
    "accounts": {"tim@example.com": {"status": "active", "tier": "pro"}},
}

DOCS = {
    "refund policy": "Refunds are processed within 5 business days.",
    "shipping": "Standard shipping takes 3 to 7 business days.",
    "account": "Pro accounts include priority support.",
}

class SupportResponse(BaseModel):
    stage: str = Field(description="Stage like answered, refunded or rejected")
    successful: bool
    message: str

@tool
def search_knowledge_base(query: str) -> str:
    """search support docs"""
    for title, body in DOCS.items():
        if title in query.lower():
            return body
    return "No matching support articles found."

@tool
def lookup_order(order_id: str) -> dict:
    """lookup order in datbase by ID"""
    return MOCK_DB["orders"].get(order_id, {"error": "order not found"})

@tool(approval_required=True)
def process_refund(order_id: str, amount: float) -> str:
    """request a refund for human approval think before you run this"""
    return f"Refund {amount:.2f} for order {order_id}"

@guardrail
def safe_support_request(prompt: str) -> GuardrailResult:
    """Block obvoius prompt injection attempts"""
    blocked = ["ignore", "ignore previous", "system prompt", "jailbreak"]
    passed = not any(phrase in prompt.lower() for phrase in blocked)
    return GuardrailResult(passed=passed, message="Please ask a normal question, this is blocked.")       

support_agent = Agent(
    name ="support_agent",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions=(
        "You are a costumer support agent. User the knowledge base first."
        "If the costumer wants a refund: when you know the order ID, call"
        "lookup_order to get the amount. Before callling process_refunnd,"
        "write a short plain-English sentence describing exactly what refound "
        "you are about to issue, for expample: 'I am goint to refund $49.99 "
        "for order A100.' Then call process_refund. The tool will pause for "
        "human approval automatically. If the order ID is missing, ask the "
        "customer for it. Always populate the meassage field with a clear reply."
    ),
    output_type=SupportResponse,
    tools=[search_knowledge_base, lookup_order, process_refund],
    memory=ConversationMemory(max_messages=50),
    guardrails=[Guardrail(safe_support_request, position=Position.INPUT, on_fail=OnFail.RAISE)],
    max_turns=10,
)

def run_interactive(prompt: str) -> None:
    with AgentRuntime() as runtime:
        handle = start(support_agent, prompt, runtime=runtime)
        stream = handle.stream()

        order_id, amount = None, None
        for event in stream:
            if event.type == EventType.TOOL_CALL and event.args:
                order_id = event.args.get("order_id") or order_id
            elif event.type == EventType.TOOL_RESULT and isinstance(event.result, dict):
                amount = event.result.get("total") or amount
            elif event.type == EventType.WAITING:
                print(f"\nAprroval required: refund ${amount:.2f} for order {order_id}")
                decision = input("Approve? (y/n): ").lower().strip()
                if decision == "y":
                    handle.approve()
                else:
                    handle.reject("user rejected the refund")

        result = stream.get_result()
        output = result.output.get("result")
        print(f"\n{output}\n")

if __name__ == "__main__":
    print("Starting bot agent...")
    while True:
        prompt = input("User: ").strip()
        if prompt.lower() in {"exit", "quit", "q"}:
            print("Exiting...")
            break
        if not prompt:
            continue
        run_interactive(prompt)