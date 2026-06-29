#Memory & basic Tools
import logging
from datetime import datetime

from dotenv import load_dotenv

from agentspan.agents import Agent, AgentRuntime, tool

load_dotenv(override=True)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("agentspan").setLevel(logging.WARNING)
logging.getLogger("conductor").setLevel(logging.WARNING)

@tool
def get_current_time() -> str:
    """returns the current local time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

assistant = Agent(
    name="personal_assistant",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions=(
        "You are a helpful personal assistant. "
        "You have memory of our conversation. Use it to remember details about the user."
    ),
    tools=[get_current_time],
)

def build_context_prompt(messages, new_prompt):
    if not messages:
        return new_prompt
    context = "This is our conversation history:\n"
    for m in messages:
        context += f"{m['role'].capitalize()}: {m['content']}\n"
    context += f"\nNow respond to: {new_prompt}"
    return context

if __name__ == "__main__":
    print("Starting agent...")

    messages = []

    with AgentRuntime() as runtime:
        while True:
            prompt = input("User: ").strip()
            if prompt.lower() in {"exit", "quit", "q"}:
                print("Exiting...")
                break
            if not prompt:
                continue

            full_prompt = build_context_prompt(messages, prompt)
            messages.append({"role": "user", "content": prompt})

            print("Assistant: ", end="", flush=True)
            response = ""
            for event in runtime.stream(assistant, full_prompt):
                if event.type == "done":
                    result = event.output
                    if isinstance(result, dict):
                        response = result.get("result", str(result))
                    else:
                        response = str(result)
                    print(response)
                    messages.append({"role": "assistant", "content": response})
