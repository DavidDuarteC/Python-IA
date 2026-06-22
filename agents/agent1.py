import logging
from datetime import datetime

from dotenv import load_dotenv

from agentspan.agents import Agent, AgentRuntime, ConversationMemory, run, tool

load_dotenv(override=True)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("agentspan").setLevel(logging.WARNING)
logging.getLogger("conductor").setLevel(logging.WARNING)

assistant = Agent(
    name="personal_assistant",
    model="hugging_face/meta-llama/Llama-3-70b-chat-hf",
    instructions=(
        "You are a concise personal assistant. Use tools when they help "
        "and remember useful user details across turns"
    ),
    tools=[]
)

if __name__ == "__main__":
    print("Starting agent...")

    with AgentRuntime() as runtime:
        while True:
            prompt = input("User: ").strip()
            if prompt.lower() in {"exit", "quit", "q"}:
                print("Exiting...")
                break
            if not prompt:
                continue
            result = run(assistant, prompt, runtime=runtime)
            print(f"Assistant: {result}")