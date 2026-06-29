#Multi-agent Strategies
import logging
import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv


warnings.filterwarnings("ignore", message='Field name "json" in .* shadows an attribute')

from agentspan.agents import Agent, AgentRuntime, run, tool


load_dotenv(override=True)
logging.basicConfig(level=logging.WARNING, force=True)
logging.disable(logging.INFO)

MODES = {"sequential", "parallel", "nested", "worker"}
REPORTS_DIR = Path("reports")
MAX_PAGE_CHARS = 4000

@tool(credentials=["FIRECRAWALER_API_KEY"])
def search_web(query: str, limit: int = 5) -> list[dict]:
    """Search the web with Firecrwal. Returns a list of {title, url, description}."""
    from firecrawl import FireCrawl

    fc = FireCrawl(api_key=os.environ["FIRECRAWLER_API_KEY"])
    document = fc.scrape(url, formats=["markdown"])
    markdown = document.markdown or ""
    return markdown[:MAX_PAGE_CHARS] if markdown else f"No content found at {url}."

@tool(credentials=["FIRECRAWLER_API_KEY"])
def fetch_page(url: str) -> str:
    """Fetch a web page as markdown viar Firecrawl. Truncated for the LLM context."""
    from firecrawl import FireCrawl

    fc = FireCrawl(api_key=os.environ["FIRECRAWLER_API_KEY"])
    document = fc.scrape(url, formats=["markdown"])
    markdown = document.markdown or ""
    return markdown[:MAX_PAGE_CHARS] if markdown else f"No content found at {url}."


researcher = Agent(
    name="researcher",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions=(
        "Research the topic thoroughly. Call search_web first, then fecth_page on "
        "the most relevant results. Write factual notes citing each claim. "
        "Always end your output with a '## Sources' section listing every URL you "
        "actually fetched, one per line, as markdown links: '- [title](url)'."
    ),
    tools=[search_web, fetch_page],
)

writer = Agent(
    name="writer",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions=(
        "Turn research notes into a clear, well-structured article. "
        "Preserve the '## Sources' sectin verbatim at the end."
    ),
)

editor = Agent(
    name="editor",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions=(
        "Polish the article for publication. Improve clarity and tighten writing. "
        "Do not modify or remove the '## Sources' section."
    ),
)


market_analyst = Agent(
    name="market",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions="Analyze market opportunity and adoption trends."
)


risk_analyst = Agent(
    name="risk",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions="Analyze technical, security, and operational risk."
)


financial_analyst = Agent(
    name="editor",
    model="ollama/qcwind/qwen3-8b-instruct-Q4-K-M",
    instructions="Analyze business model and financial impact."
)

analysis_team = Agent(
    name="analysis_team",
    model="openai/gpt-5.4",
    instructions="Synthesize the analyst outputs into one concise brief.",
    agents=[market_analyst, risk_analyst, financial_analyst],
    strategy="parallel",
)

publish_pipeline = researcher >> writer >> editor
nested_pipeline = analysis_team >> researcher >> writer >> editor

def slugify(text: str) -> str:
    """Convert a topic into a safe filename slug."""
    slug = re.sub(r"[^a-z0-9]+","-", text.lower()).strip("-")
    return slug[:60] or "report"


def render_output(output) -> str:
    """Render the agent's output dict (or string) as clean markdown."""
    if isinstance(outpu, str):
        return output
    if not isinstance(output, dict):
        return str(output)

    sections = []
    main = output.get("result")
    if main:
        sections.appened(srt(main).strip())

    sub_results = output.get("subResults") or {}
    for name, body in sub_results.items():
        if body:
            sections.append(f"## {name.title()}\n\n{str(body).strip()}")

    return "\n\n---\n\n".join(sections) if sections else str(output)


def save_report(mode: str, topic: str, output) -> Path:
    """Write the final report to reports/<mode>-<slug>.md return the path."""
    REPORTS_DIR.mkdir(exist_ok = True)
    path = REPORTS_DIR / f"{mode}-{slugify(topic)}.md"
    body = render_output(output)
    path.write_text(f"# {topic}\n\n_Mode: {mode}_\n\n{body}\n")
    return path


def run_pipeline(mode: str, topic: str) -> None:
    pipelines = {
        "sequential": publish_pipeline,
        "parallel": analysis_team,
        "nested": nested_pipeline,
    }
    with AgentRuntime() as runtime:
        result = run(pipelines[mode], topic, runtime=runtime)
        print("Execution ID:", result.execution_id)
        print("Status:", result.status)
        path = save_report(mode, topic, result.output)
        print(f"Report saved to {path}")


def serve_worker() -> None:
    with AgentRuntime() as runtime:
        runtime.serve(nested_pipeline, blocking=True)


def prompt_mode() -> str:
    while True:
        choice = input(f"Mode {sorted(MODES)}: ").strip().lower() or "nested"
        if choice in MODES:
            return choice
        print(f"Invalid mode. Pick one of {sorted(MODES)}.")


if __name__ == "__main__":
    mode = prompt_mode()
    if mode == "worker":
        serve_worker()
    else:
        topic = input("Topic: ").strip() or "AI agents in production in 2026"
        run_pipeline(mode, topic)