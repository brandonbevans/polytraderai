import asyncio
import operator
from typing import Annotated, TypedDict
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from serpapi import GoogleSearch
import os
from data_fetchers import fetch_active_markets
from models import Market

from langgraph_sdk import get_client


# Define the state
class AgentState(TypedDict):
    market: Market  # Changed from input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


@tool("web_search")
def web_search(query: str):
    """Finds general knowledge information using Google search."""
    search = GoogleSearch(
        {"engine": "google", "api_key": os.getenv("SERPAPI_KEY"), "q": query, "num": 5}
    )
    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return contexts


@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str,
):
    """Returns a market analysis report with the following sections:
    - `introduction`: Brief overview of the market question and current odds
    - `research_steps`: Bullet points explaining research methodology
    - `main_body`: 3-4 paragraphs analyzing market probability
    - `conclusion`: Final probability assessment and recommendation
    - `sources`: Bulletpoint list of sources used
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return f"{introduction}\n\nResearch Steps:\n{research_steps}\n\n{main_body}\n\n{conclusion}\n\nSources:\n{sources}"


system_prompt = """You are a market analysis AI that evaluates prediction markets.
You will be given a Market object with the following key information:
- question: The market question
- outcome_prices: Current probabilities for outcomes
- outcomes: The possible outcomes (usually Yes/No)
- description: Additional context about the market
- end_date: When the market resolves

Your task is to research and analyze the probability of the market outcomes.
Consider:
1. Current market prices and if they seem justified
2. Historical precedents and relevant data
3. Recent developments that could impact the outcome
4. Time until market resolution

Use the available tools to gather information, but:
- Don't reuse the same search query twice
- Limit each tool to maximum 3 uses
- Gather from diverse sources before concluding

When ready, use final_answer to provide a detailed market analysis."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Analyze market: {market_details}"),
        ("assistant", "Research progress: {scratchpad}"),
    ]
)

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [web_search, final_answer]


def format_market_details(market: Market) -> str:
    """Format market details for the prompt"""
    return f"""
Question: {market.question}
Description: {market.description}
Current Odds: {dict(zip(market.outcomes, market.outcome_prices))}
End Date: {market.end_date}
Volume: {market.volume}
"""


def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for action in intermediate_steps:
        if action.log != "TBD":
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)


oracle = (
    {
        "market_details": lambda x: format_market_details(x["market"]),
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)


def run_oracle(state: AgentState):
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")
    return {"intermediate_steps": [action_out]}


def router(state: AgentState):
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    return "final_answer"


tool_str_to_func = {
    "web_search": web_search,
    "final_answer": final_answer,
}


def run_tool(state: AgentState):
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    out = tool_str_to_func[tool_name](**tool_args)
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
    return {"intermediate_steps": [action_out]}


# Build the graph
builder = StateGraph(AgentState)

builder.add_node("oracle", run_oracle)
builder.add_node("web_search", run_tool)
builder.add_node("final_answer", run_tool)

builder.set_entry_point("oracle")
builder.add_conditional_edges("oracle", router)

for tool_obj in tools:
    if tool_obj.name != "final_answer":
        builder.add_edge(tool_obj.name, "oracle")

builder.add_edge("final_answer", END)

graph = builder.compile()


async def main():
    URL = "http://localhost:63320"
    client = get_client(url=URL)
    thread = await client.threads.create()

    market = fetch_active_markets()[0]  # Get first active market
    input = {"market": market, "chat_history": [], "intermediate_steps": []}
    run = await client.runs.create(
        thread_id=thread.id,
        graph_id=graph.id,
        inputs=input,
    )
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
    # with client.
    # market = fetch_active_markets()[0]  # Get first active market
    # result = graph.invoke(
    #     {"market": market, "chat_history": [], "intermediate_steps": []}
    # )
