import asyncio
import operator
from typing import Annotated
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph_sdk.schema import Thread
from serpapi import GoogleSearch
import os
from data_fetchers import fetch_active_markets
from models import Market
from pydantic import BaseModel, validator
from langgraph_sdk import get_client
import json


# Define the state
class AgentState(BaseModel):
    market: Market
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] = []

    @validator("intermediate_steps")
    def validate_steps(cls, v):
        if isinstance(v, list):
            return [item if isinstance(item, tuple) else (item, item.log) for item in v]
        return v


@tool("web_search")
def web_search(query: str) -> str:
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
def final_answer(report: str) -> str:
    """Returns a market analysis report. The report should be formatted as:
    {
        "introduction": "Brief overview of the market question and current odds",
        "research_steps": ["Step 1", "Step 2", ...],
        "main_body": "3-4 paragraphs analyzing market probability",
        "conclusion": "Final probability assessment and recommendation",
        "sources": ["Source 1", "Source 2", ...]
    }
    """
    try:
        if isinstance(report, str):
            report_dict = json.loads(report)
        else:
            report_dict = report

        introduction = report_dict.get("introduction", "")
        research_steps = report_dict.get("research_steps", [])
        main_body = report_dict.get("main_body", "")
        conclusion = report_dict.get("conclusion", "")
        sources = report_dict.get("sources", [])

        research_steps_str = "\n".join([f"- {r}" for r in research_steps])
        sources_str = "\n".join([f"- {s}" for s in sources])

        return f"""
Market Analysis Report
---------------------

{introduction}

Research Steps:
{research_steps_str}

Analysis:
{main_body}

Conclusion:
{conclusion}

Sources:
{sources_str}
"""
    except json.JSONDecodeError:
        return f"Error: Could not parse report format. Received: {report}"


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

When ready, use final_answer to provide a detailed market analysis in the following format:
{
    "introduction": "Brief overview of the market and current odds",
    "research_steps": ["List of research steps taken"],
    "main_body": "Detailed analysis of findings",
    "conclusion": "Final probability assessment",
    "sources": ["List of sources used"]
}
"""

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


def create_scratchpad(intermediate_steps: list[tuple[AgentAction, str]]):
    research_steps = []
    for action, output in intermediate_steps:
        if output != "TBD":
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n" f"Output: {output}"
            )
    return "\n---\n".join(research_steps)


oracle = (
    {
        "market_details": lambda x: format_market_details(x.market),
        "chat_history": lambda x: x.chat_history,
        "scratchpad": lambda x: create_scratchpad(x.intermediate_steps),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)


def run_oracle(state: AgentState):
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")

    return AgentState(
        market=state.market,
        chat_history=state.chat_history,
        intermediate_steps=[(action, "TBD")],
    )


def router(state: AgentState):
    if state.intermediate_steps:
        action, _ = state.intermediate_steps[-1]  # Properly unpack the tuple
        return action.tool
    return "final_answer"


tool_str_to_func = {
    "web_search": web_search,
    "final_answer": final_answer,
}


def run_tool(state: AgentState):
    last_action, _ = state.intermediate_steps[-1]
    tool_name = last_action.tool
    tool_args = last_action.tool_input

    # Extract the single argument value from the args dictionary
    if isinstance(tool_args, dict):
        if tool_name == "web_search":
            arg_value = tool_args.get("query", "")
        elif tool_name == "final_answer":
            arg_value = json.dumps(tool_args)  # Convert dict to JSON string
        else:
            arg_value = next(iter(tool_args.values()), "")
    else:
        arg_value = tool_args

    # Call the tool with a single argument
    out = tool_str_to_func[tool_name](arg_value)
    action = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))

    return AgentState(
        market=state.market,
        chat_history=state.chat_history,
        intermediate_steps=[(action, str(out))],
    )


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

graph: CompiledStateGraph = builder.compile()


async def main():
    URL = "http://localhost:50860"
    client = get_client(url=URL)

    thread: Thread = await client.threads.create()

    market = fetch_active_markets()[0]  # Get first active market
    initial_state = AgentState(market=market, chat_history=[], intermediate_steps=[])

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="research_agent",
        input=initial_state.model_dump(),  # Convert to dict for JSON serialization
    )
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
