import asyncio
from langgraph.graph.state import CompiledStateGraph
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient
from data_fetchers import fetch_active_markets
from langgraph_sdk.schema import Thread

from models import (
    GenerateAnalystsState,
)
from graph import get_full_graph

URL = "http://localhost:2024"
graph: CompiledStateGraph = get_full_graph()


async def main():
    client: LangGraphClient = get_client(url=URL)
    thread: Thread = await client.threads.create()

    market = fetch_active_markets()[0]  # Get first active market
    initial_state = GenerateAnalystsState(market=market, max_analysts=1, analysts=[])

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="research_agent",
        input=initial_state.model_dump(),
    )
    print(run)
    observe_state(thread["thread_id"])


def observe_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config=config)
    print(state)


if __name__ == "__main__":
    asyncio.run(main())
