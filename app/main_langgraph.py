import asyncio
import uuid
from langgraph.graph.state import CompiledStateGraph
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient
from app.news.main import get_relevant_articles
from app.data_fetchers import fetch_active_markets
from langgraph_sdk.schema import Thread

from app.models import (
    GenerateAnalystsState,
    RecentNewsResearchMarketState,
)
from app.graph import get_full_graph

URL = "http://localhost:2024"
graph: CompiledStateGraph = get_full_graph()


async def main():
    client: LangGraphClient = get_client(url=URL)
    thread_id = str(uuid.uuid4())
    thread: Thread = await client.threads.create(thread_id=thread_id)
    thread = {
        "configurable": {
            "thread_id": thread_id,
            "search_api": "tavily",
            "planner_provider": "anthropic",
            "planner_model": "claude-3-7-sonnet-latest",
            "writer_provider": "openai",
            "writer_model": "gpt-4o",
            "max_search_depth": 1,
        }
    }

    market = fetch_active_markets()[0]  # Get first active market
    initial_state = GenerateAnalystsState(market=market)

    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id="research_agent",
        input=initial_state.model_dump(),
        config=thread,
    )
    print(run)
    observe_state(thread_id)


async def news_agent():
    client: LangGraphClient = get_client(url=URL)
    thread_id = str(uuid.uuid4())
    thread: Thread = await client.threads.create(thread_id=thread_id)
    thread = {"configurable": {"thread_id": thread_id, "search_api": "tavily"}}

    article_market_match_fulls = await get_relevant_articles()
    initial_state = RecentNewsResearchMarketState(
        market=article_market_match_fulls[0].market,
        articles=article_market_match_fulls[0].articles,
    )

    run = await client.runs.create(
        thread_id=thread_id,
        assistant_id="news_agent",
        input=initial_state.model_dump(),
        config=thread,
    )
    print(run)
    observe_state(thread_id)


def observe_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config=config)
    print(state)


if __name__ == "__main__":
    asyncio.run(news_agent())
