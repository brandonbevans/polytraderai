from typing import List
import uuid
import asyncio
from app.models import ArticleMarketMatchFull
from app.news.main import get_relevant_articles
from app.data_fetchers import fetch_active_markets, fetch_markets_with_positions
from app.models import (
    GenerateAnalystsState,
    Market,
    RecentNewsResearchMarketState,
)
from app.graph import get_full_graph, get_news_graph
import time


def manage_positions():
    """Manage positions for a list of markets."""
    markets: List[Market] = fetch_markets_with_positions()
    print(len(markets))


def main():
    thread_id = str(uuid.uuid4())
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
    graph = get_full_graph()
    markets = fetch_active_markets()[:30]
    for market in markets:
        initial_state = GenerateAnalystsState(market=market, max_analysts=3)
        graph.invoke(initial_state.model_dump(), config=thread)
        time.sleep(30)


async def main_news():
    thread_id = str(uuid.uuid4())
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
    graph = get_news_graph()
    market_with_articles_list: List[
        ArticleMarketMatchFull
    ] = await get_relevant_articles()
    for market_with_articles in market_with_articles_list:
        initial_state = RecentNewsResearchMarketState(
            market=market_with_articles.market,
            articles=market_with_articles.articles,
        )
        graph.invoke(initial_state.model_dump(), config=thread)
        time.sleep(5)


if __name__ == "__main__":
    asyncio.run(main_news())
