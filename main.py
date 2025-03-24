from typing import List
import uuid
from data_fetchers import fetch_active_markets, fetch_markets_with_positions
from models import (
    GenerateAnalystsState,
    Market,
)
from graph import get_full_graph
import time


def manage_positions():
    """Manage positions for a list of markets."""
    markets: List[Market] = fetch_markets_with_positions()
    print(len(markets))


def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_full_graph()

    markets = fetch_active_markets()[:30]
    for market in markets:
        initial_state = GenerateAnalystsState(market=market, max_analysts=3)
        graph.invoke(initial_state.model_dump(), config)
        time.sleep(10)


if __name__ == "__main__":
    main()
