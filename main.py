import uuid
from data_fetchers import fetch_active_markets
from models import (
    GenerateAnalystsState,
)
from graph import get_full_graph
import time


def main():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_full_graph()

    markets = fetch_active_markets()[:10]
    for market in markets:
        initial_state = GenerateAnalystsState(market=market, max_analysts=3)
        graph.invoke(initial_state.model_dump(), config)
        time.sleep(10)


if __name__ == "__main__":
    main()
