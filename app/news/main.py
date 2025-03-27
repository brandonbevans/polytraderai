import feedparser
from fastapi import FastAPI
from datetime import datetime, timedelta
import time
from typing import List
import uvicorn

from app.data_fetchers import fetch_active_markets
from app.models import Market, Article, ArticleMarketMatch
from app.llms import geminiflash


app = FastAPI(title="Simple RSS Feed Endpoint")

# Define a simple RSS feed list
RSS_FEEDS = [
    {"name": "Google News", "url": "https://news.google.com/rss"},
]


@app.get("/get_articles", response_model=List[Article])
async def get_recent_articles(lookback_time: int = 15) -> list[Article]:
    """
    Get articles from the first RSS feed published in the last specified time
    """
    # Use the first feed in the list
    feed_url = RSS_FEEDS[0]["url"]
    feed_name = RSS_FEEDS[0]["name"]
    print(f"Fetching articles from {feed_name} ({feed_url})")

    # Parse the feed
    feed = feedparser.parse(feed_url)

    # Get current time
    now = datetime.now()
    cutoff_time = now - timedelta(minutes=lookback_time)

    # Filter and process entries
    recent_articles = []

    for entry in feed.entries:
        # Extract the publication date
        published_time = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_time = datetime.fromtimestamp(time.mktime(entry.published_parsed))

        # If we can't parse the time, include the article anyway
        if published_time is None or published_time >= cutoff_time:
            article = Article(
                title=entry.title,
                url=entry.link,
                published=entry.published if hasattr(entry, "published") else None,
                summary=entry.summary if hasattr(entry, "summary") else None,
                published_parsed=published_time,
            )
            recent_articles.append(article)

    print(
        f"Found {len(recent_articles)} articles from the last {lookback_time} minutes"
    )
    return recent_articles


MATCH_MARKETS_INSTRUCTIONS = """
You are a market analyst. You are given a list of articles and a list of markets.
You need to match the articles to the markets. A match consists of an a market and any number of articles.
Don't include any articles that don't match the market. It's okay to not match any articles to a market, in fact in most cases that's expected.
You should baiscally base the match on the semantic meaning of the title and the semantic meaning of the market question.
Here are the market questions:
{markets}

Here are the article titles:
{articles}

"""


@app.get("/webhook")
async def webhook_trigger():
    """
    A simple webhook endpoint that fetches recent articles and
    could trigger your LangGraph agent (simplified for demonstration)
    """
    markets: List[Market] = fetch_active_markets()
    articles: list[Article] = await get_recent_articles()

    match_market_instructions = MATCH_MARKETS_INSTRUCTIONS.format(
        markets=[market.question for market in markets],
        articles=[article.title for article in articles],
    )

    match_market_response = geminiflash.with_structured_output(
        ArticleMarketMatch
    ).invoke(match_market_instructions)

    print(match_market_response)

    # In a real implementation, you might do something like:
    # for article in articles:
    #     asyncio.create_task(send_to_langgraph(market, articles))

    return {"status": "success", "articles_found": len(articles)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
