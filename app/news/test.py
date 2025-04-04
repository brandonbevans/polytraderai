import asyncio
from app.news.main import get_relevant_articles
from app.models import ArticleMarketMatchFull


async def main():
    article_market_matches: list[ArticleMarketMatchFull] = await get_relevant_articles()
    for article_market_match in article_market_matches:
        print(article_market_match.market.question)
        print(article_market_match.articles)
        print("-" * 100)


if __name__ == "__main__":
    asyncio.run(main())
