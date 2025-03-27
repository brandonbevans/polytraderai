import asyncio
from app.news.main import webhook_trigger


async def main():
    article_market_match = await webhook_trigger()
    print(article_market_match)


if __name__ == "__main__":
    asyncio.run(main())
