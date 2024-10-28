from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import logging
from config import Config
from pytrends.request import TrendReq
from models import Market
import pytz
from pydantic import ValidationError
import json


def fetch_active_markets() -> List[Market]:
    market_analyzer_logger: logging.Logger = logging.getLogger("MarketAnalyzer")
    market_analyzer_logger.debug("Fetching active markets")

    # Calculate the minimum end date (e.g., 24 hours from now)
    min_end_date = datetime.now(pytz.UTC) + timedelta(
        hours=Config.MARKET_TIME_THRESHOLD
    )

    # Prepare query parameters
    params = {
        "limit": 50,  # Adjust this number as needed
        "offset": 0,
        "order": "volume",  # Sort by volume
        "ascending": False,  # Highest volume first
        "active": True,
        "closed": False,
        "liquidity_num_min": Config.MARKET_LIQUIDITY_THRESHOLD,
        "end_date_min": min_end_date.isoformat(),
        "volume_num_min": Config.MARKET_VOLUME_THRESHOLD,  # Minimum volume, adjust as needed
    }

    url: str = f"{Config.GAMMA_ENDPOINT}/markets"

    try:
        response: requests.Response = requests.get(url, params=params)
        response.raise_for_status()
        markets_data = response.json()
        market_analyzer_logger.debug(f"Received {len(markets_data)} markets from API.")

        markets = []
        for market_data in markets_data:
            try:
                # Pre-process the data
                for field in ["outcomes", "outcomePrices", "clobTokenIds"]:
                    if isinstance(market_data.get(field), str):
                        try:
                            market_data[field] = json.loads(market_data[field])
                        except json.JSONDecodeError:
                            market_data[field] = (
                                market_data[field]
                                .strip("[]")
                                .replace('"', "")
                                .split(",")
                            )

                # Set default values for potentially missing fields
                market_data.setdefault("fee", 0.0)
                market_data.setdefault("image", "")
                market_data.setdefault("icon", "")
                market_data.setdefault("description", "")

                market = Market(**market_data)

                # Check if both Yes and No odds are between 0.20 and 0.80
                yes_odds = market.outcome_prices[0] if market.outcome_prices else 0
                no_odds = (
                    market.outcome_prices[1] if len(market.outcome_prices) > 1 else 0
                )

                if 0.20 < yes_odds < 0.80 and 0.20 < no_odds < 0.80:
                    markets.append(market)
                else:
                    market_analyzer_logger.debug(
                        f"Skipping market {market.condition_id} due to odds outside range: Yes={yes_odds}, No={no_odds}"
                    )
            except ValidationError as ve:
                market_analyzer_logger.warning(f"Skipping invalid market: {ve}")
                market_analyzer_logger.debug(f"Invalid market data: {market_data}")

        markets = [market for market in markets if market.enable_order_book]
        market_analyzer_logger.info(f"Fetched {len(markets)} relevant markets")
        return markets
    except requests.RequestException as e:
        market_analyzer_logger.error(f"Error fetching active markets: {e}")
        return []
    except Exception as e:
        market_analyzer_logger.error(f"Unexpected error: {e}")
        return []


def fetch_market_data(condition_id: str) -> Optional[Dict[str, Any]]:
    url = f"{Config.GAMMA_ENDPOINT}/markets/{condition_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching market data: {e}")
        return None


def fetch_order_book(condition_id: str) -> Optional[Dict[str, Any]]:
    url = f"{Config.CLOB_ENDPOINT}/book?market={condition_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching order book: {e}")
        return None


def fetch_google_trends_data(market_topic: str) -> Optional[Dict[str, Any]]:
    pytrends = TrendReq(hl="en-US", tz=360)

    try:
        pytrends.build_payload([market_topic], timeframe="now 7-d")
        interest_over_time_df = pytrends.interest_over_time()

        if not interest_over_time_df.empty:
            avg_interest = interest_over_time_df[market_topic].mean()
            normalized_score = min(avg_interest / 100, 1.0)  # Normalize to 0-1 scale

            return {
                "score": normalized_score,
                "raw_data": interest_over_time_df.to_dict(),
            }
        else:
            logging.warning(f"No Google Trends data found for {market_topic}")
            return None
    except Exception as e:
        logging.error(f"Error fetching Google Trends data: {e}")
        return None
