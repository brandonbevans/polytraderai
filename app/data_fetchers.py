from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import logging
from app.config import Config
from pytrends.request import TrendReq
from app.models import Market
import pytz
from pydantic import ValidationError
import json
import os
import dotenv

dotenv.load_dotenv()

# Add a constant for timeout duration
REQUESTS_TIMEOUT = 30  # seconds


def fetch_user_positions() -> set[str]:
    """Fetch all markets where the user has an existing position.

    Args:
        wallet_id: Ethereum wallet address

    Returns:
        set: Set of condition_ids where user has positions
    """
    wallet_id = os.environ.get("POLYMARKET_PROXY_ADDRESS")
    url = f"https://data-api.polymarket.com/positions?sizeThreshold=.1&user={wallet_id}"

    try:
        response = requests.get(url, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
        positions_data = response.json()

        # Extract condition_ids from positions
        markets_with_positions = {
            position["conditionId"]
            for position in positions_data
            if float(position.get("size", 0)) > 0.1  # Additional size check
        }

        return markets_with_positions
    except Exception as e:
        print(f"Error fetching user positions:  {e}")
        return set()


def fetch_markets_with_positions() -> List[Market]:
    """Fetch all markets where the user has an existing position."""
    market_condition_ids_with_positions: set[str] = fetch_user_positions()
    # Prepare query parameters
    condition_ids_str = "&".join(
        f"condition_ids={condition_id}"
        for condition_id in market_condition_ids_with_positions
    )

    url: str = f"{Config.GAMMA_ENDPOINT}/markets?{condition_ids_str}"

    try:
        response: requests.Response = requests.get(url, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
        markets_data = response.json()
        return [
            format_market_response_to_market(market_data)
            for market_data in markets_data
        ]
    except Exception as e:
        print(f"Error fetching markets with positions: {e}")
        return []


def format_market_response_to_market(market_data: Dict[str, Any]) -> Market:
    # Pre-process the data
    for field in ["outcomes", "outcomePrices", "clobTokenIds"]:
        if isinstance(market_data.get(field), str):
            try:
                market_data[field] = json.loads(market_data[field])
            except json.JSONDecodeError:
                market_data[field] = (
                    market_data[field].strip("[]").replace('"', "").split(",")
                )

    # Set default values for potentially missing fields
    market_data.setdefault("fee", 0.0)
    market_data.setdefault("image", "")
    market_data.setdefault("icon", "")
    market_data.setdefault("description", "")

    return Market(**market_data)


def fetch_active_markets() -> List[Market]:
    """Fetch active markets, excluding those where user has positions if wallet_id provided."""
    market_analyzer_logger: logging.Logger = logging.getLogger("MarketAnalyzer")
    market_analyzer_logger.debug("Fetching active markets")

    # Get markets with existing positions if wallet_id provided
    markets_to_exclude = fetch_user_positions()
    market_analyzer_logger.debug(
        f"Excluding {len(markets_to_exclude)} markets with existing positions"
    )

    # Calculate the minimum end date (e.g., 24 hours from now)
    min_end_date = datetime.now(pytz.UTC) + timedelta(
        hours=Config.MARKET_TIME_THRESHOLD
    )

    # Prepare query parameters
    params = {
        "limit": 300,  # Adjust this number as needed
        "offset": 0,
        "order": "volume",  # Sort by volume
        "ascending": False,  # Highest volume first
        "active": True,
        "closed": False,
        "liquidity_num_min": Config.MARKET_LIQUIDITY_THRESHOLD,
        "end_date_min": min_end_date.isoformat(),
        "end_date_max": (min_end_date + timedelta(days=90)).isoformat(),
        "volume_num_min": Config.MARKET_VOLUME_THRESHOLD,  # Minimum volume, adjust as needed
    }

    url: str = f"{Config.GAMMA_ENDPOINT}/markets"

    try:
        response: requests.Response = requests.get(
            url, params=params, timeout=REQUESTS_TIMEOUT
        )
        response.raise_for_status()
        markets_data = response.json()
        market_analyzer_logger.debug(f"Received {len(markets_data)} markets from API.")

        # Add position filtering
        filtered_markets = []
        for market_data in markets_data:
            try:
                # Skip markets where we have positions
                if market_data["conditionId"] in markets_to_exclude:
                    market_analyzer_logger.debug(
                        f"Skipping market {market_data['conditionId']} due to existing position"
                    )
                    continue

                market = format_market_response_to_market(market_data)

                # Check if both Yes and No odds are between 0.20 and 0.80
                yes_odds = market.outcome_prices[0] if market.outcome_prices else 0
                no_odds = (
                    market.outcome_prices[1] if len(market.outcome_prices) > 1 else 0
                )

                if 0.10 < yes_odds < 0.90 and 0.10 < no_odds < 0.90:
                    filtered_markets.append(market)
                else:
                    market_analyzer_logger.debug(
                        f"Skipping market {market.condition_id} due to odds outside range: Yes={yes_odds}, No={no_odds}"
                    )

            except ValidationError as ve:
                market_analyzer_logger.warning(f"Skipping invalid market: {ve}")
                market_analyzer_logger.debug(f"Invalid market data: {market_data}")

        # Final filtering for order book enabled markets
        markets = [market for market in filtered_markets if market.enable_order_book]

        market_analyzer_logger.info(
            f"Fetched {len(markets)} relevant markets "
            f"(excluded {len(markets_to_exclude)} with positions)"
        )
        return markets

    except requests.RequestException as e:
        market_analyzer_logger.error(f"Error fetching active markets: {e}")
        return []
    except Exception as e:
        market_analyzer_logger.error(f"Unexpected error: {e}")
        return []


def fetch_order_book(condition_id: str) -> Optional[Dict[str, Any]]:
    url = f"{Config.CLOB_ENDPOINT}/book?market={condition_id}"
    try:
        response = requests.get(url, timeout=REQUESTS_TIMEOUT)
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


if __name__ == "__main__":
    print(fetch_active_markets())
