import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get("SECRET_KEY")
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    # Logging configuration
    LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOGGING_FILE = "ai_prediction_market_trader.log"

    # API endpoints
    CLOB_ENDPOINT = os.environ.get("CLOB_ENDPOINT")
    GAMMA_ENDPOINT = os.environ.get("GAMMA_ENDPOINT")

    # API keys
    POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY")
    PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
    TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY")
    TIKTOK_API_KEY = os.environ.get("TIKTOK_API_KEY")

    # Market analysis settings
    MARKET_LIQUIDITY_THRESHOLD = int(
        os.environ.get("MARKET_LIQUIDITY_THRESHOLD", 10000)
    )
    MARKET_TIME_THRESHOLD = int(os.environ.get("MARKET_TIME_THRESHOLD", 24))
    MARKET_ANALYSIS_INTERVAL = int(os.environ.get("MARKET_ANALYSIS_INTERVAL", 300))
    MARKET_VOLUME_THRESHOLD = int(os.environ.get("MARKET_VOLUME_THRESHOLD", 10000))
    MARKET_LIMIT = int(os.environ.get("MARKET_LIMIT", 100))

    # Database configuration (if needed)
    DATABASE_URI = os.environ.get("DATABASE_URI")

    # Other configuration settings
    MAX_BET_SIZE = float(os.environ.get("MAX_BET_SIZE", 100.0))
    RISK_TOLERANCE = float(os.environ.get("RISK_TOLERANCE", 0.5))
