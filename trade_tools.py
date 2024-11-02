from models import Balances, OrderResponse, TraderState
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderType, MarketOrderArgs
import os
import logging
from py_clob_client.constants import POLYGON, AMOY
from dotenv import load_dotenv
from web3 import Web3

logger = logging.getLogger(__name__)
load_dotenv()


class PolymarketTrader:
    def __init__(self):
        # Initialize CLOB client

        # For Mumbai testnet
        is_production = (
            True  # os.getenv("ENVIRONMENT", "development").lower() == "production"
        )
        key = os.getenv("POLYMARKET_PRIVATE_KEY")

        if is_production:
            host = "https://clob.polymarket.com"
            chain_id = POLYGON
            self.client = ClobClient(
                host=host,
                key=key,
                chain_id=chain_id,
                signature_type=1,
                funder=os.getenv("POLYMARKET_PROXY_ADDRESS"),
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())

        else:
            # Mumbai testnet setup
            host = "https://clob.polymarket.com"
            chain_id = AMOY
            self.client = ClobClient(
                host=host,
                key=key,
                chain_id=chain_id,
                signature_type=1,
                funder=os.getenv("POLYMARKET_PROXY_ADDRESS"),
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())


def _trade_execute(order_args: MarketOrderArgs):
    trader = PolymarketTrader()

    signed_order = trader.client.create_order(order_args)
    resp = trader.client.post_order(signed_order, OrderType.GTC)
    return resp


def trade_execution(state: TraderState):
    """Execute trades based on market analysis recommendation."""
    try:
        # Create order arguments
        order_args = state.order_details.order_args

        resp = _trade_execute(order_args)

        return {
            "order_response": OrderResponse(
                status="success",
                response=resp,
            )
        }
    except Exception as e:
        logger.error(f"Trade execution failed: {str(e)}")
        return {
            "order_response": OrderResponse(
                status="failure", response={"failure": str(e)}
            )
        }


def get_balances(state: Balances):
    """Get the current USDC balance of the trader"""
    # Polygon RPC URL - you may want to use an environment variable for this
    w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))

    # USDC contract address on Polygon
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

    # USDC ABI - we only need the balanceOf function
    USDC_ABI = [
        {
            "inputs": [{"name": "account", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]

    # wallet_address = trader.client.get_address()
    wallet_address = os.getenv("POLYMARKET_PROXY_ADDRESS")
    # Create contract instance
    usdc_contract = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_ABI)

    # Get balance
    balance = usdc_contract.functions.balanceOf(wallet_address).call()
    # USDC has 6 decimals
    balance_formatted = balance / 1e6

    return {"balances": {"USDC": balance_formatted}}
