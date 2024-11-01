import pprint
from langchain_core.tools import tool
from models import Balances, OrderDetails, OrderResponse
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
import os
import logging
from py_clob_client.constants import POLYGON, AMOY
from py_clob_client.clob_types import ApiCreds
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
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())

        else:
            # Mumbai testnet setup
            host = "https://clob.polymarket.com"
            chain_id = AMOY
            creds = ApiCreds(
                api_key=os.getenv("CLOB_API_KEY"),
                api_secret=os.getenv("CLOB_SECRET"),
                api_passphrase=os.getenv("CLOB_PASS_PHRASE"),
            )
            self.client = ClobClient(host, key=key, chain_id=chain_id, creds=creds)


@tool("trade_execution")
def trade_execution(order_details: OrderDetails):
    """Execute trades based on market analysis recommendation."""
    try:
        # Create order arguments
        order_args = OrderArgs(
            price=order_details.price,
            size=order_details.size,
            side=order_details.side,
            token_id=order_details.token_id,
            expiration=order_details.expiration,
        )

        trader = PolymarketTrader()
        signed_order = trader.client.create_order(order_args)
        resp = trader.client.post_order(signed_order, OrderType.GTC)
        pprint.pprint(resp)

        return {
            "order_response": OrderResponse(
                status="success",
                side=order_details.side,
                size=order_details.size,
                price=order_details.price,
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

    trader = PolymarketTrader()
    # wallet_address = trader.client.get_address()
    wallet_address = "0xdC8A2C33f07Ff317B6cA53C5b7318184B6Ac3009"
    print(wallet_address)
    # Create contract instance
    usdc_contract = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_ABI)

    # Get balance
    balance = usdc_contract.functions.balanceOf(wallet_address).call()
    # USDC has 6 decimals
    balance_formatted = balance / 1e6

    return {"balances": {"USDC": balance_formatted}}


if __name__ == "__main__":
    lol = trade_execution.invoke(
        input={
            "order_details": OrderDetails(
                price=0.5, size=1, side="BUY", token_id="1", expiration="0"
            )
        }
    )
    print(lol)
