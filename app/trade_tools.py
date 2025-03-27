from models import Balances, OrderResponse, TraderState
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderType, OrderArgs
import os
import logging
from py_clob_client.constants import POLYGON
from dotenv import load_dotenv
from web3 import Web3

logger = logging.getLogger(__name__)
load_dotenv()


class PolymarketTrader:
    def __init__(self):
        # Initialize CLOB client

        key = os.getenv("POLYMARKET_PRIVATE_KEY")
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


def _trade_execute(order_args: OrderArgs):
    trader = PolymarketTrader()
    if os.getenv("TRADE_EXECUTION").lower() == "true":
        signed_order = trader.client.create_order(order_args)
        resp = trader.client.post_order(signed_order, OrderType.GTC)
        return resp
    else:
        print("🚫 Trade execution disabled")
        return {
            "order_response": OrderResponse(
                status="success", response={"message": "Trade execution disabled"}
            )
        }


def trade_execution(state: TraderState):
    """Execute trades based on market analysis recommendation."""
    if state.recommendation.conviction >= 75:
        print(f"🚀 Executing trade for market: {state.market.question}")
        try:
            # Create order arguments
            order_args: OrderArgs = state.order_details.order_args
            print(
                f"  Order: {order_args.side} {order_args.size} units at {order_args.price}"
            )

            resp = _trade_execute(order_args)

            status = "success"
            if isinstance(resp, dict) and resp.get("status") == "failure":
                status = "failure"
            print(f"  Trade execution {status}")
            print("Final Recommendation:")
            print(state.recommendation)
            return {
                "order_response": OrderResponse(
                    status="success",
                    response=resp,
                )
            }
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            print(f"  ❌ Trade execution failed: {str(e)}")
            return {
                "order_response": OrderResponse(
                    status="failure", response={"failure": str(e)}
                )
            }
    else:
        print(
            f"🚫 Not executing trade due to low conviction - {state.recommendation.conviction}"
        )

        return {
            "order_response": OrderResponse(
                status="success", response={"message": "Low conviction"}
            )
        }


def get_balances(state: Balances):
    """Get the current USDC balance of the trader"""
    print("💵 Checking account balances")
    # Get RPC URL from environment variable
    rpc_url = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
    w3 = Web3(Web3.HTTPProvider(rpc_url))

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

    print(f"  USDC balance: ${balance_formatted:.2f}")

    return {"balances": {"USDC": balance_formatted}}
