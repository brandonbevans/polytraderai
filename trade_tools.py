from langchain_core.tools import tool
from models import OrderDetails
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
import os
import logging

logger = logging.getLogger(__name__)


class PolymarketTrader:
    def __init__(self):
        # Initialize CLOB client
        host = "https://clob.polymarket.com"
        key = os.getenv("POLYMARKET_PRIVATE_KEY")
        chain_id = 137  # Polygon mainnet

        self.client = ClobClient(host=host, key=key, chain_id=chain_id)


@tool("trade_execution")
def trade_execution(order_details: OrderDetails) -> str:
    """Execute trades based on market analysis recommendation."""
    try:
        # Order details are automatically generated
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

        return (
            f"Order placed successfully:\n"
            f"Action: {order_details.side}\n"
            f"Size: {order_details.size:.2f}\n"
            f"Price: ${order_details.price:.4f}\n"
            f"Order Type: {order_details.order_type}\n"
            f"Response: {resp}"
        )
    except Exception as e:
        logger.error(f"Trade execution failed: {str(e)}")
        return f"Trade execution failed: {str(e)}"
