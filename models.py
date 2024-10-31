from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
from datetime import datetime
import json
from langgraph.graph import MessagesState
import operator
from typing import Annotated


class Event(BaseModel):
    id: str
    ticker: str


class Market(BaseModel):
    id: str
    question: str
    condition_id: str = Field(alias="conditionId")
    slug: str
    end_date: datetime = Field(alias="endDate")
    start_date: datetime = Field(alias="startDate")
    fee: float
    image: str
    icon: str
    description: str
    outcomes: List[str]
    outcome_prices: List[float] = Field(alias="outcomePrices")
    volume: Union[float, str]
    active: bool
    closed: bool
    market_maker_address: str = Field(alias="marketMakerAddress")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    new: bool
    archived: bool
    restricted: bool
    question_id: str = Field(alias="questionID")
    enable_order_book: bool = Field(alias="enableOrderBook")
    order_price_min_tick_size: float = Field(alias="orderPriceMinTickSize")
    order_min_size: float = Field(alias="orderMinSize")
    volume_num: float = Field(alias="volumeNum")
    end_date_iso: str = Field(alias="endDateIso")
    start_date_iso: str = Field(alias="startDateIso")
    has_reviewed_dates: bool = Field(alias="hasReviewedDates")
    clob_token_ids: List[str] = Field(alias="clobTokenIds")
    accepting_orders: bool = Field(alias="acceptingOrders")
    # comment_count: int = Field(alias="commentCount")
    _sync: bool
    ready: bool
    funded: bool
    cyom: bool
    pager_duty_notification_enabled: bool = Field(alias="pagerDutyNotificationEnabled")
    approved: bool
    rewards_min_size: float = Field(alias="rewardsMinSize")
    rewards_max_spread: float = Field(alias="rewardsMaxSpread")
    spread: float
    last_trade_price: float = Field(alias="lastTradePrice")
    best_ask: float = Field(alias="bestAsk")
    automatically_active: bool = Field(alias="automaticallyActive")
    clear_book_on_start: bool = Field(alias="clearBookOnStart")

    @validator("outcomes", "outcome_prices", "clob_token_ids", pre=True)
    def parse_string_to_list(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v.strip("[]").replace('"', "").split(",")
        return v

    @validator("outcome_prices", pre=True)
    def convert_to_float(cls, v):
        if isinstance(v, list):
            return [float(price) for price in v]
        return v

    class Config:
        populate_by_name = True
        extra = "ignore"


class MarketList(BaseModel):
    markets: List[Market]


class Prediction(BaseModel):
    market: Market


class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(description="Name of the analyst.")
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class Recommendation(BaseModel):
    recommendation: str = Field(
        description="Recommendation on whether to buy one of the outcomes, or do nothing.",
        default="",
    )
    conviction: int = Field(
        description="Conviction score for the recommendation, between 0 and 100.",
        default=0,
        ge=0,
        le=100,
    )


class TraderState(BaseModel):
    recommendation: Recommendation
    market: Market


class GenerateAnalystsState(BaseModel):
    """State for generating analysts"""

    market: Market
    max_analysts: int
    analysts: List[Analyst] = Field(default_factory=list)  # Default empty list

    class Config:
        arbitrary_types_allowed = True  # Allow Market type


class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class ResearchGraphState(BaseModel):
    """State for the overall research graph"""

    market: Market
    max_analysts: int
    analysts: List[Analyst] = Field(default_factory=list)
    sections: Annotated[List[str], operator.add] = Field(default_factory=list)
    introduction: str = ""
    content: str = ""
    conclusion: str = ""
    final_report: str = ""
    recommendation: Recommendation = Field(
        default_factory=lambda: Recommendation(recommendation="", conviction=0)
    )

    class Config:
        arbitrary_types_allowed = True  # Allow Market type


class OrderDetails(BaseModel):
    """Model for order details required by Polymarket CLOB"""

    token_id: str = Field(description="The token ID for the outcome being traded")
    price: float = Field(
        description="The price at which to execute the trade", gt=0, le=1
    )
    size: float = Field(description="The size of the order in USD", gt=0)
    side: str = Field(description="The side of the trade (BUY or SELL)")
    expiration: str = Field(
        description="Order expiration timestamp in Unix milliseconds",
        default="100000000000",  # Default to far future
    )
    order_type: str = Field(
        description="Type of order (GTC, GTD, or IOC)", default="GTC"
    )
    salt: str = Field(description="Unique identifier for the order", default="")
    maker: str = Field(description="Address of the order maker", default="")

    @validator("side")
    def validate_side(cls, v):
        if v.upper() not in ["BUY", "SELL"]:
            raise ValueError("side must be either BUY or SELL")
        return v.upper()

    @validator("order_type")
    def validate_order_type(cls, v):
        if v.upper() not in ["GTC", "GTD", "IOC"]:
            raise ValueError("order_type must be GTC, GTD, or IOC")
        return v.upper()


class TradeState(BaseModel):
    """Complete state for trade execution"""

    market: Market
    recommendation: Recommendation
    order_details: Optional[OrderDetails] = None
    balance: float = Field(description="Available balance for trading", default=0.0)
    max_position_size: float = Field(
        description="Maximum allowed position size in USD", default=100.0
    )
    min_conviction: int = Field(
        description="Minimum conviction required to execute trade",
        default=40,
        ge=0,
        le=100,
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("order_details", pre=True, always=True)
    def set_order_details(cls, v, values):
        """Automatically generate order details from market and recommendation if not provided"""
        if v is None and "market" in values and "recommendation" in values:
            market = values["market"]
            recommendation = values["recommendation"]

            # Parse recommendation
            rec_lower = recommendation.recommendation.lower()
            if "buy yes" in rec_lower:
                side, outcome_index = "BUY", 0
            elif "buy no" in rec_lower:
                side, outcome_index = "BUY", 1
            elif "sell yes" in rec_lower:
                side, outcome_index = "SELL", 0
            elif "sell no" in rec_lower:
                side, outcome_index = "SELL", 1
            else:
                raise ValueError("Invalid recommendation format")

            # Calculate size based on conviction
            size = min(
                100.0 * (recommendation.conviction / 100),
                values.get("max_position_size", 100.0),
            )

            return OrderDetails(
                token_id=market.clob_token_ids[outcome_index],
                price=market.outcome_prices[outcome_index],
                size=size,
                side=side,
            )
        return v
