from pydantic import BaseModel, Field, validator
from typing import List, Union
from datetime import datetime
import json
from langgraph.graph import MessagesState
import operator
from typing import Annotated
from py_clob_client.clob_types import OrderArgs


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


class Balances(BaseModel):
    balances: dict = Field(default={})


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


class Recommendation(BaseModel):
    outcome_index: int = Field(
        description="Index of the outcome to buy (0 or 1)", ge=0, le=1, default=0
    )
    conviction: int = Field(
        description="Conviction score for the recommendation, between 0 and 100",
        ge=0,
        le=100,
        default=0,
    )
    reasoning: str = Field(
        description="Detailed reasoning for the recommendation", default=""
    )

    class Config:
        arbitrary_types_allowed = True


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


class OrderResponse(BaseModel):
    status: str = Field(description="Status of the order", default="fail")
    response: dict = Field(description="Response from the order", default={})


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


class OrderDetails(BaseModel):
    order_args: OrderArgs


class TraderState(BaseModel):
    """State for trade execution"""

    market: Market
    recommendation: Recommendation
    balances: dict
    order_details: OrderDetails = Field(
        default=OrderDetails(
            order_args=OrderArgs(  # nosec
                price=0.0, size=0.0001, side="buy", token_id=""
            )
        )
    )
    order_response: OrderResponse = Field(default_factory=lambda: OrderResponse())

    class Config:
        arbitrary_types_allowed = True


class ResearchGraphState(BaseModel):
    """State for the overall research graph"""

    market: Market
    max_analysts: int
    analysts: List[Analyst] = Field(default_factory=list)
    sections: Annotated[List[str], operator.add] = Field(default_factory=list)
    recommendation: Recommendation = Field(
        default_factory=lambda: Recommendation(recommendation="", conviction=0)
    )
    order_response: OrderResponse = Field(default_factory=lambda: OrderResponse())
    balances: dict = Field(default={})
    performance: str = Field(default="")
    order_details: OrderDetails = Field(
        default=OrderDetails(
            order_args=OrderArgs(  # nosec
                price=0.0, size=0.0001, side="buy", token_id=""
            )
        )
    )

    class Config:
        arbitrary_types_allowed = True  # Allow Market type
