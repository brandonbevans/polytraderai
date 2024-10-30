from pydantic import BaseModel, Field, validator
from typing import List, Union
from datetime import datetime
import json


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
