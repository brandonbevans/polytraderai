from models import (
    ResearchGraphState,
    Market,
    Recommendation,
    TraderState,
    OrderDetails,
)
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from llms import claude37, claude37thinking


def get_trader_instructions(
    market: Market, recommendation: Recommendation, balances: dict
):
    trader_instructions = f"""You are an expert crypto derivatives trader specializing in prediction markets. Your task is to execute a trade based on a thorough market analysis.


    Market:
    {str(market)}

    Recommendation:
    Selected Outcome: {market.outcomes[recommendation.outcome_index]}
    Conviction: {recommendation.conviction}/100
    Reasoning: {recommendation.reasoning}

    ACCOUNT STATUS:
    Available USDC Balance: {balances["USDC"]:.2f}

    Your task is to create an order that:
    1. POSITION SIZING:
    - Base Position Size on both conviction and available USDC balance
    - For high conviction (>80): Use up to 25% of available USDC balance
    - For medium conviction (60-80): Use up to 15% of available USDC balance
    - For lower conviction (<60): No trade
    - Never exceed $1000 per trade regardless of conviction
    - Remember: Total USDC spent = size * price
    - Position Size needs to be at least $5

    2. TOKEN SELECTION:
    - Use token_id: {market.clob_token_ids[recommendation.outcome_index]}
    - This corresponds to the recommended outcome: {market.outcomes[recommendation.outcome_index]}

    3. PRICE EXECUTION:
    - Current market price is: {market.outcome_prices[recommendation.outcome_index]:.4f}
    - Set limit price 0.5% above current highest bid to ensure execution while minimizing slippage
    - This places us at top of order book but better than market order price
    - Example: If current price is 0.4000, set limit at ~0.4020


    RISK MANAGEMENT RULES:
    - Never risk more than 25% of available balance on a single trade
    - Ensure minimum order size is respected (minimum $5)
    - Account for price impact on larger orders
    - Leave some balance for future opportunities

    Output a OrderDetails object with these exact fields:
    - order_args: OrderArgs object with the correct token ID and position size
    OrderArgs:
        - price: The current market price for the outcome
        - size: The amount of shares to buy at the above price
        - side: "BUY"
        - token_id: The correct token ID for the chosen outcome
    
    Note: The product of size and price MUST be at least $1
    """
    return trader_instructions


def trade_configuration(state: TraderState):
    """Node to create the order details"""
    print(f"💰 Configuring trade for market: {state.market.question}")

    market = state.market
    recommendation = state.recommendation
    balances = state.balances

    recommendation.outcome_index = int(recommendation.outcome_index)
    instructions = get_trader_instructions(market, recommendation, balances)

    llm_trader = claude37.with_structured_output(OrderDetails)
    order_details = llm_trader.invoke(
        [SystemMessage(content=instructions)]
        + [
            HumanMessage(
                content="Read over the instructions and create an OrderDetails object based on the details provided."
            )
        ],
    )

    print(f"  Order details: {order_details}")

    return {"order_details": order_details}


recommendation_instructions = """You are a market analyst creating a recommendation for a prediction market.

MARKET DETAILS:
Question: {market.question}
Description: {market.description}
Outcomes: {market.outcomes}
Current Odds: 
- {market.outcomes[0]}: {market.outcome_prices[0]:.2%}
- {market.outcomes[1]}: {market.outcome_prices[1]:.2%}
End Date: {market.end_date}
Volume: {market.volume}

Your task is to analyze the research provided and make a recommendation on which outcome to BUY.
Note: You can only BUY one of the two outcomes - you cannot SELL as we don't have any existing positions.

Based on the provided memos from your analysts:
{context}

Create a recommendation that includes:
1. Which outcome to BUY (must be one of: {market.outcomes})
2. Your conviction level (0-100) in this recommendation
3. Detailed reasoning explaining:
   - Why you chose this outcome
   - Why the current market odds are incorrect
   - Key evidence supporting your view
   - Potential risks to your thesis

Remember:
- You can only recommend BUYING one of the two outcomes
- Higher conviction (>70) should only be used when evidence strongly suggests market odds are wrong
- Lower conviction (<30) suggests staying out of the market
- Consider time until market resolution and potential catalysts

Output your recommendation as a structured object with:
- outcome_index: 0 for {market.outcomes[0]}, 1 for {market.outcomes[1]}
- conviction: 0-100 score
- reasoning: Detailed explanation of your recommendation
"""


def write_recommendation(state: ResearchGraphState):
    """Node to write the recommendation"""
    print(f"📊 Writing trade recommendation for market: {state.market.question}")

    # Full set of sections
    final_report = state.final_report
    market = state.market

    system_message = recommendation_instructions.format(
        market=market,
        context=final_report,
    )
    recommendation = claude37thinking.with_structured_output(Recommendation).invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Create a recommendation based upon these memos.")]
    )

    print(f"  Recommendation: {recommendation}")

    # Return as a dict for state update
    return {"recommendation": recommendation}
