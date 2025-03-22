import sqlite3
import time
import uuid
from langgraph_sdk import get_client
from data_fetchers import fetch_active_markets
from langgraph_sdk.schema import Thread
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_openai import ChatOpenAI

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from models import (
    Market,
    GenerateAnalystsState,
    InterviewState,
    Perspectives,
    Recommendation,
    ResearchGraphState,
    SearchQuery,
    TraderState,
    OrderDetails,
)
from trade_tools import get_balances, trade_execution

### LLM

llm = ChatOpenAI(model="gpt-4o", temperature=0)


analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the prediction market details:

Market Question: {market.question}
Description: {market.description}
Current Odds: {market_odds}
End Date: {market.end_date}
Volume: {market.volume}
    
2. Determine the most interesting themes based on the market details and feedback above.
                    
3. Pick the top {max_analysts} themes.

4. Assign one analyst to each theme. Each analyst should focus on a different aspect that could affect the market outcome."""


def create_analysts(state: GenerateAnalystsState):
    """Create analysts"""

    market = state.market
    max_analysts = state.max_analysts

    # Format market odds
    market_odds = str(market)

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        market=market,
        market_odds=market_odds,
        max_analysts=max_analysts,
    )

    # Generate question
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of analysts.")]
    )

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


# Generate analyst question
question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


def generate_question(state: InterviewState):
    """Node to generate a question"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {"messages": [question]}


# Search query writing
search_instructions = SystemMessage(
    content="""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query"""
)


def search_web(state: InterviewState):
    """Retrieve docs from web search"""

    # Search
    tavily_search = TavilySearchResults(max_results=3)

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia(state: InterviewState):
    """Retrieve docs from wikipedia"""

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    # Search
    search_docs = WikipediaLoader(
        query=search_query.search_query, load_max_docs=2
    ).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


# Generate expert answer
answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""


def generate_answer(state: InterviewState):
    """Node to answer a question"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """Save interviews"""

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer"""

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_interview"
    return "ask_question"


# Write a summary (section of the final report) of the interview
section_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.

2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""


def write_section(state: InterviewState):
    """Node to write a section"""

    # Get state
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    # Append it to state
    return {"sections": [section.content]}


# Add nodes and edges
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges(
    "answer_question", route_messages, ["ask_question", "save_interview"]
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)


def initiate_all_interviews(state: ResearchGraphState):
    """Conditional edge to initiate all interviews via Send() API or return to create_analysts"""

    # Check if human feedback
    # Otherwise kick off interviews in parallel via Send() API
    market = state.market
    return [
        Send(
            "conduct_interview",
            {
                "analyst": analyst,
                "messages": [
                    HumanMessage(
                        content=f"So you said you were analyzing the market question: {market.question}?"
                    )
                ],
            },
        )
        for analyst in state.analysts
    ]


# Update the recommendation instructions
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

    # Full set of sections
    sections = state.sections
    market = state.market
    market_odds = str(market)

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Get recommendation using structured output
    system_message = recommendation_instructions.format(
        market=market, market_odds=market_odds, context=formatted_str_sections
    )
    recommendation = llm.with_structured_output(Recommendation).invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Create a recommendation based upon these memos.")]
    )

    # Return as a dict for state update
    return {"recommendation": recommendation}


def get_trader_instructions(
    market: Market, recommendation: Recommendation, balances: dict
):
    trader_instructions = f"""You are an expert crypto derivatives trader specializing in prediction markets. Your task is to execute a trade based on a thorough market analysis.

    MARKET DETAILS:
    Question: {market.question}
    Outcomes: {market.outcomes}
    Current Odds:
    - {market.outcomes[0]}: {market.outcome_prices[0]:.2%}
    - {market.outcomes[1]}: {market.outcome_prices[1]:.2%}

    RECOMMENDATION:
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

    market = state.market
    recommendation = state.recommendation
    balances = state.balances

    recommendation.outcome_index = int(recommendation.outcome_index)
    instructions = get_trader_instructions(market, recommendation, balances)

    llm_trader = llm.with_structured_output(OrderDetails)
    order_details = llm_trader.invoke(
        [SystemMessage(content=instructions)]
        + [
            HumanMessage(
                content="Read over the instructions and create an OrderDetails object based on the details provided."
            )
        ],
    )

    return {"order_details": order_details}


# Add nodes and edges
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_recommendation", write_recommendation)
builder.add_node("check_balances", get_balances)
builder.add_node("trade_configuration", trade_configuration)
builder.add_node("trade_execution", trade_execution)
# Logic
builder.add_edge(START, "create_analysts")
builder.add_conditional_edges(
    "create_analysts", initiate_all_interviews, ["conduct_interview"]
)
builder.add_edge("conduct_interview", "write_recommendation")
builder.add_edge("write_recommendation", "check_balances")
builder.add_edge("check_balances", "trade_configuration")
builder.add_edge("trade_configuration", "trade_execution")
builder.add_edge("trade_execution", END)

db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

memory = SqliteSaver(conn)
# Compile
graph = builder.compile(checkpointer=memory)


async def main():
    URL = "http://localhost:55147"
    client = get_client(url=URL)

    thread: Thread = await client.threads.create()

    market = fetch_active_markets()[0]  # Get first active market
    initial_state = GenerateAnalystsState(market=market, max_analysts=1, analysts=[])

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="research_agent",
        input=initial_state.model_dump(),
    )
    print(run)


def run_in_sdk(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    markets = fetch_active_markets()[:5]
    for market in markets:
        initial_state = GenerateAnalystsState(
            market=market, max_analysts=3, analysts=[]
        )
        output = graph.invoke(initial_state.model_dump(), config)
        print(output)
        time.sleep(100)


def observe_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config=config)
    print(state)


if __name__ == "__main__":
    # asyncio.run(main())
    thread_id = str(uuid.uuid4())
    run_in_sdk(thread_id)
    # observe_state(thread_id)
