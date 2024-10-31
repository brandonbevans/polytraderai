import asyncio
import operator
from langgraph_sdk import get_client
from pydantic import BaseModel, Field
from typing import Annotated, List
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
from langgraph.graph import END, MessagesState, START, StateGraph

from models import Market

### LLM

llm = ChatOpenAI(model="gpt-4o", temperature=0)

### Schema


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


### Nodes and edges


def format_market_odds(market: Market) -> str:
    """Format market odds into a readable string"""
    return {
        outcome: price for outcome, price in zip(market.outcomes, market.outcome_prices)
    }


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
    market_odds = format_market_odds(market)

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
    interview = state["interview"]
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


# Write a report based on the interviews
report_writer_instructions = """You are an expert market analyst creating a comprehensive report on this prediction market:

Market Question: {market.question}
Description: {market.description}
Current Odds: {market_odds}
End Date: {market.end_date}
Volume: {market.volume}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific prediction market.
2. They write up their findings into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative that helps evaluate the market probabilities.
5. Based on all of the memos, provide a final probability assessment for the market outcomes. 
6. Most importantly, based on the provided odds and make a recommendation on whether to buy one of the outcomes, or do nothing.


To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Market Analysis
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat.
9. End with a clear probability assessment for the market outcomes.

Here are the memos from your analysts to build your report from: 

{context}"""

recommendation_instructions = """You are an expert market analyst creating a comprehensive report on this prediction market:

Market Question: {market.question}
Description: {market.description}
Current Odds: {market_odds}
End Date: {market.end_date}
Volume: {market.volume}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific prediction market.
2. They write up their findings into a memo.

Your task: 

1. You will be given a collection of memos from your analysts, along with the current odds for the prediction market.
2. Think carefully about the insights from each memo.
3. Based on the provided odds and your analysis of the memos, make a recommendation on whether to buy one of the outcomes, or do nothing. 
Also provide your conviction score for the recommendation, where the conviction score is defined as how confident you are that buying this outcome has relative edge over the odds.

Here are the memos from your analysts to build your report from:
{context}
"""


# def write_report(state: ResearchGraphState):
#     """Node to write the final report body"""

#     # Full set of sections
#     sections = state.sections
#     market = state.market
#     market_odds = format_market_odds(market)

#     # Concat all sections together
#     formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

#     # Summarize the sections into a final report
#     system_message = report_writer_instructions.format(
#         market=market, market_odds=market_odds, context=formatted_str_sections
#     )
#     report = llm.invoke(
#         [SystemMessage(content=system_message)]
#         + [HumanMessage(content="Write a report based upon these memos.")]
#     )
#     return {"content": report.content}


def write_recommendation(state: ResearchGraphState):
    """Node to write the recommendation"""

    # Full set of sections
    sections = state.sections
    market = state.market
    market_odds = format_market_odds(market)

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


# Write the introduction or conclusion
intro_conclusion_instructions = """You are a market analyst finishing a report on this prediction market:

Market Question: {market.question}
Description: {market.description}
Current Odds: {market_odds}
End Date: {market.end_date}
Volume: {market.volume}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header and include a final probability assessment.

Here are the sections to reflect on for writing: {formatted_str_sections}"""


def write_introduction(state: ResearchGraphState):
    """Node to write the introduction"""

    # Full set of sections
    sections = state.sections
    market = state.market
    market_odds = format_market_odds(market)

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Write introduction
    instructions = intro_conclusion_instructions.format(
        market=market,
        market_odds=market_odds,
        formatted_str_sections=formatted_str_sections,
    )
    intro = llm.invoke(
        [instructions] + [HumanMessage(content="Write the report introduction")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    """Node to write the conclusion"""

    # Full set of sections
    sections = state.sections
    market = state.market
    market_odds = format_market_odds(market)

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    # Write conclusion
    instructions = intro_conclusion_instructions.format(
        market=market,
        market_odds=market_odds,
        formatted_str_sections=formatted_str_sections,
    )
    conclusion = llm.invoke(
        [instructions] + [HumanMessage(content="Write the report conclusion")]
    )
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion"""

    # Save full final report
    content = state.content
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = (
        state.introduction + "\n\n---\n\n" + content + "\n\n---\n\n" + state.conclusion
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


# Add nodes and edges
builder = StateGraph(ResearchGraphState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("conduct_interview", interview_builder.compile())
# builder.add_node("write_report", write_report)
# builder.add_node("write_introduction", write_introduction)
# builder.add_node("write_conclusion", write_conclusion)
builder.add_node("write_recommendation", write_recommendation)
# builder.add_node("finalize_report", finalize_report)

# Logic
builder.add_edge(START, "create_analysts")
builder.add_conditional_edges(
    "create_analysts", initiate_all_interviews, ["conduct_interview"]
)
# builder.add_edge("conduct_interview", "write_report")
# builder.add_edge("conduct_interview", "write_introduction")
# builder.add_edge("conduct_interview", "write_conclusion")
# builder.add_edge(
#     ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
# )
builder.add_edge("conduct_interview", "write_recommendation")
builder.add_edge("write_recommendation", END)

# Compile
graph = builder.compile()


async def main():
    URL = "http://localhost:49472"
    client = get_client(url=URL)

    thread: Thread = await client.threads.create()

    market = fetch_active_markets()[0]  # Get first active market
    initial_state = GenerateAnalystsState(market=market, max_analysts=3, analysts=[])

    run = await client.runs.create(
        thread_id=thread["thread_id"],
        assistant_id="research_agent2",
        input=initial_state.model_dump(),  # Use model_dump() for serialization
    )
    print(run)


if __name__ == "__main__":
    asyncio.run(main())
