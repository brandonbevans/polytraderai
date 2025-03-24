from models import (
    Perspectives,
    GenerateAnalystsState,
    InterviewState,
    SearchQuery,
    ResearchGraphState,
    AnalystThemes,
)
from llms import gpt4o
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langgraph.constants import Send

from langchain_community.tools.tavily_search import TavilySearchResults


create_themes_instructions = """Given the following web search results, create a list of themes that are relevant to the market. 
The themes will be used to create an 'analyst' persona, who will then research the market based on the theme. 
So create the theme in such a way that when the analyst does the research about the theme, they will be researching the most important aspects of the market. 
Put another way, the theme should be not just something of interest, but the subject matter that has the most impact on the market question itself.

Market: {market}

Search results:
{search_docs}

1. Analyze the search results and identify the most relevant themes.
2. Create a list of themes that are relevant to the market.
3. Return the list of themes in a list of strings."""


def search_web_for_themes(state: GenerateAnalystsState):
    """Search web for relevant themes"""
    print("🔍 Searching web for relevant themes")

    tavily_search = TavilySearchResults(max_results=10)
    structured_llm = gpt4o.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        "Create the best search query to find the most relevant themes to answer the Market Question for the following market: "
        + str(state.market)
        + " Prioritize the most important themes and the most relevant sources. Keep the query to no more than 300 characters."
    )
    search_docs = tavily_search.invoke(search_query.search_query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    theme_llm = gpt4o.with_structured_output(AnalystThemes)
    analyst_themes = theme_llm.invoke(
        create_themes_instructions.format(
            market=str(state.market), search_docs=formatted_search_docs
        )
    )

    return {"analyst_themes": analyst_themes}


analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the prediction market details:
{market}
    
2. Look at the existing themes here:
{themes}

3. Look at the existing analysts here:
{analysts}

4. If there are themes that don't have a matching analyst, create a new analyst for that theme."""


def create_analysts(state: GenerateAnalystsState):
    """Create analysts"""
    print(f"⚙️ Creating analysts for market: {state.market.question}")
    market = state.market
    max_analysts = state.max_analysts
    if len(state.analyst_themes.themes) == 0:
        return {"analysts": []}
    analyst_themes = sorted(
        state.analyst_themes.themes, key=lambda x: x.confidence, reverse=True
    )
    analyst_themes = analyst_themes[:max_analysts]
    ### Reformat all of this, look into the themes object in state, if there's stuff there - then we can generate some analysts
    ### If not just return and it will auotmatically trigger the search_web_for_themes node

    # Enforce structured output
    structured_llm = gpt4o.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        market=str(market),
        analysts=str(state.analysts),
        themes=str(analyst_themes),
    )

    # Generate question
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of analysts.")]
    )

    print(f"✅ Created {len(analysts.analysts)} analysts")
    for analyst in analysts.analysts:
        print(f"  - {analyst}")

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
    print(f"❓ Generating question for analyst: {state['analyst']}")

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = gpt4o.invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {"messages": [question]}


answer_instructions = """You are an expert being interviewed by an analyst.

Here is the analyst area of focus: {goals}. 
        
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
    print(f"💬 Generating expert answer for {state['analyst']}")

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = gpt4o.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    print(f"  Expert answered with {len(answer.content)} characters")

    # Append it to state
    return {"messages": [answer]}


def save_interview(state: InterviewState):
    """Save interviews"""
    print(f"💾 Saving interview with {state['analyst']}")

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    print(f"  Interview contains {len(messages)} messages")

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
    print(f"🔍 Searching web for {state['analyst']}")

    # Search
    tavily_search = TavilySearchResults(max_results=6)

    # Search query
    structured_llm = gpt4o.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    print(f"  Query: {search_query.search_query}")

    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    print(f"  Found {len(search_docs)} documents")

    return {"context": [formatted_search_docs]}


def start_interviews_or_create_better_analysts(state: ResearchGraphState):
    """Conditional edge to initiate all interviews via Send() API or keep searching the web for better themes"""
    analyst_themes = state.analyst_themes
    market = state.market
    if len(analyst_themes.themes) == 0:
        return "search_web_for_themes"
    for analyst in state.analysts:
        print(f"  - Interview scheduled for {analyst}")

    print(f"🔄 Initiating interviews with {len(state.analysts)} analysts")
    return [
        Send(
            "conduct_interview",
            {
                "analyst": analyst,
                "messages": [
                    HumanMessage(
                        content=f"So you said you were analyzing the following question: {market.question}?"
                    )
                ],
            },
        )
        for analyst in state.analysts
    ]


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
    print(f"📝 Writing report section for {state['analyst']}")

    # Get state
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = gpt4o.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    print(f"  Created report section with {len(section.content)} characters")

    # Append it to state
    return {"sections": [section.content]}
