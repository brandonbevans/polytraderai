from app.analysts import (
    generate_topic,
    generate_question,
    generate_answer,
    save_interview,
    route_messages,
    search_web,
    write_section,
)
from app.trader import (
    write_recommendation,
    trade_configuration,
    write_recommendation_from_news,
)

from langgraph.graph import END, START, StateGraph
from app.models import (
    InterviewState,
    ResearchGraphState,
    RecentNewsResearchMarketState,
)
from app.trade_tools import get_balances, trade_execution

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from open_deep_research.graph import graph as deep_research_graph


def get_interview_graph():
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)
    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", "save_interview"]
    )
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    return interview_builder.compile()


def get_full_graph():
    # Add nodes and edges
    builder = StateGraph(ResearchGraphState)
    builder.add_node("generate_topic", generate_topic)
    builder.add_node("deep_research", deep_research_graph)
    builder.add_node("write_recommendation", write_recommendation)
    builder.add_node("check_balances", get_balances)
    builder.add_node("trade_configuration", trade_configuration)
    builder.add_node("trade_execution", trade_execution)

    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_topic", "deep_research")
    builder.add_edge("deep_research", "write_recommendation")
    builder.add_edge("write_recommendation", "check_balances")
    builder.add_edge("check_balances", "trade_configuration")
    builder.add_edge("trade_configuration", "trade_execution")
    builder.add_edge("trade_execution", END)

    db_path = "state_db/example.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)

    memory = SqliteSaver(conn)
    # Compile
    graph = builder.compile(checkpointer=memory)
    return graph


def get_news_graph():
    builder = StateGraph(RecentNewsResearchMarketState)
    builder.add_node("write_recommendation_from_news", write_recommendation_from_news)
    builder.add_node("check_balances", get_balances)
    builder.add_node("trade_configuration", trade_configuration)
    builder.add_node("trade_execution", trade_execution)

    builder.add_edge(START, "write_recommendation_from_news")
    builder.add_edge("write_recommendation_from_news", "check_balances")
    builder.add_edge("check_balances", "trade_configuration")
    builder.add_edge("trade_configuration", "trade_execution")
    builder.add_edge("trade_execution", END)

    db_path = "state_db/example.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)

    memory = SqliteSaver(conn)
    # Compile
    graph = builder.compile(checkpointer=memory)
    return graph
