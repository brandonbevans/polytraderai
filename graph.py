from analysts import (
    generate_question,
    generate_answer,
    save_interview,
    route_messages,
    search_web,
    write_section,
    start_interviews_or_create_better_analysts,
    search_web_for_themes,
)
from trader import (
    write_recommendation,
    trade_configuration,
)

from langgraph.graph import END, START, StateGraph
from models import (
    InterviewState,
    ResearchGraphState,
)
from trade_tools import get_balances, trade_execution

import sqlite3
from analysts import (
    create_analysts,
)

from langgraph.checkpoint.sqlite import SqliteSaver


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
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("search_web_for_themes", search_web_for_themes)
    builder.add_node("conduct_interview", get_interview_graph())
    builder.add_node("write_recommendation", write_recommendation)
    builder.add_node("check_balances", get_balances)
    builder.add_node("trade_configuration", trade_configuration)
    builder.add_node("trade_execution", trade_execution)

    builder.add_edge(START, "create_analysts")
    # builder.add_edge("search_web_for_themes", "conduct_interview")
    builder.add_conditional_edges(
        "create_analysts",
        start_interviews_or_create_better_analysts,
        ["conduct_interview", "search_web_for_themes"],
    )
    builder.add_edge("search_web_for_themes", "create_analysts")
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
    return graph
