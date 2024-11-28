from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

_ = load_dotenv()

class AgentState(TypedDict):
    topic: str
    FirstAgentOpinion: str
    SecondAgentOpinion: str
    summary: str
    current_iteration: int
    max_iterations: int


OPINION_PROMPT = """
You are an AI assistant which should speak with other assistant about provided topic.
"""


def first_agent(state: AgentState):
    summary = state["summary"]

    summary_text = f"\n\nSummary of previous conversation:\n{summary}" if summary else ""

    user_message = HumanMessage(
        content=f"Here is the topic:\n\n{state['topic']}\n\nHere is your interlocutor word:\n\n{state['SecondAgentOpinion']}\n\nHere is summary of previous messages:\n\n{summary_text}")

    messages = [
        SystemMessage(content=OPINION_PROMPT),
        user_message,
    ]

    response = ChatOpenAI(model="gpt-4o", temperature=0.6).invoke(messages)
    return {"FirstAgentOpinion": response.content}


def second_agent(state: AgentState):
    summary = state["summary"]

    summary_text = f"\n\nSummary of previous conversation:\n{summary}" if summary else ""

    user_message = HumanMessage(
        content=f"Here is the topic:\n\n{state['topic']}\n\nHere is your interlocutor word:\n\n{state['FirstAgentOpinion']}\n\nHere is summary of previous messages:\n\n{summary_text}")

    messages = [
        SystemMessage(content=OPINION_PROMPT),
        user_message
    ]

    response = ChatOpenAI(model="gpt-4o", temperature=0.6).invoke(messages)
    return {
        "SecondAgentOpinion": response.content,
        "current_iteration": state.get("current_iteration", 1) + 1
    }


def should_continue(state: AgentState):
    """
    Checks if we can make one more iteration.
    """
    if state["current_iteration"] > state["max_iterations"]:
        return END
    return "first_agent"


builder = StateGraph(AgentState)

builder.add_node("first_agent", first_agent)
builder.add_node("second_agent", second_agent)

builder.set_entry_point("first_agent")
builder.add_edge("first_agent", "second_agent")


builder.add_conditional_edges(
    "second_agent",
    should_continue,
    {END: END, "first_agent": "first_agent"}
)

graph = builder.compile()


# Save as PNG
graph_image = graph.get_graph().draw_mermaid_png()
with open("graph_diagram.png", "wb") as file:
    file.write(graph_image)
print("Saved as PNG 'graph_diagram.png'")


# Thread configuration and graph input
thread = {"configurable": {"thread_id": "1"}}

user_input = {
    "topic": "Miami",
    "FirstAgentOpinion": str,
    "SecondAgentOpinion": str,
    "current_iteration": 1,
    "max_iterations": 3,
}

# Stream through the graph with the user-defined task
for state in graph.stream(user_input, thread):
    print(state)


