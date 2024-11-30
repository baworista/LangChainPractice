from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

_ = load_dotenv()

class AgentState(TypedDict):
    topic: str
    FirstAgentMessage: str
    SecondAgentMessage: str
    current_iteration: int
    max_iterations: int


FIRST_AGENT_PROMPT = """
You are an AI assistant which should speak with other assistant about provided topic.
"""

SECOND_AGENT_PROMPT = """
You are an AI assistant which should speak with other assistant about provided topic.
"""

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, max_tokens=500)

def first_agent(state: AgentState):
    user_message = HumanMessage(
        content=f"Here is the topic: {state['topic']}\n\nHere is your interlocutor message:\n\n{state['SecondAgentMessage']}")

    messages = [
        SystemMessage(content=FIRST_AGENT_PROMPT),
        user_message,
    ]

    response = model.invoke(messages)
    return {"FirstAgentMessage": response.content}


def second_agent(state: AgentState):
    user_message = HumanMessage(
        content=f"Here is the topic: {state['topic']}\n\nHere is your interlocutor message:\n\n{state['FirstAgentMessage']}")

    messages = [
        SystemMessage(content=SECOND_AGENT_PROMPT),
        user_message
    ]

    response = model.invoke(messages)
    return {
        "SecondAgentMessage": response.content,
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
    "FirstAgentMessage": str,
    "SecondAgentMessage": str,
    "current_iteration": 1,
    "max_iterations": 3,
}

# Stream through the graph with the user-defined task
for state in graph.stream(user_input, thread):
    print(state)
    print(f"Topic: {state.get('topic')}")


