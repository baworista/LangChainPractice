from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

_ = load_dotenv()


class AgentState(TypedDict):
    topic: str
    current_iteration: int
    max_iterations: int
    history: List[Dict[str, str]]


FIRST_AGENT_PROMPT = """
You are a scientist participating in a collaborative discussion about a specific topic.
Your role is to build on the other participant's scientific input by analyzing, hypothesizing, or proposing experimental or technical approaches related to the topic.
Keep the discussion focused on practical, theoretical, or experimental aspects, avoiding unnecessary philosophical or moral reflections. 
Respond in a clear, concise, and analytical manner to drive the conversation forward."
"""

SECOND_AGENT_PROMPT = """
You are a scientist contributing to a collaborative discussion about a particular topic.
Your role is to engage with the other participant's input by presenting scientific concepts, proposing methods, or exploring practical approaches to solving related challenges.
Stay focused on evidence-based reasoning, technical ideas, and actionable steps. 
Avoid broad moral or philosophical debates and ensure your response advances the scientific exploration of the topic.
"""


model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=300)


def truncate_history(history: List[Dict[str, str]], max_length: int = 10):
    """
    Truncates the conversation history to the last `max_length` exchanges.
    """
    return history[-max_length:]


def format_history(history: List[Dict[str, str]]) -> str:
    """
    Converts structured history into a readable string for the model.
    """
    return "\n\n".join([f"{entry['agent']}: {entry['message']}" for entry in history])


def first_agent(state: AgentState):
    history_str = format_history(state["history"])

    user_message = HumanMessage(
        content=f"Here is the topic: {state['topic']}\n\n"
                f"Here is a Conversation history:\n\n{history_str}"
    )

    messages = [
        SystemMessage(content=FIRST_AGENT_PROMPT),
        user_message,
    ]

    response = model.invoke(messages)
    new_message = response.content

    # Return the updated state, preserving all keys
    return {
        "topic": state["topic"],
        "current_iteration": state["current_iteration"],
        "max_iterations": state["max_iterations"],
        "history": truncate_history(state["history"] + [{"agent": "first_agent", "message": new_message}]),
    }


def second_agent(state: AgentState):
    history_str = format_history(state["history"])

    user_message = HumanMessage(
        content=f"Here is the topic: {state['topic']}\n\n"
                f"Here is a Conversation history:\n\n{history_str}"
    )

    messages = [
        SystemMessage(content=SECOND_AGENT_PROMPT),
        user_message,
    ]

    response = model.invoke(messages)
    new_message = response.content

    # Return the updated state, preserving all keys
    return {
        "topic": state["topic"],
        "current_iteration": state["current_iteration"] + 1,
        "max_iterations": state["max_iterations"],
        "history": truncate_history(state["history"] + [{"agent": "second_agent", "message": new_message}]),
    }



def should_continue(state: AgentState):
    """
    Checks if we can make one more iteration.
    """
    if state["current_iteration"] >= state["max_iterations"]:
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
    "topic": "Внешнеполитическая обстановка российской федерации",
    "current_iteration": 0,
    "max_iterations": 10,
    "history": []  # Initialize empty history
}

# response = graph.invoke(user_input, thread)
#
# print(response)

# Stream through the graph with the user-defined task
for state in graph.stream(user_input, thread):
    print("-" * 50)  # Separator for readability
    print("Current State (Raw):", state)  # Print the entire state for debugging

    # Extract the current node's state dynamically
    current_node_state = next(iter(state.values()))  # Get the first value from the dictionary

    # Safely access keys from the current node's state
    print("Processed State:")
    print(f"Topic: {current_node_state.get('topic', 'N/A')}")
    print(f"Current Iteration: {current_node_state.get('current_iteration', 'N/A')}")
    print(f"Max Iterations: {current_node_state.get('max_iterations', 'N/A')}")
    print("History:")
    for entry in current_node_state.get("history", []):
        print(f"{entry['agent']}: {entry['message']}")
    print("-" * 50)  # Separator for clarity


