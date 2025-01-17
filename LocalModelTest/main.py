from langchain_core.messages import SystemMessage
from langchain_ollama import OllamaLLM
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph

llm = OllamaLLM(model="phi4")

analyst_prompt = """
You are Alise, a highly skilled analyst specializing in {topic}.
Your task is to provide a thorough analysis of the topic using the provided information. 
You should:
1. Identify key issues, trends, or findings in the provided data.
2. Discuss any patterns, anomalies, or notable observations.
3. Formulate potential implications or conclusions based on your analysis.
4. Recommend next steps for further investigation or action.

If there are previous reports or recommendations from reviewers, incorporate them into your analysis, critically considering their value or any contradictions.

Be clear and objective, but also make sure to propose fresh insights that could drive deeper understanding or exploration of the topic.
"""


reviewer_prompt = """
You are Mark, a seasoned reviewer with expertise in {topic}.
Your role is to critically evaluate the analysis provided by Alise.
You should:
1. Assess the strengths and weaknesses of the analysis in terms of accuracy, depth, and clarity.
2. Identify any missing or overlooked aspects in the analysis.
3. Suggest areas where further exploration or clarification is needed.
4. Highlight potential biases or assumptions that may affect the conclusions.

Your feedback should be constructive but rigorous, helping to improve the analysis and drive towards more comprehensive insights. Ensure your review challenges assumptions and encourages deeper exploration where necessary.
"""


class CustomState(MessagesState):
    topic: str


def analyst_node(state):
    topic = state["topic"]

    print(f"Analyst Alise activated.")

    system_prompt = analyst_prompt.format(topic=topic)

    messages = state["messages"]

    if len(messages) != 0:
        last_message = "Your previous report: \n" + messages[-2].content + "\n\n\nReviewers recommendations: \n" + messages[-1].content
    else:
        last_message = "This is the beginning of conversation. Make your initial analysis based on the questionnaire results."

    llm_messages = [SystemMessage(content=system_prompt),
                last_message
                ]
    output = llm.invoke(llm_messages)

    print("=" * 50)
    print("Analyst Alise output:")
    print(output)
    print("=" * 50)

    return {"messages": [output]}


def reviewer_node(state):
    topic = state["topic"]

    print(f"Reviewer Mark activated.")

    system_prompt = reviewer_prompt.format(topic=topic)

    last_message = state["messages"][-1].content

    llm_messages = [SystemMessage(content=system_prompt),
                last_message,
                ]

    output = llm.invoke(llm_messages)

    print("=" * 50)
    print("Reviewer Mark output:")
    print(output)
    print("=" * 50)

    return {"messages": [output]}


def define_edge(state):
    messages = state.get("messages", [])

    # Check if the number of messages is 6 or more
    if len(messages) >= 7:
        # Return the END constant and the overall state update
        return END

    # If the condition is not met, return the next node
    return "Reviewer"



app_builder = StateGraph(CustomState)

app_builder.add_node("Analyst", analyst_node)
app_builder.add_node("Reviewer", reviewer_node)

app_builder.set_entry_point("Analyst")

app_builder.add_conditional_edges("Analyst", define_edge, ["Reviewer", END])
app_builder.add_edge("Reviewer", "Analyst")

graph = app_builder.compile()

# Save as PNG
graph_image = graph.get_graph(xray=1).draw_mermaid_png()
with open("graph_diagram.png", "wb") as file:
    file.write(graph_image)
print("Saved as PNG 'graph_diagram.png'")


# Thread configuration and graph input
thread = {"configurable": {"thread_id": "1"}}

user_input = {
    "topic": "Time machine development",
}

graph.invoke(user_input, thread)


