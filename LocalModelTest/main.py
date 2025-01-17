from langchain_core.messages import SystemMessage
from langchain_ollama import OllamaLLM
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph

# Инициализация модели
llm = OllamaLLM(model="phi4")

# Промпты для аналитика и ревьюера
ANALYST_PROMPT = """
You are Alise, a highly skilled technical analyst specializing in {topic}.
Your task is to provide a detailed technical analysis based on the available data.
You should:
1. Identify key technical challenges or opportunities in the provided information.
2. Evaluate the effectiveness and efficiency of different technologies or solutions mentioned.
3. Assess the technical feasibility, scalability, and performance implications.
4. Identify potential technical limitations or risks and propose ways to address them.
5. Recommend further technical steps for investigation or optimization.

If there are previous reports or recommendations from reviewers, critically consider how they affect the technical aspects of the solution and whether there are any inconsistencies or areas needing further technical refinement.
Be precise and objective, with a focus on delivering actionable insights that will lead to improved technical decision-making and problem-solving.
"""

REVIEWER_PROMPT = """
You are Mark, a technical reviewer with deep expertise in {topic}.
Your role is to rigorously assess the technical analysis provided by Alise.
You should:
1. Evaluate the technical soundness of the analysis in terms of accuracy and feasibility.
2. Identify any overlooked technical aspects or potential flaws in the analysis.
3. Suggest improvements or alternative technical approaches where necessary.
4. Highlight any assumptions made that may impact the technical validity of the analysis.

Your feedback should be precise, focusing on improving the technical quality of the analysis, uncovering any weaknesses or oversights, and ensuring the solution is robust and efficient.
"""

# Класс состояния
class CustomState(MessagesState):
    topic: str

# Узлы графа
def analyst_node(state):
    topic = state["topic"]
    print(f"Analyst Alise activated.")

    system_prompt = ANALYST_PROMPT.format(topic=topic)
    messages = state.get("messages", [])

    if messages:
        last_message = (
            "Your previous report: \n" + messages[-2].content + "\n\n\n" +
            "Reviewers recommendations: \n" + messages[-1].content
        )
    else:
        last_message = "This is the beginning of the conversation. Make your initial analysis based on the questionnaire results."

    llm_messages = [
        SystemMessage(content=system_prompt),
        last_message
    ]
    output = llm.invoke(llm_messages)

    print("=" * 50)
    print("Analyst Alise output:")
    print(output)
    print("=" * 50)

    return {"messages": messages + [output]}

def reviewer_node(state):
    topic = state["topic"]
    print(f"Reviewer Mark activated.")

    system_prompt = REVIEWER_PROMPT.format(topic=topic)
    last_message = state["messages"][-1].content

    llm_messages = [
        SystemMessage(content=system_prompt),
        last_message
    ]
    output = llm.invoke(llm_messages)

    print("=" * 50)
    print("Reviewer Mark output:")
    print(output)
    print("=" * 50)

    return {"messages": state["messages"] + [output]}

# Определение переходов
def define_edge(state):
    messages = state.get("messages", [])
    if len(messages) >= 7:  # Завершаем цикл после 7 сообщений
        return END
    return "Reviewer"  # Переход к ревьюеру

# Создание графа
app_builder = StateGraph(CustomState)

app_builder.add_node("Analyst", analyst_node)
app_builder.add_node("Reviewer", reviewer_node)
app_builder.set_entry_point("Analyst")
app_builder.add_conditional_edges("Analyst", define_edge, ["Reviewer", END])
app_builder.add_edge("Reviewer", "Analyst")

# Компиляция графа
graph = app_builder.compile()

# Сохранение визуализации графа
graph_image = graph.get_graph(xray=1).draw_mermaid_png()
with open("graph_diagram.png", "wb") as file:
    file.write(graph_image)
print("Saved as PNG 'graph_diagram.png'")

# Тестовый ввод и запуск графа
thread = {"configurable": {"thread_id": "1"}}
user_input = {
    "topic": "Time machine development",
}

graph.invoke(user_input, thread)
