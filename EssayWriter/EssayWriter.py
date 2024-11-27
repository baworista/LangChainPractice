import sqlite3
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tavily import TavilyClient
import os

_ = load_dotenv()

db_path = "EssayWriterMemory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)

class AgentState(TypedDict):
    task: str                   # start task from user
    plan: str                   # plan generated by llm
    draft: str                  # tmp draft of essay
    critique: str               # critique generated by llm
    content: List[str]          # list of info that were found by researcher llm with tools
    revision_number: int        # current revision num
    max_revisions: int          # max revisions num

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""


WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


class Queries(BaseModel):
    queries: List[str]

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def plan_node(state: AgentState):
    """
    Takes plan_prompt and user's task from state and takes response from model.
    """
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    """
    Ask model about three search queries.
    It's return must be an object strictly specified by Pydantic(list of strings in our case).
    We use tavily research then on it.
    """
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


def generation_node(state: AgentState):
    """
    Connect all parts(task, plan and content) to generate version of essay.
    Return draft and increased revision number.
    """
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1
    }


def reflection_node(state: AgentState):
    """
    Takes draft and reflection prompt to generate critique and recommendations.
    """
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


def should_continue(state):
    """
    Checks if we can make one more iteration.
    """
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


builder = StateGraph(AgentState)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate",
    should_continue,
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer=memory)

# Save as PNG
graph_image = graph.get_graph().draw_mermaid_png()
with open("graph_diagram.png", "wb") as file:
    file.write(graph_image)
print("Saved as PNG 'graph_diagram.png'")

# Now lets try use it
# thread = {"configurable": {"thread_id": "1"}}
# for s in graph.stream({
#     'task': "what is the difference between langchain and langsmith",
#     "max_revisions": 3,
#     "revision_number": 1,
#     "content": []
# }, thread):
#     print(s)

