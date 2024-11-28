import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tavily import TavilyClient
import os
from pydantic import BaseModel
from typing import List, Dict, Tuple

_ = load_dotenv()

class AgentState(TypedDict):
    budget: int                                  # Budget in USD ($)
    weather_preference: str                      # User's preferred weather (e.g., "rainy", "sunny")
    activity_type: str                           # Desired activity type (e.g., "hiking", "beach", "culture")
    nationality: str                             # User's nationality (to determine visa requirements)
    travel_date: datetime.date                   # Departure date
    stay_duration_days: int                      # Duration of the trip in days
    visa_info: Dict[str, bool]                   # Mapping of country names to whether a visa is required
    weather_data: Dict[str, str]                 # Country-specific weather description (e.g., "sunny", "rainy")
    critiques: List[Tuple[str, str]]             # List of critiques (country, critique text)
    iterations: int                              # Current iteration count
    max_iterations: int                          # Maximum number of iterations allowed



VISA_INFO_PROMPT = """
You are a travel assistant helping a user plan their trip. Using the web search tool, 
find the latest visa regulations for citizen's nationality. Your goal is to identify:
Countries they can travel to without a visa.
Countries they can travel to with a visa.
"""


def visa_finder_node(state: AgentState):
    """
    Takes VISA_INFO_PROMPT and user's nationality to find the latest visa regulations.
    """
    messages = [
        SystemMessage(content=VISA_INFO_PROMPT),
        HumanMessage(content=state["nationality"])
    ]
    response = ChatOpenAI(model="gpt-4o", temperature=0.6).invoke(messages)
    return {"plan": response.content}




builder = StateGraph(AgentState)

builder.add_node("visa_finder", visa_finder_node)

builder.set_entry_point("visa_finder")
builder.set_finish_point("visa_finder")

graph = builder.compile()


# Save as PNG
graph_image = graph.get_graph().draw_mermaid_png()
with open("graph_diagram.png", "wb") as file:
    file.write(graph_image)
print("Saved as PNG 'graph_diagram.png'")


# Thread configuration and graph input
thread = {"configurable": {"thread_id": "1"}}

user_input = {
    "budget": 500,  # User budget
    "weather_preference": "not important",  # User does not prioritize weather
    "activity_type": "walking",  # Activity type preferred by the user
    "nationality": "Ukrainian",  # User's nationality
    "travel_date": datetime(2025, 2, 2).date(),  # Travel departure date
    "stay_duration_days": 7,  # Duration of stay in days
    "visa_info": {},  # Initially no visa information
    "weather_data": {},  # Initially no weather data
    "critiques": [],  # Initially no critiques
    "iterations": 0,  # Starting iteration
    "max_iterations": 3  # Maximum allowed iterations
}

# Stream through the graph with the user-defined task
for state in graph.stream(user_input, thread):
    print(state)


# CRITIQUE_PROMPT = """
# You are a travel advisor reviewing potential destinations for a client.
# For each suggested country, provide constructive considerations or potential challenges
# the client should be aware of, while maintaining a positive and helpful tone.
# """


