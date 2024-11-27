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

class AgentState(TypedDict):
    task: str                   # start task from user
    plan: str                   # plan generated by llm
    draft: str                  # tmp draft of essay
    critique: str               # critique generated by llm
    content: List[str]          # list of info that were found by researcher llm with tools
    revision_number: int        # current revision num
    max_revisions: int          # max revisions num


from pydantic import BaseModel
from typing import List, Dict, Tuple

class AgentState(TypedDict):
    budget: int                                 #budget in $
    weather_preference: str                     #string with user preferences e.g.rainy
    activity_type: str
    nationality: str
    travel_date: datetime.date
    stay_duration_days: int
    potential_destinations: List[str] = []
    visa_info: Dict[str, bool] = {}
    weather_data: Dict[str, Dict] = {}
    critiques: List[str, str] = []
    iterations: int = 0
    max_iterations: int = 3
