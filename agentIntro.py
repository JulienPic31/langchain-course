from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import tavily

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

# Example of a custom defined tool (using the tavily package)
# @tool
# def search_tool(query: str) -> str:
#     """
#     Tool that searches over the internet 
#     Args:
#         query: The query to search for
#     Returns:
#         The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query = query)

# Example of defining a custom response schematic of our agent
# The Source class defines the base structure that a source should have
# The AgentResponse class then defines how the agent provides the answer to the prompt, in this case it provides the answer and a list of the sources it used to generate that answer
class Source (BaseModel):
    """Schematic for a source used by the agent"""

    url: str = Field (description = "The URL of the source")

class AgentResponse (BaseModel):
    """Schematic for agent response with answer and sources"""

    answer: str = Field (description = "The agent's answer to the query")
    sources: List[Source] = Field (default_factory = list, description = "The list of sources used to generate the answer")



llm = ChatOllama(model = "gpt-oss:20b", temperature = 0)
# This tool definition uses the pre-defined TavilySearch tool from langchain_tavily (more accurate, tavily codes better than you do)
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools, response_format = AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content = "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)


if __name__ == "__main__":
    main()