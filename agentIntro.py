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


llm = ChatOllama(model = "gpt-oss:20b", temperature = 0)
# This tool definition uses the pre-defined TavilySearch tool from langchain_tavily (more accurate, tavily codes better than you do)
tools = [TavilySearch()]
agent = create_agent(model = llm, tools = tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content = "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)


if __name__ == "__main__":
    main()