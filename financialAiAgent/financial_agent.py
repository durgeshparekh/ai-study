import os

from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


load_dotenv()

# web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-70b-versatile"),
    # model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# financial agent
financial_agent = Agent(
    name="Financial AI Agent",
    role="Analyze financial data",
    model=Groq(id="llama-3.1-70b-versatile"),
    # model=OpenAIChat(id="gpt-4o"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# multi-agent
multi_agent = Agent(
    model=Groq(id="llama-3.1-70b-versatile"),
    # model=OpenAIChat(id="gpt-4o"),
    team=[web_search_agent, financial_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)
