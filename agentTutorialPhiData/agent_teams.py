from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.playground import Playground, serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Define the model name to be used by the agents
model_name = "llama-3.3-70b-versatile"
open_ai_model = "gpt-4o"

# Create a web agent with DuckDuckGo tool for web searches
web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id=open_ai_model),
    # model=Groq(id=model_name),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tool_calls=True,
    markdown=True
)

# Create a finance agent with YFinanceTools for financial data
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    # model=Groq(id=model_name),
    model=OpenAIChat(id=open_ai_model),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# Create a team of agents including the web agent and finance agent
agent_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Always include resources", "Use tables to display data"],
    # model=Groq(id=model_name),
    model=OpenAIChat(id=open_ai_model),
    show_tool_calls=True,
    markdown=True,
)


# Print the response from the agent team for the given query
agent_team.print_response("Summarize analyst recommendations and fundamentals for TESLA")