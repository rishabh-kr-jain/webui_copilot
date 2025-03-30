# agents/web_agent.py

import os
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import BaseChatPromptTemplate
from typing import List

def web_search_tool(query: str) -> str:
    """
    A mock tool function that should call an actual Web API (e.g. Bing, Google).
    For demonstration, we'll just return a placeholder string.
    """
    # TODO: Implement a real web search using your chosen API
    return f"Mock search result for '{query}'. (Implement real web search here)"

class CustomPromptTemplate(BaseChatPromptTemplate):
    """A custom chat prompt instructing the agent how to use the web search tool."""
    def format_messages(self, **kwargs) -> List[dict]:
        user_query = kwargs.get("input", "")
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful AI that can use the 'web_search' tool to answer questions. "
                "If you need external information, call the tool. If you can answer directly, you may do so."
            )
        }
        user_message = {
            "role": "user",
            "content": user_query
        }
        return [system_message, user_message]

class WebAgent:
    def __init__(self):
        self.llm = OpenAI(
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.tools = [
            Tool(
                name="web_search",
                func=web_search_tool,
                description="Search the web for any general question."
            )
        ]
        self.agent_prompt = CustomPromptTemplate(input_variables=["input"])
        
        single_action_agent = LLMSingleActionAgent(
            llm=self.llm,
            prompt=self.agent_prompt,
            tools=self.tools,
            verbose=False
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=single_action_agent,
            tools=self.tools,
            verbose=True
        )

    def run(self, query: str) -> str:
        """
        Runs the web agent on a general question.
        """
        result = self.agent_executor.run(query)
        return result
