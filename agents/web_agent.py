# agents/web_agent.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# ****** ADDED/MODIFIED IMPORTS ******
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, StringPromptTemplate # Modified import
from langchain.schema import AgentAction, AgentFinish, OutputParserException # Added import
from langchain.chains import LLMChain # Added import
from typing import List, Union # Added Union
import re # Added import for regex parsing

# Your web_search_tool remains the same
def web_search_tool(query: str) -> str:
    return f"Mock search result for '{query}'. (Implement real web search here)"

# ****** MODIFIED PROMPT TEMPLATE (Example for ReAct style) ******
# NOTE: This is a basic ReAct prompt template. You might need to refine it.
# It instructs the LLM to think step-by-step and specify actions clearly.
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplateForReAct(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

# ****** ADDED OUTPUT PARSER (Example for ReAct style) ******
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # Sometimes the LLM might just return text directly if it thinks it can answer
            # If it does not contain 'Action:', assume it's a final answer
            if "Action:" not in llm_output:
                 return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=llm_output,
                )
            # Otherwise, raise an error for malformed output
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"') # Handle potential quotes
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class WebAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            # Use gemini-1.5-flash or another capable model for complex reasoning
            model="gemini-2.0-flash", # Changed model
            temperature=0.0,
            google_api_key=os.getenv("gemini_api"),
            # Optional: Convert usage to single turn if needed, depending on model/task
            # convert_system_message_to_human=True
        )

        self.tools = [
            Tool(
                name="web_search",
                func=web_search_tool,
                description="Useful for when you need to answer questions about current events or general knowledge." # Improved description
            )
        ]

        # ****** SETUP LLMChain, OutputParser, Prompt ******
        self.agent_prompt = CustomPromptTemplateForReAct(
            template=template,
            tools=self.tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            input_variables=["input", "intermediate_steps"]
        )

        self.llm_chain = LLMChain(llm=self.llm, prompt=self.agent_prompt)

        self.output_parser = CustomOutputParser()

        tool_names = [tool.name for tool in self.tools]

        # ****** CORRECTED LLMSingleActionAgent Initialization ******
        single_action_agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"], # Define the stop sequence
            allowed_tools=tool_names # Good practice to specify allowed tools
        )

        # AgentExecutor remains largely the same, but uses the correctly formed agent
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=single_action_agent,
            tools=self.tools,
            verbose=True,
            # Add handle_parsing_errors=True for robustness
            handle_parsing_errors="Check your output and make sure it conforms to the format."
        )

    def run(self, query: str) -> str:
        """
        Runs the web agent on a general question.
        """
        # Use invoke for newer Langchain versions
        result = self.agent_executor.invoke({"input": query})
        return result.get("output", "No output found.") # Extract output correctly