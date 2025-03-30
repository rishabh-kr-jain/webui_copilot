# orchestrator.py

import os
from typing import Literal, Dict, Any # Added Dict, Any for type hinting invoke results
from langchain_google_genai import ChatGoogleGenerativeAI
# Ensure these imports point to your potentially updated agent classes
from agents.food_security_agent import FoodSecurityAgent
from agents.clinical_agent import ClinicalAgent
from agents.web_agent import WebAgent # Assumes this agent was updated as per previous suggestion

class Orchestrator:
    def __init__(self,
                 food_vectorstore_dir: str,
                 clinical_vectorstore_dir: str):
        """
        Initialize and store references to each agent.
        """
        google_api_key = os.getenv("gemini_api")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        # Assuming these Agent classes are initialized correctly internally
        # (e.g., they also check for the API key if needed)
        self.food_agent = FoodSecurityAgent(food_vectorstore_dir)
        self.clinical_agent = ClinicalAgent(clinical_vectorstore_dir)
        # Ensure WebAgent is initialized correctly (potentially no args needed if self-contained)
        self.web_agent = WebAgent()

        # For question classification - consider gemini-1.5-flash for better instruction following
        self.classifier_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # Or stick with 2.0-flash if preferred
            temperature=0.0,
            google_api_key=google_api_key,
        )

    def classify_question(self, question: str) -> Literal["food", "clinical", "web"]:
        """
        Use an LLM to classify the question into one of three categories.
        """
        # Prompt asking for strictly one word response
        prompt = f"""Classify the following user question into one of three categories:
        1. 'food': Related to UN Food Security, agriculture, food aid, malnutrition statistics.
        2. 'clinical': Related to medical data, clinical trials, healthcare procedures, diseases.
        3. 'web': General knowledge, current events, or topics not covered by 'food' or 'clinical'.

        User Question: "{question}"

        Respond with only the single category word: food, clinical, or web."""

        try:
            # Use invoke() for the LLM call, as direct call is deprecated
            response = self.classifier_llm.invoke(prompt)
            # Access the string content from the response object (e.g., AIMessage)
            raw_response = response.content.strip().lower()
            print(f"Classifier raw response: '{raw_response}'") # Optional: for debugging

            # Prefer exact matching now that the prompt is stricter
            if raw_response == "food":
                return "food"
            elif raw_response == "clinical":
                return "clinical"
            elif raw_response == "web":
                 return "web"
            else:
                # Fallback if the LLM didn't respond as expected
                print(f"Warning: Classifier returned unexpected response: '{raw_response}'. Defaulting to 'web'.")
                return "web"

        except Exception as e:
            print(f"Error during classification LLM call: {e}")
            # Default fallback in case of error
            return "web"

    def run(self, question: str) -> str:
        """
        Classify the question and route to the correct agent's invoke method.
        """
        category = self.classify_question(question)
        print(f"Routing question to: {category}_agent") # Optional: for debugging

        agent_to_use = None
        if category == "food":
            agent_to_use = self.food_agent
        elif category == "clinical":
            agent_to_use = self.clinical_agent
        else: # Default to web agent
            agent_to_use = self.web_agent

        if agent_to_use is None:
             return "Error: Could not determine appropriate agent."

        try:
            # ***** IMPORTANT *****
            # Assuming your agents now have an 'invoke' method (like AgentExecutor)
            # and expect a dictionary input, typically {"input": question}
            # and return a dictionary, typically {"output": answer}
            # Adjust the keys "input" and "output" if your agents use different ones.
            agent_input = {"input": question}
            result: Dict[str, Any] = agent_to_use.invoke(agent_input) # Use invoke

            # Extract the answer from the result dictionary
            final_answer = result.get("output", "Agent did not return a standard output.") # Use .get for safety
            return final_answer

        except AttributeError:
             # Fallback if the agent doesn't have 'invoke' (maybe still uses 'run')
             print(f"Warning: Agent '{category}' does not have an 'invoke' method. Trying 'run'.")
             try:
                 # This assumes agent.run(question) returns a string directly
                 return agent_to_use.run(question)
             except Exception as e:
                 print(f"Error running agent '{category}' with fallback 'run': {e}")
                 return f"An error occurred while processing your request with the {category} agent (fallback)."
        except Exception as e:
            print(f"Error running agent '{category}' with invoke: {e}")
            return f"An error occurred while processing your request with the {category} agent."