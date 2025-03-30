# orchestrator.py

import os
from typing import Literal
from langchain.llms import OpenAI

from agents.food_security_agent import FoodSecurityAgent
from agents.clinical_agent import ClinicalAgent
from agents.web_agent import WebAgent

class Orchestrator:
    def __init__(self, 
                 food_vectorstore_dir: str, 
                 clinical_vectorstore_dir: str):
        """
        Initialize and store references to each agent.
        """
        self.food_agent = FoodSecurityAgent(food_vectorstore_dir)
        self.clinical_agent = ClinicalAgent(clinical_vectorstore_dir)
        self.web_agent = WebAgent()

        # For question classification
        self.classifier_llm = OpenAI(
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def classify_question(self, question: str) -> Literal["food", "clinical", "web"]:
        """
        Use a naive LLM-based approach to classify the question 
        into one of three categories.
        """
        prompt = f"""
        You are a classifier. The user question is: "{question}".
        Decide if it is about 'UN Food Security', 'Clinical data', or 'General web'.
        Respond exactly with one of: food / clinical / web
        """
        raw_response = self.classifier_llm(prompt).strip().lower()
        
        # Very naive matching:
        if "food" in raw_response:
            return "food"
        elif "clinic" in raw_response:
            return "clinical"
        else:
            return "web"

    def run(self, question: str) -> str:
        """
        Classify the question and route to the correct agent.
        """
        category = self.classify_question(question)
        if category == "food":
            return self.food_agent.run(question)
        elif category == "clinical":
            return self.clinical_agent.run(question)
        else:
            return self.web_agent.run(question)
