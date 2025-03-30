# agents/food_security_agent.py

import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Replace with Gemini/PaLM if needed
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Chroma

FOOD_SECURITY_PROMPT_TEMPLATE = """
You are an expert on United Nations food security. 
You have access to the following document excerpts:
{context}

User's question: {question}

Provide a concise, accurate answer based on the given context. 
If the answer is not in the context, say 'I'm not sure based on the information I have.' 
"""

class FoodSecurityAgent:
    def __init__(self, persist_directory: str):
        """Initialize a Food Security Agent with a Chroma vectorstore."""
        self.persist_directory = persist_directory
        
        # Load the vectorstore
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory
        )
        # Create a retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.llm = OpenAI(
            temperature=0.0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=FOOD_SECURITY_PROMPT_TEMPLATE
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
        )

    def run(self, query: str) -> str:
        result = self.qa_chain.run(query)
        return result
