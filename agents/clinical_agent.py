# agents/clinical_agent.py

import os
from typing import Dict, Any
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma

CLINICAL_PROMPT_TEMPLATE = """You are an expert in clinical studies.
You have access to structured study documents.
User's question: {question}

Given the data:
{context}

Provide a clear, concise answer, referencing relevant studies if possible." """

class ClinicalAgent:
    def __init__(self, persist_directory: str = "clinical_index"):
        """Initialize a Clinical Agent with a Chroma vectorstore using HuggingFace embeddings."""
        self.persist_directory = persist_directory
        print(f"ClinicalAgent initializing with directory: {self.persist_directory}")

        google_api_key = os.getenv("gemini_api")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set for the LLM.")

        try:
            hf_model_name = "sentence-transformers/all-mpnet-base-v2"
            print(f"Initializing HuggingFace Embeddings model: {hf_model_name}")
            self.embedding_function = HuggingFaceEmbeddings(model_name=hf_model_name)
        except Exception as e:
             raise RuntimeError(f"Failed to initialize HuggingFaceEmbeddings model '{hf_model_name}'. "
                                f"Ensure 'sentence-transformers' and 'torch'/'tensorflow' are installed. Error: {e}")

        try:
            print(f"Loading Chroma DB from: '{self.persist_directory}' using embeddings: '{hf_model_name}'")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            print("Chroma DB loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Chroma vector store from '{self.persist_directory}'. "
                               f"Ensure the directory exists and contains a valid Chroma database "
                               f"created with the '{hf_model_name}' embeddings. Error: {e}")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=google_api_key,
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=CLINICAL_PROMPT_TEMPLATE
        )

        chain_type_kwargs = {"prompt": self.prompt}

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=False
        )

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the RetrievalQA chain using the invoke method.
        Expects input like {"input": "user question"}.
        Returns output like {"output": "answer"}.
        """
        query = input_data.get("input")
        if not query:
            print("Warning: No 'input' key found in invoke data.")
            return {"output": "Error: Missing 'input' key in request."}

        try:
            chain_input = {"query": query}
            print(f"Invoking qa_chain with query: '{query[:50]}...'")
            result = self.qa_chain.invoke(chain_input)
            answer = result.get("result", "Agent did not return a result.")
            print(f"qa_chain returned answer: '{answer[:50]}...'")
            return {"output": answer}
        except Exception as e:
            print(f"Error during ClinicalAgent invoke: {e}")
            return {"output": f"An error occurred while processing your request: {e}"}

    # Removed the 'run' method to maintain consistency with FoodSecurityAgent