# agents/food_security_agent.py

import os
from typing import Dict, Any
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI # Still use Gemini for the generation step
# === CHANGE 1: Import HuggingFaceEmbeddings ===
# Use the community embeddings module
from langchain_community.embeddings import HuggingFaceEmbeddings
# If the above fails, try: from langchain.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
# Use the community vectorstores module
from langchain_community.vectorstores import Chroma
# If the above fails, try: from langchain.vectorstores import Chroma


# Prompt remains the same
FOOD_SECURITY_PROMPT_TEMPLATE = """You are an expert assistant specializing in United Nations food security data and reports. Your knowledge is based solely on the following document excerpts.

Context:
{context}

User's Question: {question}

Based ONLY on the provided context, answer the user's question concisely and accurately. If the answer cannot be found within the context, state explicitly: 'Based on the provided documents, I cannot answer that question.' Do not add any information or interpretation not present in the context."""

class FoodSecurityAgent:
    # === CHANGE 2: Update default persist_directory to match ingestion script ===
    # Although the orchestrator passes the directory, setting a matching default is good practice
    def __init__(self, persist_directory: str = "un_food_index"):
        """Initialize a Food Security Agent with a Chroma vectorstore, using HuggingFace embeddings."""
        self.persist_directory = persist_directory
        print(f"FoodSecurityAgent initializing with directory: {self.persist_directory}")

        google_api_key = os.getenv("gemini_api")
        if not google_api_key:
            # API key is needed for the LLM part, even if embeddings are local
            raise ValueError("GOOGLE_API_KEY environment variable not set for the LLM.")

        # === CHANGE 3: Use the EXACT SAME HuggingFace embedding model as ingestion ===
        try:
            # Specify the model used in your ingestion script
            hf_model_name = "sentence-transformers/all-mpnet-base-v2"
            print(f"Initializing HuggingFace Embeddings model: {hf_model_name}")
            # You might want to specify device ('cpu', 'cuda') if needed
            # model_kwargs = {'device': 'cpu'} # Uncomment and adjust if necessary
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=hf_model_name,
                # model_kwargs=model_kwargs # Uncomment if using model_kwargs
            )
        except Exception as e:
             raise RuntimeError(f"Failed to initialize HuggingFaceEmbeddings model '{hf_model_name}'. "
                                f"Ensure 'sentence-transformers' and 'torch'/'tensorflow' are installed. Error: {e}")

        # Load the vectorstore, providing the CORRECT HuggingFace embedding function
        try:
            print(f"Loading Chroma DB from: '{self.persist_directory}' using embeddings: '{hf_model_name}'")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function # Pass the initialized HF embedder
            )
            print("Chroma DB loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Chroma vector store from '{self.persist_directory}'. "
                               f"Ensure the directory exists and contains a valid Chroma database "
                               f"created with the '{hf_model_name}' embeddings. Error: {e}")

        # --- The rest of the agent remains the same as the previous corrected version ---

        # Create a retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Retrieve top 3 documents
        )

        # LLM for generation (still Google Gemini)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # Or your preferred Gemini model
            temperature=0.3,
            google_api_key=google_api_key,
        )

        # Define the prompt template
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=FOOD_SECURITY_PROMPT_TEMPLATE
        )

        # Pass the custom prompt to RetrievalQA
        chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=False
        )

    # invoke method remains the same (handles dict input/output)
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the RetrievalQA chain using the invoke method.
        Expects input like {"input": "user question"} (matching orchestrator).
        Returns output like {"output": "answer"} (matching orchestrator).
        """
        query = input_data.get("input")
        if not query:
            print("Warning: No 'input' key found in invoke data.")
            return {"output": "Error: Missing 'input' key in request."}

        try:
            # RetrievalQA expects the query under the key "query"
            chain_input = {"query": query}
            print(f"Invoking qa_chain with query: '{query[:50]}...'") # Log query start
            result = self.qa_chain.invoke(chain_input)

            # Standard output key for RetrievalQA is "result"
            answer = result.get("result", "Agent did not return a result.")
            print(f"qa_chain returned answer: '{answer[:50]}...'") # Log answer start

            # Return the answer under the key "output" to match orchestrator
            return {"output": answer}

        except Exception as e:
            print(f"Error during FoodSecurityAgent invoke: {e}")
            return {"output": f"An error occurred while processing your request: {e}"}