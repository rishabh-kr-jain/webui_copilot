# main.py

import os
from fastapi import FastAPI
from pydantic import BaseModel

from orchestrator import Orchestrator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import requests
from dotenv import load_dotenv
load_dotenv()

# Paths to your persisted vectorstores
UN_VECTORSTORE_DIR = "un_food_index"
CLINICAL_VECTORSTORE_DIR = "clinical_index"

# Initialize orchestrator
orchestrator = Orchestrator(
    food_vectorstore_dir=UN_VECTORSTORE_DIR,
    clinical_vectorstore_dir=CLINICAL_VECTORSTORE_DIR
)

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint: US GDP over 100 years (values in nominal trillions USD)
@app.get("/api/gdp-usa-100yrs")
def get_gdp_usa_100yrs():
    data = [
        {"year": 1925, "gdp": 1.0},
        {"year": 1930, "gdp": 0.9},
        {"year": 1935, "gdp": 1.1},
        {"year": 1940, "gdp": 1.2},
        {"year": 1945, "gdp": 1.4},
        {"year": 1950, "gdp": 1.6},
        {"year": 1955, "gdp": 1.7},
        {"year": 1960, "gdp": 2.0},
        {"year": 1965, "gdp": 2.2},
        {"year": 1970, "gdp": 2.1},  # slight dip
        {"year": 1975, "gdp": 2.3},
        {"year": 1980, "gdp": 2.8},
        {"year": 1985, "gdp": 2.7},  # minor drop
        {"year": 1990, "gdp": 3.2},
        {"year": 1995, "gdp": 3.8},
        {"year": 2000, "gdp": 4.5},
        {"year": 2005, "gdp": 4.3},  # slight dip
        {"year": 2010, "gdp": 5.0},
        {"year": 2015, "gdp": 5.8},
        {"year": 2020, "gdp": 5.6},  # dip during downturn
        {"year": 2025, "gdp": 6.2}
    ]
    return {"data": data}

# Endpoint: Global CO₂ Emissions over 50 years (values in million metric tons)
@app.get("/api/co2-world-50yrs")
def get_co2_world_50yrs():
    data = [
        {"year": 1975, "co2": 15000},
        {"year": 1980, "co2": 15700},
        {"year": 1985, "co2": 15500},  # slight dip
        {"year": 1990, "co2": 16000},
        {"year": 1995, "co2": 15800},  # dip
        {"year": 2000, "co2": 16500},
        {"year": 2005, "co2": 16200},  # slight decline
        {"year": 2010, "co2": 16800},
        {"year": 2015, "co2": 16500},  # small drop
        {"year": 2020, "co2": 17000},
        {"year": 2025, "co2": 16800}   # slight decline
    ]
    return {"data": data}

# Endpoint: Global Agricultural Land Area over 50 years (values in square kilometers)
@app.get("/api/agri-land-world-50yrs")
def get_agri_land_world_50yrs():
    data = [
        {"year": 1975, "agriLand": 49500000},
        {"year": 1980, "agriLand": 49480000},  # slight dip
        {"year": 1985, "agriLand": 49520000},  # rise
        {"year": 1990, "agriLand": 49500000},  # dip
        {"year": 1995, "agriLand": 49530000},  # rise
        {"year": 2000, "agriLand": 49510000},  # dip
        {"year": 2005, "agriLand": 49540000},  # rise
        {"year": 2010, "agriLand": 49520000},  # slight dip
        {"year": 2015, "agriLand": 49550000},  # rise
        {"year": 2020, "agriLand": 49530000},  # dip
        {"year": 2025, "agriLand": 49560000}   # rise
    ]
    return {"data": data}

# Endpoint: Fourth Dataset remains unchanged (example data)
@app.get("/api/fourth-dataset")
def get_fourth_dataset():
    data = [
        {"year": 1975, "value": 100},
        {"year": 1980, "value": 110},
        {"year": 1985, "value": 120},
        {"year": 1990, "value": 130},
        {"year": 1995, "value": 140},
        {"year": 2000, "value": 150},
        {"year": 2005, "value": 160},
        {"year": 2010, "value": 170},
        {"year": 2015, "value": 180},
        {"year": 2020, "value": 190},
        {"year": 2025, "value": 200}
    ]
    return {"data": data}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    user_question = request.question
    answer = orchestrator.run(user_question)
    return ChatResponse(answer=answer)
