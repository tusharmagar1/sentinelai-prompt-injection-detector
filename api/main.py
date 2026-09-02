from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib

from src.risk_engine import calculate_risk
from fastapi.middleware.cors import CORSMiddleware 

# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="SentinelAI - Prompt Injection Detector",
    description="AI-powered API for detecting prompt injection attacks.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --------------------------------------------------
# Request Model
# --------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str


# --------------------------------------------------
# Response Model
# --------------------------------------------------

class PredictionResponse(BaseModel):
    prompt: str
    prediction: str
    injection_probability: float
    risk_score: float
    action: str


# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

classifier = joblib.load(
    "models/embedding_classifier.pkl"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "SentinelAI Prompt Injection Detector API",
        "status": "running"
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PromptRequest):

    # Convert prompt into semantic embedding
    embedding = embedding_model.encode(
        [request.prompt]
    )

    # Predict class
    prediction = classifier.predict(
        embedding
    )[0]

    # Get probability for each class
    probabilities = classifier.predict_proba(
        embedding
    )[0]

    # IMPORTANT:
    # Assuming class 1 = prompt_injection
    injection_probability = float(
        probabilities[1]
    )

    # Calculate risk and action
    risk = calculate_risk(
        injection_probability
    )

    # Return API response
    return PredictionResponse(
        prompt=request.prompt,
        prediction=(
            "prompt_injection"
            if prediction == 1
            else "benign"
        ),
        injection_probability=round(
            injection_probability,
            4
        ),
        risk_score=risk["risk_score"],
        action=risk["action"]
    )