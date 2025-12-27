from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging
from explain_failure import explain_failure

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app (ONLY ONCE)
app = FastAPI(
    title="AI Interview Failure Analyzer",
    description="Predicts interview failure reason and explains skill gaps",
    version="1.0"
)

# Load model & vectorizer
try:
    model = pickle.load(open("model/failure_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    logger.info("Model and vectorizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/vectorizer: {e}")
    model = None
    vectorizer = None

# Input schema
class InterviewInput(BaseModel):
    resume_text: str
    jd_text: str
    self_reflection: str

# Root endpoint
@app.get("/")
def root():
    return {"status": "AI Interview Failure Analyzer running"}

# Health check
@app.get("/health")
def health():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InterviewInput):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not data.resume_text.strip() or not data.jd_text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")

    try:
        result = explain_failure(
            data.resume_text,
            data.jd_text,
            data.self_reflection,
            model,
            vectorizer
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
