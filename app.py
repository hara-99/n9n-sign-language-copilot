"""
FastAPI Server for AI Copilot for Sign Language to Speech
Serves REST APIs for perception, decision engine, context NLP, dataset exploration,
and mounts static frontend web application.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

from engine.nlp_engine import ContextEngine
from engine.decision_engine import DecisionEngine
from engine.dataset_service import DatasetService

app = FastAPI(
    title="AI Copilot for Sign Language to Speech",
    description="Real-time Sign Language Recognition, Continuous Sequence Interpretation, Decision Engine & Multilingual Speech",
    version="2.0.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
dataset_service = DatasetService()
context_engine = ContextEngine()
decision_engine = DecisionEngine(profile_name="balanced")

# Pydantic Schemas
class PredictFrameRequest(BaseModel):
    image_b64: str

class NLPReconstructRequest(BaseModel):
    tokens: Optional[List[str]] = None
    raw_text: Optional[str] = None

class DecisionRequest(BaseModel):
    token: Optional[str] = None
    confidence: float = 0.0
    hold_time_ms: float = 0.0
    is_emergency: bool = False
    sentence_complete: bool = False
    top_candidates: Optional[List[Dict[str, Any]]] = None

class SignerProfileRequest(BaseModel):
    profile_name: Optional[str] = None
    confidence_threshold: Optional[float] = None
    clarify_lower_bound: Optional[float] = None
    hold_duration_ms: Optional[float] = None
    debounce_ms: Optional[float] = None
    tremor_filter: Optional[bool] = None

# API Endpoints
@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "app_name": "AI Copilot for Sign Language to Speech",
        "model_loaded": dataset_service.model is not None,
        "classes_count": len(dataset_service.classes),
        "total_dataset_images": dataset_service.get_dataset_stats().get("total_dataset_images", 0),
        "active_profile": decision_engine.profile_name
    }

@app.get("/api/dataset/stats")
async def get_dataset_stats():
    return dataset_service.get_dataset_stats()

@app.get("/api/dataset/sample/{class_name}")
async def get_sample(class_name: str, index: Optional[int] = None):
    sample = dataset_service.get_sample_image(class_name, index)
    if not sample:
        raise HTTPException(status_code=404, detail=f"No samples found for class '{class_name}'")
    return sample

@app.post("/api/predict")
async def predict_frame(payload: PredictFrameRequest):
    result = dataset_service.predict_base64_frame(payload.image_b64)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/api/nlp/reconstruct")
async def reconstruct_nlp(payload: NLPReconstructRequest):
    result = context_engine.reconstruct_sentence(
        tokens=payload.tokens or [],
        raw_text=payload.raw_text
    )
    return result

@app.post("/api/decision")
async def evaluate_decision(payload: DecisionRequest):
    result = decision_engine.evaluate(
        predicted_token=payload.token,
        confidence=payload.confidence,
        hold_time_ms=payload.hold_time_ms,
        is_emergency=payload.is_emergency,
        sentence_complete=payload.sentence_complete,
        top_candidates=payload.top_candidates
    )
    return result

@app.get("/api/decision/stats")
async def get_decision_stats():
    return decision_engine.get_stats()

@app.post("/api/profile")
async def update_profile(payload: SignerProfileRequest):
    data = {k: v for k, v in payload.dict().items() if v is not None}
    decision_engine.update_profile(data)
    return {
        "status": "profile_updated",
        "current_profile": decision_engine.profile_name,
        "config": decision_engine.profile
    }

# Static Files & SPA Route
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Copilot Web UI is initializing..."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
