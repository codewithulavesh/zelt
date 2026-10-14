"""
Production REST Microservice for Zelt.
Deployable with Uvicorn, Gunicorn, Docker, or Kubernetes.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


if HAS_FASTAPI:
    app = FastAPI(
        title="Zelt AI Reasoning Microservice",
        description="Calibrated, risk-controlled early exit for reasoning Large Language Models.",
        version="0.2.2",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ReasonRequest(BaseModel):
        prompt: Optional[str] = Field(None, description="Reasoning prompt or math word problem.")
        question: Optional[str] = Field(None, description="Alias for prompt.")
        model: Optional[str] = Field(None, description="Model ID or name override.")
        alpha: Optional[float] = Field(0.02, description="Maximum permitted harm rate bound.")
        delta: Optional[float] = Field(0.10, description="Confidence failure probability.")

    class ReasonResponse(BaseModel):
        answer: str
        tokens_used: int
        tokens_estimated_full: int
        tokens_saved: int
        saved_percent: float
        stopped_early: bool
        cost_saved_usd: float
        carbon_saved_grams: float
        elapsed_seconds: float

    class CompareRequest(BaseModel):
        prompt: Optional[str] = Field(None, description="Prompt to compare.")
        question: Optional[str] = Field(None, description="Alias for prompt.")

    class CompareResponse(BaseModel):
        question: str
        full_answer: str
        early_answer: str
        match: bool
        full_tokens: int
        early_tokens: int
        tokens_saved: int
        saved_percent: float
        speedup: float
        stopped: bool

    class BatchRequest(BaseModel):
        prompts: Optional[List[str]] = Field(None, description="List of prompt strings.")
        questions: Optional[List[str]] = Field(None, description="Alias for prompts.")

    @app.get("/health", status_code=status.HTTP_200_OK)
    def health_check():
        return {
            "status": "healthy",
            "service": "zelt-ai",
            "timestamp": time.time(),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics():
        import zelt

        # Return live Prometheus metrics
        batch_sample = zelt.batch(["2+2"], verbose=False)
        return batch_sample.export_prometheus()

    @app.post("/ask", response_model=ReasonResponse)
    @app.post("/v1/reason", response_model=ReasonResponse)
    def reason(req: ReasonRequest):
        import zelt

        query = req.question or req.prompt
        if not query:
            raise HTTPException(
                status_code=422,
                detail="Either 'question' or 'prompt' must be provided in the request body.",
            )

        try:
            res = zelt.ask(query)
            d = res.to_dict()
            return ReasonResponse(
                answer=str(res),
                tokens_used=d["tokens"],
                tokens_estimated_full=d["full_estimated_tokens"],
                tokens_saved=d["tokens_saved"],
                saved_percent=d["saved_percent"],
                stopped_early=d["stopped_early"],
                cost_saved_usd=d["cost_saved_usd"],
                carbon_saved_grams=d["carbon_saved_grams"],
                elapsed_seconds=d["elapsed_seconds"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/stream")
    def stream_endpoint(req: ReasonRequest):
        import zelt

        query = req.question or req.prompt
        if not query:
            raise HTTPException(
                status_code=422,
                detail="Either 'question' or 'prompt' must be provided in the request body.",
            )

        def event_generator():
            for chunk in zelt.stream(query):
                payload = {
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "is_probe": chunk.is_probe,
                    "confidence": chunk.confidence,
                    "score": chunk.score,
                    "stopped": chunk.stopped,
                    "answer": chunk.answer,
                }
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/compare", response_model=CompareResponse)
    @app.post("/v1/compare", response_model=CompareResponse)
    def compare(req: CompareRequest):
        import zelt

        query = req.question or req.prompt
        if not query:
            raise HTTPException(
                status_code=422,
                detail="Either 'question' or 'prompt' must be provided in the request body.",
            )

        try:
            comp = zelt.compare(query, verbose=False)
            d = comp.to_dict()
            return CompareResponse(
                question=d["question"],
                full_answer=str(d["full_answer"]),
                early_answer=str(d["early_answer"]),
                match=d["match"],
                full_tokens=d["full_tokens"],
                early_tokens=d["early_tokens"],
                tokens_saved=d["tokens_saved"],
                saved_percent=d["saved_pct"],
                speedup=d["speedup"],
                stopped=d["stopped"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch")
    @app.post("/v1/batch")
    def batch_process(req: BatchRequest):
        import zelt

        items = req.questions or req.prompts
        if not items:
            raise HTTPException(
                status_code=422,
                detail="Either 'questions' or 'prompts' list must be provided.",
            )

        try:
            batch_res = zelt.batch(items, verbose=False)
            return batch_res.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/info")
    def info():
        import zelt

        return {
            "version": zelt.__version__,
            "presets": zelt.list_presets(),
        }

else:

    def create_app():
        raise RuntimeError(
            "FastAPI is required to run the production microservice. Install with: pip install 'zelt-ai[server]'"
        )
