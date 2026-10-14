"""
Production REST Microservice for Zelt.
Deployable with Uvicorn, Gunicorn, Docker, or Kubernetes.
"""

import time
from typing import List, Optional

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


if HAS_FASTAPI:
    app = FastAPI(
        title="Zelt AI Reasoning Microservice",
        description="Calibrated, risk-controlled early exit for reasoning Large Language Models.",
        version="0.2.1",
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
        prompt: str = Field(..., description="Reasoning prompt or math word problem.")
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
        prompt: str

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
        prompts: List[str]

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

    @app.post("/v1/reason", response_model=ReasonResponse)
    def reason(req: ReasonRequest):
        import zelt

        try:
            res = zelt.ask(req.prompt)
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

    @app.post("/v1/compare", response_model=CompareResponse)
    def compare(req: CompareRequest):
        import zelt

        try:
            comp = zelt.compare(req.prompt, verbose=False)
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

    @app.post("/v1/batch")
    def batch_process(req: BatchRequest):
        import zelt

        try:
            batch_res = zelt.batch(req.prompts, verbose=False)
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
