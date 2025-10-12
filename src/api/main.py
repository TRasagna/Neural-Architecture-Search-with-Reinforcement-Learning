"""FastAPI service for NAS model serving and search management."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import torch
import json
import uuid
from datetime import datetime
import asyncio
import os

from ..child.builder import ChildModelBuilder
from ..search_space.cnn_space import CNNSearchSpace
from ..search_space.transformer_space import TransformerSearchSpace
from ..ray.run_search import run_distributed_search
from ..mlops.model_registry import ModelRegistry
from ..utils.config import load_config


app = FastAPI(
    title="NAS-RL API",
    description="Neural Architecture Search with Reinforcement Learning API",
    version="1.0.0"
)

security = HTTPBearer()

# Global variables
model_registry = ModelRegistry()
active_searches = {}


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    architecture_id: str
    input_data: List[List[float]]
    model_type: str = "cnn"


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[List[float]]
    architecture_id: str
    model_type: str
    inference_time_ms: float


class SearchRequest(BaseModel):
    """Request model for starting new search."""
    search_name: str
    config: Dict[str, Any]
    priority: str = "normal"
    notification_webhook: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search status."""
    search_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    best_architectures: Optional[List[Dict]] = None
    start_time: str
    estimated_completion: Optional[str] = None


class ArchitectureInfo(BaseModel):
    """Architecture information model."""
    architecture_id: str
    specification: Dict[str, Any]
    metrics: Dict[str, float]
    complexity: Dict[str, float]
    created_at: str


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    token = credentials.credentials
    expected_token = os.getenv("API_SECRET_KEY", "default-secret-key")

    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/best-architectures", response_model=List[ArchitectureInfo])
async def get_best_architectures(
    limit: int = 10,
    model_type: str = "cnn",
    token: str = Depends(verify_token)
):
    """Get best performing architectures."""
    try:
        architectures = model_registry.get_best_architectures(
            limit=limit, 
            model_type=model_type
        )

        return [
            ArchitectureInfo(
                architecture_id=arch["id"],
                specification=arch["spec"],
                metrics=arch["metrics"],
                complexity=arch.get("complexity", {}),
                created_at=arch["created_at"]
            )
            for arch in architectures
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """Make predictions using specified architecture."""
    import time
    start_time = time.time()

    try:
        # Get architecture from registry
        architecture = model_registry.get_architecture(request.architecture_id)
        if not architecture:
            raise HTTPException(status_code=404, detail="Architecture not found")

        # Build model
        if request.model_type == "cnn":
            search_space = CNNSearchSpace()
        else:
            search_space = TransformerSearchSpace()

        builder = ChildModelBuilder(search_space)
        model = builder.build_model(architecture["spec"])

        # Load weights if available
        if "checkpoint_path" in architecture:
            checkpoint = torch.load(architecture["checkpoint_path"], map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])

        model.eval()

        # Make predictions
        input_tensor = torch.tensor(request.input_data, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = torch.softmax(outputs, dim=-1).tolist()

        inference_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            predictions=predictions,
            architecture_id=request.architecture_id,
            model_type=request.model_type,
            inference_time_ms=inference_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trigger-search", response_model=SearchResponse)
async def trigger_search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Trigger new architecture search."""
    try:
        search_id = str(uuid.uuid4())

        # Validate config
        required_fields = ["search", "dataset", "controller"]
        for field in required_fields:
            if field not in request.config:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required config field: {field}"
                )

        # Store search info
        search_info = {
            "id": search_id,
            "name": request.search_name,
            "status": "queued",
            "config": request.config,
            "priority": request.priority,
            "webhook": request.notification_webhook,
            "start_time": datetime.now().isoformat(),
            "progress": {"architectures_evaluated": 0, "best_accuracy": 0.0}
        }

        active_searches[search_id] = search_info

        # Start search in background
        background_tasks.add_task(
            run_search_background,
            search_id,
            request.config,
            request.notification_webhook
        )

        return SearchResponse(
            search_id=search_id,
            status="queued",
            start_time=search_info["start_time"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search-status/{search_id}", response_model=SearchResponse)
async def get_search_status(
    search_id: str,
    token: str = Depends(verify_token)
):
    """Get status of running search."""
    if search_id not in active_searches:
        raise HTTPException(status_code=404, detail="Search not found")

    search_info = active_searches[search_id]

    return SearchResponse(
        search_id=search_id,
        status=search_info["status"],
        progress=search_info.get("progress"),
        best_architectures=search_info.get("best_architectures"),
        start_time=search_info["start_time"],
        estimated_completion=search_info.get("estimated_completion")
    )


@app.delete("/search/{search_id}")
async def cancel_search(
    search_id: str,
    token: str = Depends(verify_token)
):
    """Cancel running search."""
    if search_id not in active_searches:
        raise HTTPException(status_code=404, detail="Search not found")

    search_info = active_searches[search_id]

    if search_info["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Search cannot be cancelled")

    search_info["status"] = "cancelled"

    return {"message": f"Search {search_id} cancelled"}


@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring."""
    return {
        "active_searches": len([s for s in active_searches.values() if s["status"] == "running"]),
        "total_searches": len(active_searches),
        "registered_architectures": model_registry.count_architectures(),
        "system_load": get_system_metrics()
    }


async def run_search_background(
    search_id: str,
    config: Dict,
    webhook: Optional[str] = None
):
    """Run search in background task."""
    try:
        # Update status
        active_searches[search_id]["status"] = "running"

        # Save config to temporary file
        config_path = f"/tmp/search_config_{search_id}.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config, f)

        # Run search
        results = run_distributed_search(config_path)

        # Process results
        best_results = results.get_best_result("val_acc", "max")

        # Update search info
        active_searches[search_id].update({
            "status": "completed",
            "best_architectures": [best_results.config],
            "final_metrics": best_results.metrics,
            "completion_time": datetime.now().isoformat()
        })

        # Register best architecture
        model_registry.register_architecture(
            architecture_spec=best_results.config["architecture_spec"],
            metrics=best_results.metrics,
            search_id=search_id
        )

        # Send webhook notification if provided
        if webhook:
            await send_webhook_notification(webhook, search_id, "completed")

    except Exception as e:
        active_searches[search_id].update({
            "status": "failed",
            "error": str(e),
            "completion_time": datetime.now().isoformat()
        })

        if webhook:
            await send_webhook_notification(webhook, search_id, "failed", str(e))


async def send_webhook_notification(webhook: str, search_id: str, status: str, error: str = None):
    """Send notification to webhook."""
    import aiohttp

    payload = {
        "search_id": search_id,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

    if error:
        payload["error"] = error

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook, json=payload) as response:
                pass
    except Exception:
        pass  # Ignore webhook failures


def get_system_metrics() -> Dict[str, float]:
    """Get basic system metrics."""
    import psutil

    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
