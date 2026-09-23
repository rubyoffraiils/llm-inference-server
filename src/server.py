"""FastAPI app exposing the model over HTTP."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import load_model
from scheduler import BatchingScheduler

# Loaded once at startup and kept resident, rather than per-request --
# reloading a model on every call would dominate latency.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tokenizer = load_model()
    scheduler = BatchingScheduler(model, tokenizer)
    scheduler.start()
    _state["scheduler"] = scheduler
    yield
    await scheduler.stop()
    _state.clear()


app = FastAPI(lifespan=lifespan)


class ProcessRequest(BaseModel):
    prompt: str


class ProcessResponse(BaseModel):
    output: str


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest) -> ProcessResponse:
    try:
        result = await _state["scheduler"].submit(request.prompt)
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="server at capacity, retry shortly")
    return ProcessResponse(output=result.text)
