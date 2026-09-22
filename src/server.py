"""FastAPI app exposing the model over HTTP."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from model import generate, load_model

# Loaded once at startup and kept resident, rather than per-request --
# reloading a model on every call would dominate latency.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tokenizer = load_model()
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


class ProcessRequest(BaseModel):
    prompt: str


class ProcessResponse(BaseModel):
    output: str


@app.post("/process", response_model=ProcessResponse)
async def process(request: ProcessRequest) -> ProcessResponse:
    output = generate(request.prompt, _state["model"], _state["tokenizer"])
    return ProcessResponse(output=output)
