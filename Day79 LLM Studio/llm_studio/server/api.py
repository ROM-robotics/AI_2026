"""OpenAI-compatible REST API server for LLM Studio."""

import time
import json
import uuid
import threading
from typing import Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm_studio.config import ServerConfig
from llm_studio.models.engine import InferenceEngine, ChatMessage


# ── Request / Response models ──────────────────────────────────────────

class MessageInput(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: List[MessageInput]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    model: str = "local-model"
    prompt: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None


class EmbeddingRequest(BaseModel):
    model: str = "local-model"
    input: str | List[str]


# ── API Server class ──────────────────────────────────────────────────

class LLMServer:
    """OpenAI-compatible API server wrapping the inference engine."""

    def __init__(self, engine: InferenceEngine, config: ServerConfig):
        self.engine = engine
        self.config = config
        self._app: Optional[FastAPI] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    def _create_app(self) -> FastAPI:
        """Build the FastAPI application."""
        app = FastAPI(
            title="LLM Studio API",
            description="OpenAI-compatible local LLM API server",
            version="1.0.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        engine = self.engine
        api_key = self.config.api_key

        # ── Auth dependency ────────────────────────────────────────
        async def verify_api_key(authorization: Optional[str] = Header(None)):
            if api_key and api_key.strip():
                if not authorization or not authorization.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Missing API key")
                token = authorization.split(" ", 1)[1]
                if token != api_key:
                    raise HTTPException(status_code=401, detail="Invalid API key")

        # ── Routes ─────────────────────────────────────────────────
        @app.get("/v1/models")
        async def list_models(_=Depends(verify_api_key)):
            model_name = "local-model"
            if engine.model_path:
                from pathlib import Path
                model_name = Path(engine.model_path).stem
            return {
                "object": "list",
                "data": [{
                    "id": model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llm-studio",
                }]
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            _=Depends(verify_api_key),
        ):
            if not engine.is_loaded:
                raise HTTPException(status_code=503, detail="No model loaded")

            messages = [
                ChatMessage(role=m.role, content=m.content)
                for m in request.messages
            ]
            kwargs = {}
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.top_p is not None:
                kwargs["top_p"] = request.top_p
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens

            model_name = "local-model"
            if engine.model_path:
                from pathlib import Path
                model_name = Path(engine.model_path).stem

            if request.stream:
                return StreamingResponse(
                    _stream_chat(engine, messages, model_name, kwargs),
                    media_type="text/event-stream",
                )

            content = engine.chat_completion(messages, **kwargs)
            return _make_chat_response(content, model_name)

        @app.post("/v1/completions")
        async def completions(
            request: CompletionRequest,
            _=Depends(verify_api_key),
        ):
            if not engine.is_loaded:
                raise HTTPException(status_code=503, detail="No model loaded")

            kwargs = {}
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens

            model_name = "local-model"
            if engine.model_path:
                from pathlib import Path
                model_name = Path(engine.model_path).stem

            if request.stream:
                return StreamingResponse(
                    _stream_completion(engine, request.prompt, model_name, kwargs),
                    media_type="text/event-stream",
                )

            text = engine.text_completion(request.prompt, **kwargs)
            return _make_completion_response(text, model_name)

        @app.post("/v1/embeddings")
        async def embeddings(
            request: EmbeddingRequest,
            _=Depends(verify_api_key),
        ):
            if not engine.is_loaded:
                raise HTTPException(status_code=503, detail="No model loaded")

            inputs = request.input if isinstance(request.input, list) else [request.input]
            data = []
            for i, text in enumerate(inputs):
                emb = engine.get_embeddings(text)
                data.append({
                    "object": "embedding",
                    "embedding": emb,
                    "index": i,
                })
            return {
                "object": "list",
                "data": data,
                "model": request.model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "model_loaded": engine.is_loaded,
                "model": engine.model_path,
            }

        return app

    # ── Server control ─────────────────────────────────────────────
    def start(self):
        """Start the API server in a background thread."""
        if self._running:
            return

        self._app = self._create_app()
        config = uvicorn.Config(
            self._app,
            host=self.config.host,
            port=self.config.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)

        def _run():
            self._running = True
            self._server.run()
            self._running = False

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        self._running = True

    def stop(self):
        """Stop the API server."""
        if self._server and self._running:
            self._server.should_exit = True
            if self._server_thread:
                self._server_thread.join(timeout=5)
            self._running = False


# ── Helper functions ───────────────────────────────────────────────────

def _make_chat_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _make_completion_response(text: str, model: str) -> dict:
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_chat(engine, messages, model, kwargs):
    """Yield SSE events for streaming chat completion."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    for token in engine.chat_completion_stream(messages, **kwargs):
        data = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"

    # Final chunk
    final = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_completion(engine, prompt, model, kwargs):
    """Yield SSE events for streaming text completion."""
    chunk_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    for token in engine.text_completion_stream(prompt, **kwargs):
        data = {
            "id": chunk_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "text": token,
                "index": 0,
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"
