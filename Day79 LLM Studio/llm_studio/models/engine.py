"""Inference engine — wraps llama-cpp-python for local model execution."""

import time
import threading
from dataclasses import dataclass
from typing import Generator, List, Optional, Dict, Any

from llm_studio.config import InferenceConfig


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str          # "system" | "user" | "assistant"
    content: str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class InferenceEngine:
    """Manages a loaded LLM and performs inference via llama.cpp."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._model = None
        self._model_path: Optional[str] = None
        self._lock = threading.Lock()
        self._loading = False
        self._load_progress = 0.0

    # ── Properties ─────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def load_progress(self) -> float:
        return self._load_progress

    # ── Model lifecycle ────────────────────────────────────────────
    def load_model(self, model_path: str, callback=None) -> bool:
        """Load a GGUF model file into memory."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )

        with self._lock:
            self._loading = True
            self._load_progress = 0.0

        try:
            # Unload previous model
            self.unload_model()

            def progress_cb(progress: float):
                self._load_progress = progress * 100
                if callback:
                    callback(self._load_progress)

            model = Llama(
                model_path=model_path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                seed=self.config.seed if self.config.seed > 0 else -1,
                verbose=False,
            )

            with self._lock:
                self._model = model
                self._model_path = model_path
                self._loading = False
                self._load_progress = 100.0

            return True

        except Exception as e:
            with self._lock:
                self._loading = False
                self._load_progress = 0.0
            raise RuntimeError(f"Failed to load model: {e}")

    def unload_model(self):
        """Unload the current model from memory."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._model_path = None
                self._load_progress = 0.0

    # ── Inference ──────────────────────────────────────────────────
    def chat_completion(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> str:
        """Generate a chat completion (blocking, full response)."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")

        msg_dicts = [m.to_dict() for m in messages]
        params = self._build_params(kwargs)

        with self._lock:
            response = self._model.create_chat_completion(
                messages=msg_dicts,
                **params,
            )

        content = response["choices"][0]["message"]["content"]
        return content.strip()

    def chat_completion_stream(
        self,
        messages: List[ChatMessage],
        **kwargs,
    ) -> Generator[str, None, None]:
        """Generate a streaming chat completion."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")

        msg_dicts = [m.to_dict() for m in messages]
        params = self._build_params(kwargs)
        params["stream"] = True

        with self._lock:
            stream = self._model.create_chat_completion(
                messages=msg_dicts,
                **params,
            )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    def text_completion(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Generate a plain text completion."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")

        params = self._build_params(kwargs)

        with self._lock:
            response = self._model(prompt, **params)

        return response["choices"][0]["text"]

    def text_completion_stream(
        self,
        prompt: str,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Generate a streaming text completion."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")

        params = self._build_params(kwargs)
        params["stream"] = True

        with self._lock:
            stream = self._model(prompt, **params)

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    # ── Embeddings ─────────────────────────────────────────────────
    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text (if model supports it)."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")
        with self._lock:
            return self._model.embed(text)

    # ── Token utilities ────────────────────────────────────────────
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the loaded model."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")
        return self._model.tokenize(text.encode("utf-8"))

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenize(text))

    def get_model_info(self) -> Dict[str, Any]:
        """Get info about the loaded model."""
        if not self.is_loaded:
            return {}
        return {
            "path": self._model_path,
            "n_ctx": self.config.n_ctx,
            "n_threads": self.config.n_threads,
            "n_gpu_layers": self.config.n_gpu_layers,
            "vocab_size": self._model.n_vocab(),
        }

    # ── Internal ───────────────────────────────────────────────────
    def _build_params(self, overrides: dict) -> dict:
        """Build inference parameters from config + overrides."""
        params = {
            "temperature": overrides.get("temperature", self.config.temperature),
            "top_p": overrides.get("top_p", self.config.top_p),
            "top_k": overrides.get("top_k", self.config.top_k),
            "repeat_penalty": overrides.get("repeat_penalty", self.config.repeat_penalty),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
        }
        return params
