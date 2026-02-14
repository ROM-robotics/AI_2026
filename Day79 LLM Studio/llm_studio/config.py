"""Configuration management for LLM Studio."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────
APP_DIR = Path.home() / ".llm_studio"
MODELS_DIR = APP_DIR / "models"
CONFIG_FILE = APP_DIR / "config.yaml"
LOGS_DIR = APP_DIR / "logs"


@dataclass
class InferenceConfig:
    """Model inference parameters."""
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 2048
    seed: int = -1
    n_batch: int = 512


@dataclass
class ServerConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 1234
    api_key: Optional[str] = None
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    """Main application configuration."""
    models_dir: str = str(MODELS_DIR)
    theme: str = "dark"
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    last_model: Optional[str] = None
    system_prompt: str = "You are a helpful AI assistant."

    def save(self):
        """Save config to YAML file."""
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls) -> "AppConfig":
        """Load config from YAML file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = yaml.safe_load(f) or {}
                inf_data = data.pop("inference", {})
                srv_data = data.pop("server", {})
                return cls(
                    inference=InferenceConfig(**inf_data),
                    server=ServerConfig(**srv_data),
                    **{k: v for k, v in data.items()
                       if k in cls.__dataclass_fields__}
                )
            except Exception:
                pass
        return cls()


def ensure_dirs():
    """Create necessary directories."""
    for d in [APP_DIR, MODELS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
