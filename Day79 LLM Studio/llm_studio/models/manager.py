"""Model manager — download, list, delete GGUF models."""

import os
import time
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Callable

from huggingface_hub import hf_hub_download, HfApi, list_repo_tree

from llm_studio.config import AppConfig, MODELS_DIR


@dataclass
class ModelInfo:
    """Metadata for a locally stored model."""
    name: str
    filename: str
    path: str
    size_bytes: int
    quantization: str = "unknown"

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def size_display(self) -> str:
        if self.size_bytes < 1024 ** 3:
            return f"{self.size_bytes / (1024 ** 2):.1f} MB"
        return f"{self.size_gb:.2f} GB"


@dataclass
class HFModelFile:
    """A GGUF file available on HuggingFace."""
    repo_id: str
    filename: str
    size_bytes: int

    @property
    def size_display(self) -> str:
        if self.size_bytes < 1024 ** 3:
            return f"{self.size_bytes / (1024 ** 2):.1f} MB"
        return f"{self.size_bytes / (1024 ** 3):.2f} GB"


class ModelManager:
    """Manage local GGUF model files."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ── Local models ───────────────────────────────────────────────
    def list_local_models(self) -> List[ModelInfo]:
        """List all local GGUF model files."""
        models = []
        for f in sorted(self.models_dir.rglob("*.gguf")):
            quant = self._detect_quantization(f.name)
            models.append(ModelInfo(
                name=f.stem,
                filename=f.name,
                path=str(f),
                size_bytes=f.stat().st_size,
                quantization=quant,
            ))
        return models

    def delete_model(self, filename: str) -> bool:
        """Delete a model file."""
        path = self.models_dir / filename
        if path.exists():
            path.unlink()
            return True
        return False

    def get_model_path(self, filename: str) -> Optional[str]:
        """Get full path for a model filename."""
        path = self.models_dir / filename
        if path.exists():
            return str(path)
        return None

    # ── HuggingFace integration ────────────────────────────────────
    def search_hf_models(self, query: str, limit: int = 20) -> list:
        """Search HuggingFace for GGUF models."""
        api = HfApi()
        results = []
        try:
            models = api.list_models(
                search=query,
                sort="downloads",
                direction=-1,
                limit=limit,
            )
            for m in models:
                if "gguf" in (m.id or "").lower() or any(
                    "gguf" in t.lower() for t in (m.tags or [])
                ):
                    results.append({
                        "repo_id": m.id,
                        "downloads": getattr(m, "downloads", 0),
                        "likes": getattr(m, "likes", 0),
                    })
        except Exception:
            pass
        return results

    def list_gguf_files(self, repo_id: str) -> List[HFModelFile]:
        """List available GGUF files in a HuggingFace repo."""
        files = []
        try:
            api = HfApi()
            siblings = api.model_info(repo_id).siblings or []
            for s in siblings:
                if s.rfilename.endswith(".gguf"):
                    files.append(HFModelFile(
                        repo_id=repo_id,
                        filename=s.rfilename,
                        size_bytes=s.size or 0,
                    ))
        except Exception:
            pass
        return files

    def download_model(
        self,
        repo_id: str,
        filename: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional[str]:
        """Download a GGUF model from HuggingFace."""
        try:
            dest = self.models_dir / filename
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.models_dir),
                local_dir_use_symlinks=False,
            )
            if progress_callback:
                progress_callback(100.0)
            return path
        except Exception as e:
            return None

    # ── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _detect_quantization(filename: str) -> str:
        """Detect quantization type from filename."""
        name = filename.upper()
        quant_types = [
            "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
            "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
            "Q6_K", "Q8_0", "F16", "F32",
            "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_XS",
            "IQ4_NL", "IQ4_XS",
        ]
        for qt in quant_types:
            if qt in name:
                return qt
        return "unknown"

    def get_storage_usage(self) -> dict:
        """Get storage usage info."""
        total = 0
        count = 0
        for f in self.models_dir.rglob("*.gguf"):
            total += f.stat().st_size
            count += 1
        disk = shutil.disk_usage(str(self.models_dir))
        return {
            "model_count": count,
            "total_size": total,
            "disk_free": disk.free,
            "disk_total": disk.total,
        }
