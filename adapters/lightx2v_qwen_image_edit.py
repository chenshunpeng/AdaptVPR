"""HTTP adapter for AdaptVPR's pinned LightX2V Qwen image-edit stack."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
MODEL_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
LORA_ID = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LORA_REVISION = "d74eba145674fd7e31b949324e148e21e7118abd"
LORA_FILENAME = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LIGHTX2V_REPOSITORY = "https://github.com/ModelTC/LightX2V.git"
LIGHTX2V_COMMIT = "522609ecc121b49c20d201b3f00c3dc052821bce"
INFER_STEPS = 4
GUIDANCE_SCALE = 1.0
ATTN_MODE = "torch_sdpa"
SOURCE_REVISION_FILE = ".adaptvpr-source-revision"


class GenerateRequest(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    infer_steps: int = Field(default=4, ge=1)
    guidance_scale: float = Field(default=1.0, gt=0)


class AdapterState:
    pipe = None
    error: str | None = None
    source_commit: str | None = None
    source_modified: bool | None = None


state = AdapterState()
generation_lock = threading.Lock()


def _required_path(name: str) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        raise RuntimeError(f"{name} must point to a downloaded checkpoint")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"{name} does not exist: {path}")
    return path


def _configure_source_checkout() -> str:
    root = _required_path("LIGHTX2V_ROOT")
    configured = os.getenv("LIGHTX2V_GIT_COMMIT", LIGHTX2V_COMMIT).strip()
    if configured != LIGHTX2V_COMMIT:
        raise RuntimeError(f"LIGHTX2V_GIT_COMMIT must remain pinned to {LIGHTX2V_COMMIT}")
    revision_file = root / SOURCE_REVISION_FILE
    if not (root / ".git").exists():
        revision = revision_file.read_text(encoding="utf-8").strip() if revision_file.is_file() else ""
        if revision != LIGHTX2V_COMMIT:
            raise RuntimeError(
                f"LIGHTX2V_ROOT must be a Git checkout or contain {SOURCE_REVISION_FILE}="
                f"{LIGHTX2V_COMMIT}"
            )
        state.source_modified = False
        sys.path.insert(0, str(root))
        return revision
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"LIGHTX2V_ROOT is not a readable Git checkout: {root}") from exc
    if commit != LIGHTX2V_COMMIT:
        raise RuntimeError(f"LightX2V commit mismatch: expected {LIGHTX2V_COMMIT}, found {commit}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    state.source_modified = bool(dirty)
    sys.path.insert(0, str(root))
    return commit


def load_pipeline():
    """Load the exact model/LoRA pair documented for the public adapter."""
    state.source_commit = _configure_source_checkout()
    from lightx2v import LightX2VPipeline

    model_path = _required_path("LIGHTX2V_MODEL_PATH")
    lora_path = _required_path("LIGHTX2V_LORA_PATH")
    pipe = LightX2VPipeline(
        model_path=str(model_path),
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )
    if os.getenv("LIGHTX2V_CPU_OFFLOAD", "1").lower() not in {"0", "false", "no"}:
        pipe.enable_offload(
            cpu_offload=True,
            offload_granularity="block",
            text_encoder_offload=False,
            vae_offload=False,
        )
    pipe.enable_lora(
        [{"path": str(lora_path), "strength": 1.0}],
        lora_dynamic_apply=False,
    )
    pipe.create_generator(
        attn_mode=ATTN_MODE,
        resize_mode="adaptive",
        infer_steps=INFER_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    return pipe


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.pipe = load_pipeline()
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="AdaptVPR LightX2V Qwen adapter", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    ready = state.pipe is not None
    return {
        "status": "ok" if ready else "error",
        "model_loaded": ready,
        "generator_ready": ready,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "lora_id": LORA_ID,
        "lora_revision": LORA_REVISION,
        "lora_filename": LORA_FILENAME,
        "repository": LIGHTX2V_REPOSITORY,
        "source_commit": state.source_commit,
        "source_modified": state.source_modified,
        "expected_commit": LIGHTX2V_COMMIT,
        "sampling": {
            "infer_steps": INFER_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "attn_mode": ATTN_MODE,
            "resize_mode": "adaptive",
        },
        "error": state.error,
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    if state.pipe is None:
        raise HTTPException(status_code=503, detail=state.error or "model is not loaded")

    image_path = Path(request.image_path).resolve()
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail=f"image_path is not readable: {image_path}")

    if request.infer_steps != INFER_STEPS or request.guidance_scale != GUIDANCE_SCALE:
        raise HTTPException(
            status_code=422,
            detail=(
                "request sampling settings differ from the loaded checkpoint configuration: "
                f"infer_steps={INFER_STEPS}, guidance_scale={GUIDANCE_SCALE}"
            ),
        )

    output_dir = Path(os.getenv("LIGHTX2V_OUTPUT_DIR", "/tmp/adaptvpr_lightx2v")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{uuid.uuid4().hex}.png"
    try:
        with generation_lock:
            state.pipe.generate(
                seed=request.seed,
                image_path=str(image_path),
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                save_result_path=str(result_path),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    if not result_path.is_file():
        raise HTTPException(status_code=500, detail="LightX2V returned without writing the output")
    return {"result_path": str(result_path)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("LIGHTX2V_PORT", "8001")))
