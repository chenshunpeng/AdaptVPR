"""FastAPI adapter for AdaptVPR's pinned official IC-Light stack."""

from __future__ import annotations

import os
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ICLIGHT_REPOSITORY = "https://github.com/lllyasviel/IC-Light.git"
ICLIGHT_COMMIT = "bcf3f29ca85be8a4686215f477b546f5030be8b7"
BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_MODEL_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CHECKPOINT_ID = "lllyasviel/ic-light"
CHECKPOINT_REVISION = "9cad1878695f546a7fb9eaca14e2a89131ba5ffe"
CHECKPOINT_FILENAME = "iclight_sd15_fc.safetensors"
SOURCE_REVISION_FILE = ".adaptvpr-source-revision"
DEFAULT_HIGHRES_SCALE = 1.0
DEFAULT_HIGHRES_DENOISE = 0.30
RAIN_HIGHRES_DENOISE = 0.22
DEFAULT_INFERENCE_STEPS = 25
DEFAULT_HIGHRES_STEPS = 20
DEFAULT_NEGATIVE_PROMPT = (
    "lowres, bad quality, blurry, distorted buildings, warped road, deformed facade, "
    "extra text, changed signage, cartoon, painting, crowd, traffic jam, close-up person, "
    "close-up car, large foreground object, blocked main building"
)


class GenerateRequest(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    highres_scale: float = Field(default=DEFAULT_HIGHRES_SCALE, ge=0)
    highres_denoise: float = Field(default=DEFAULT_HIGHRES_DENOISE, gt=0, le=1)
    num_inference_steps: int = Field(default=DEFAULT_INFERENCE_STEPS, ge=1)
    highres_steps: int = Field(default=DEFAULT_HIGHRES_STEPS, ge=1)


class AdapterState:
    pipe_t2i = None
    pipe_i2i = None
    vae = None
    error: str | None = None
    source_commit: str | None = None


state = AdapterState()
generation_lock = threading.Lock()


def _required_path(name: str) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        raise RuntimeError(f"{name} must point to a downloaded dependency")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"{name} does not exist: {path}")
    return path


def _verify_source_checkout() -> str:
    raw_root = os.getenv("ICLIGHT_ROOT", "").strip()
    if not raw_root:
        return ""
    root = Path(raw_root).expanduser().resolve()
    if not root.exists():
        return ""
    revision_file = root / SOURCE_REVISION_FILE
    if not (root / ".git").exists():
        revision = revision_file.read_text(encoding="utf-8").strip() if revision_file.is_file() else ""
        return revision if revision == ICLIGHT_COMMIT else ""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"ICLIGHT_ROOT is not a readable Git checkout: {root}") from exc
    return commit if commit == ICLIGHT_COMMIT else ""


def _configure_unet(unet, checkpoint_path: Path) -> None:
    import safetensors.torch as sf
    import torch

    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            8,
            unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    original_forward = unet.forward

    def hooked_forward(sample, timestep, encoder_hidden_states, **kwargs):
        concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
        concat = torch.cat([concat] * (sample.shape[0] // concat.shape[0]), dim=0)
        kwargs["cross_attention_kwargs"] = {}
        return original_forward(
            torch.cat([sample, concat], dim=1), timestep, encoder_hidden_states, **kwargs
        )

    unet.forward = hooked_forward
    offset = sf.load_file(str(checkpoint_path), device="cpu")
    original = unet.state_dict()
    unet.load_state_dict({key: original[key] + offset[key] for key in original}, strict=True)


def load_pipeline() -> tuple[object, object, object]:
    import torch
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the IC-Light adapter")
    state.source_commit = _verify_source_checkout() or None
    base_model_path = _required_path("ICLIGHT_BASE_MODEL_PATH")
    checkpoint_path = _required_path("ICLIGHT_MODEL_PATH")
    if checkpoint_path.name != CHECKPOINT_FILENAME:
        raise RuntimeError(
            f"ICLIGHT_MODEL_PATH must select {CHECKPOINT_FILENAME}, found {checkpoint_path.name}"
        )

    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae").to("cuda", dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    _configure_unet(unet, checkpoint_path)
    unet = unet.to("cuda", dtype=dtype)
    pipe_t2i = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        unet=unet,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe_t2i.scheduler = DDIMScheduler.from_config(pipe_t2i.scheduler.config)
    pipe_t2i.set_progress_bar_config(disable=True)
    pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
        base_model_path,
        unet=unet,
        vae=vae,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe_i2i.scheduler = pipe_t2i.scheduler
    pipe_i2i.set_progress_bar_config(disable=True)
    return pipe_t2i, pipe_i2i, vae


def _valid_size(image) -> tuple[int, int]:
    width, height = image.size
    return max(8, width // 8 * 8), max(8, height // 8 * 8)


def _concat_condition(image, vae, width: int, height: int):
    import numpy as np
    import torch

    resized = image.resize((width, height))
    array = np.asarray(resized).astype("float32") / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16)
    with torch.inference_mode():
        return vae.encode(tensor).latent_dist.mode() * vae.config.scaling_factor


def _validate_sampling(request: GenerateRequest) -> None:
    expected = {
        "highres_scale": DEFAULT_HIGHRES_SCALE,
        "num_inference_steps": DEFAULT_INFERENCE_STEPS,
        "highres_steps": DEFAULT_HIGHRES_STEPS,
    }
    actual = {
        "highres_scale": request.highres_scale,
        "num_inference_steps": request.num_inference_steps,
        "highres_steps": request.highres_steps,
    }
    if actual != expected:
        raise HTTPException(status_code=422, detail=f"sampling settings must equal {expected}")
    if request.highres_denoise not in {DEFAULT_HIGHRES_DENOISE, RAIN_HIGHRES_DENOISE}:
        raise HTTPException(
            status_code=422,
            detail=(
                "highres_denoise must follow the pinned AdaptVPR policy: "
                f"{DEFAULT_HIGHRES_DENOISE} default or {RAIN_HIGHRES_DENOISE} rain"
            ),
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        state.pipe_t2i, state.pipe_i2i, state.vae = load_pipeline()
    except Exception as exc:
        state.error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="AdaptVPR IC-Light adapter", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    ready = state.pipe_t2i is not None and state.pipe_i2i is not None and state.vae is not None
    return {
        "status": "ok" if ready else "error",
        "model_loaded": ready,
        "generator_ready": ready,
        "repository": ICLIGHT_REPOSITORY,
        "source_commit": state.source_commit,
        "expected_commit": ICLIGHT_COMMIT,
        "base_model_id": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "sampling": {
            "highres_scale": DEFAULT_HIGHRES_SCALE,
            "highres_denoise": [RAIN_HIGHRES_DENOISE, DEFAULT_HIGHRES_DENOISE],
            "num_inference_steps": DEFAULT_INFERENCE_STEPS,
            "highres_steps": DEFAULT_HIGHRES_STEPS,
            "scheduler": "DDIMScheduler",
        },
        "error": state.error,
    }


@app.post("/generate")
def generate(request: GenerateRequest) -> dict:
    import torch
    from PIL import Image

    if state.pipe_t2i is None or state.pipe_i2i is None or state.vae is None:
        raise HTTPException(status_code=503, detail=state.error or "model is not loaded")
    image_path = Path(request.image_path).resolve()
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail=f"image_path is not readable: {image_path}")
    _validate_sampling(request)
    source = Image.open(image_path).convert("RGB")
    width, height = _valid_size(source)
    negative_prompt = request.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    output_dir = Path(os.getenv("ICLIGHT_OUTPUT_DIR", "/tmp/adaptvpr_iclight")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{uuid.uuid4().hex}.png"

    try:
        with generation_lock, torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
            base_condition = _concat_condition(source, state.vae, width, height)
            lowres = state.pipe_t2i(
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=request.num_inference_steps,
                width=width,
                height=height,
                generator=generator,
                cross_attention_kwargs={"concat_conds": base_condition},
            ).images[0]
            target_width = int(width * request.highres_scale) // 8 * 8
            target_height = int(height * request.highres_scale) // 8 * 8
            highres_condition = _concat_condition(source, state.vae, target_width, target_height)
            result = state.pipe_i2i(
                prompt=request.prompt,
                negative_prompt=negative_prompt,
                image=lowres.resize((target_width, target_height)),
                strength=request.highres_denoise,
                num_inference_steps=max(1, int(request.highres_steps / request.highres_denoise)),
                generator=generator,
                cross_attention_kwargs={"concat_conds": highres_condition},
            ).images[0]
            result.save(result_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    return {"result_path": str(result_path)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("ICLIGHT_PORT", "8002")))
