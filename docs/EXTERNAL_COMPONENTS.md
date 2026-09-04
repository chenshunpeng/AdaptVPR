# External components

AdaptVPR publishes the orchestration, routing, prompt construction, reflection,
and verification policy. Third-party implementations and model weights are not
vendored in this repository.

| Role | Component used by AdaptVPR | Integration | Public default |
|---|---|---|---|
| Planner and prompt refiner | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | OpenAI-compatible HTTP API | model alias `qwen3-vl-4b-instruct-remote`, API base `http://127.0.0.1:23002/v1` |
| Global generator | [IC-Light](https://github.com/lllyasviel/IC-Light) | Included HTTP adapter | `http://127.0.0.1:8002/generate` |
| Local/Dual generator | `Qwen/Qwen-Image-Edit-2511` + `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`, served by [LightX2V](https://github.com/ModelTC/LightX2V) | Included HTTP adapter | `http://127.0.0.1:8001/generate` |
| Appearance verifier | [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | Loaded in the AdaptVPR process with Transformers | `openai/clip-vit-base-patch32` |
| Geometry verifier | [VisMatch (SuperPoint + LightGlue)](https://github.com/gmberton/vismatch) | Imported as a local Python package | `superpoint-lightglue` |

The Qwen model value is the **served model alias**, not necessarily the model's
download path. Set `ADAPTVPR_PLANNER_MODEL` to the exact model ID returned by
your OpenAI-compatible server.

## Expected installation layout

One convenient local layout is:

```text
workspace/
├── AdaptVPR/
├── IC-Light/
├── LightX2V/
└── vismatch/
```

Paths and endpoints are configured through `.env`; no personal absolute paths
should be committed. Copy the public template first:

```bash
cp configs/default.env.example .env
```

## Planner

Serve Qwen3-VL-4B-Instruct through an OpenAI-compatible endpoint, then
configure:

```env
ADAPTVPR_PLANNER_API_KEY=local-placeholder
ADAPTVPR_PLANNER_API_BASE=http://127.0.0.1:23002/v1
ADAPTVPR_PLANNER_MODEL=qwen3-vl-4b-instruct-remote
```

The planner and the prompt refiner share this endpoint. The server implementation
and Qwen weights are not included.

## Generators

AdaptVPR talks to IC-Light and LightX2V through the small HTTP adapters under
`adapters/`. The upstream projects do not expose this API by default. Clone the
pinned upstream revisions, download the named checkpoints, set their paths in
`.env`, and start both included adapters with
`scripts/start_generation_services.sh`.

The IC-Light adapter pins GitHub commit
`bcf3f29ca85be8a4686215f477b546f5030be8b7`, checkpoint
`lllyasviel/ic-light@9cad1878695f546a7fb9eaca14e2a89131ba5ffe` file
`iclight_sd15_fc.safetensors`, and Stable Diffusion v1.5 revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
The adapters accept either a Git checkout at the pinned commit or the official
GitHub commit tarball with its commit written to `.adaptvpr-source-revision` in
the extracted source root. LightX2V reports tracked source modifications in
`/health` so compatibility patches remain visible rather than being mistaken
for an unmodified upstream checkout.

### Reproducible Qwen-LightX2V adapter

“Qwen-LightX2V” in this release means the following concrete image-to-image
stack, rather than a standalone checkpoint of that name:

- base model: `Qwen/Qwen-Image-Edit-2511`;
- acceleration LoRA repository: `lightx2v/Qwen-Image-Edit-2511-Lightning`;
- LoRA file: `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`;
- LightX2V model class/task: `qwen-image-edit-2511` / `i2i`;
- sampling: 4 inference steps, guidance scale 1.0, adaptive resize, PyTorch SDPA.
- LightX2V GitHub commit: `522609ecc121b49c20d201b3f00c3dc052821bce`.

Download both checkpoints, install LightX2V and the adapter dependencies, then
run the adapter shipped in this repository:

```bash
pip install -v -e /path/to/LightX2V
pip install -r adapters/requirements.txt
export ICLIGHT_ROOT=/path/to/IC-Light
export ICLIGHT_BASE_MODEL_PATH=/models/stable-diffusion-v1-5
export ICLIGHT_MODEL_PATH=/models/iclight_sd15_fc.safetensors
export LIGHTX2V_ROOT=/path/to/LightX2V
export LIGHTX2V_MODEL_PATH=/models/Qwen-Image-Edit-2511
export LIGHTX2V_LORA_PATH=/models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
scripts/start_generation_services.sh
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8001/health
```

The adapter exposes the exact `/health` and `/generate` contract consumed by
`generation/lightx2v.py`. Model loading errors remain visible through `/health`
instead of allowing the orchestration layer to mistake an unloaded service for
a ready generator.

`scripts/start_generation_services.sh` launches the two adapters from this
repository while adding each configured upstream checkout to its Python import
path. It does not expect or execute a third-party `app.py`. If services are
managed outside AdaptVPR, set `ICLIGHT_AUTO_START=0` and
`LIGHTX2V_AUTO_START=0` and provide only their URLs.

## Verifiers

CLIP and the geometry matcher run in the AdaptVPR process rather than through
network ports. CLIP is loaded by Transformers. The geometry package must expose:

```python
from vismatch import get_matcher
matcher = get_matcher(name, device=device, max_num_keypoints=n_kpts)
```

The returned matcher must provide `load_image(...)` and return `matched_kpts0`
and `num_inliers` when called on a pair of images.
