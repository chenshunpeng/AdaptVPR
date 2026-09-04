# Generator service API contracts

IC-Light and LightX2V are integrated as local HTTP services. The current public
client uses shared filesystem paths: the AdaptVPR process and both services must
be able to read and write the same paths. The FastAPI adapters are distributed
in this repository; the upstream source checkouts and model weights are not.

## Health checks

### IC-Light

```http
GET /health
```

Expected response includes `status: "ok"`, `model_loaded: true`, and
`generator_ready: true`. A `200 OK` response from an unloaded adapter is not
treated as ready.

### LightX2V

```http
GET /health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "generator_ready": true
}
```

If `generator_ready` is omitted, it is treated as equal to `model_loaded`.

## IC-Light generation

```http
POST /generate
Content-Type: application/json
```

Request:

```json
{
  "image_path": "/shared/path/input.jpg",
  "prompt": "generation prompt",
  "negative_prompt": "negative prompt",
  "seed": 42,
  "highres_scale": 1.0,
  "highres_denoise": 0.3,
  "num_inference_steps": 20,
  "highres_steps": 20
}
```

The four high-resolution fields are optional.

## LightX2V generation

Local and Dual routes use the same endpoint and are distinguished by the prompt
constructed by AdaptVPR.

```http
POST /generate
Content-Type: application/json
```

Request:

```json
{
  "image_path": "/shared/path/input.jpg",
  "prompt": "generation prompt",
  "negative_prompt": "negative prompt",
  "seed": 42,
  "infer_steps": 4,
  "guidance_scale": 1.0
}
```

The released adapter fixes the public reference implementation to
`Qwen/Qwen-Image-Edit-2511` plus
`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`. It is configured
for four denoising steps and guidance scale 1.
Requests whose `infer_steps` or `guidance_scale` differ from the loaded adapter
configuration are rejected instead of silently changing the reported setup.

## Generation response

Both services return:

```json
{
  "result_path": "/shared/path/generated.jpg"
}
```

`result_path` must exist and be readable by the AdaptVPR process. LightX2V output
is resized to the reference resolution if necessary.

Default request timeout is 300 seconds. Configure it with
`ICLIGHT_API_TIMEOUT` or `LIGHTX2V_API_TIMEOUT`; configure LightX2V retries with
`LIGHTX2V_API_RETRIES`.
