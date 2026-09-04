import os
import io
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_START_SCRIPT = "/path/to/LightX2V/start_server.sh"
SERVICE_LOG_DIR = Path(
    os.getenv(
        "ADAPTVPR_SERVICE_LOG_DIR",
        Path(__file__).resolve().parents[1] / "runtime" / "service_logs",
    )
)


def _is_local_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "0.0.0.0"}


def _health_url(api_url: str) -> str:
    parsed = urlparse(api_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _service_ready(session: requests.Session, api_url: str) -> bool:
    try:
        resp = session.get(_health_url(api_url), timeout=2)
        if resp.status_code != 200:
            return False
        data = resp.json()
        model_loaded = data.get("model_loaded") is True
        generator_ready = data.get("generator_ready")
        if generator_ready is None:
            generator_ready = model_loaded
        return data.get("status") == "ok" and model_loaded and generator_ready is True
    except requests.RequestException:
        return False
    except ValueError:
        return False


def _wait_service_ready(session: requests.Session, api_url: str) -> None:
    """Wait for LightX2V to finish loading instead of failing early."""
    if not api_url:
        return
    timeout = float(os.getenv("LIGHTX2V_READY_TIMEOUT", "0"))
    sleep_s = float(os.getenv("LIGHTX2V_READY_SLEEP", "5"))
    start = time.time()
    while True:
        if _service_ready(session, api_url):
            return
        if timeout > 0 and time.time() - start >= timeout:
            raise RuntimeError(f"[Lightx2v] service not ready after {timeout:.0f}s: {_health_url(api_url)}")
        print("[Lightx2v] waiting for model/generator to be ready...", flush=True)
        time.sleep(sleep_s)


def _start_local_service(session: requests.Session, api_url: str) -> None:
    if not api_url or not _is_local_url(api_url):
        return
    if os.getenv("LIGHTX2V_AUTO_START", "1").lower() in {"0", "false", "no"}:
        return
    if _service_ready(session, api_url):
        return

    script = os.getenv("LIGHTX2V_START_SCRIPT", DEFAULT_START_SCRIPT)
    SERVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SERVICE_LOG_DIR / "lightx2v.log"
    print(f"[Lightx2v] Service unavailable; starting automatically: {script}")
    log_file = open(log_path, "ab", buffering=0)
    subprocess.Popen(
        ["/bin/bash", script],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + int(os.getenv("LIGHTX2V_START_TIMEOUT", "180"))
    while time.time() < deadline:
        if _service_ready(session, api_url):
            print(f"[Lightx2v] Service ready: {api_url}")
            return
        time.sleep(2)
    print(f"[Lightx2v] Service startup timed out; log: {log_path}")


def _mock_disabled() -> bool:
    return os.getenv("ADAPTVPR_DISABLE_MOCK", "0").lower() in {"1", "true", "yes"}


class Lightx2vGenerator:

    def __init__(self, api_url: str = None):
        self.api_url = os.getenv("LIGHTX2V_API_URL", "") if api_url is None else api_url
        self._session = requests.Session()
        self._session.trust_env = False
        if self.api_url:
            print(f"[Lightx2v] API mode: {self.api_url}")
            _start_local_service(self._session, self.api_url)
        else:
            if _mock_disabled():
                raise RuntimeError("[Lightx2v] ADAPTVPR_DISABLE_MOCK=1 but LIGHTX2V_API_URL is not configured")
            print("[Lightx2v] No API configured; using mock mode")

    def _save_temp_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(buf.read())
        tmp.close()
        return tmp.name

    def _call_api(self, ref_image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        if not self.api_url:
            if _mock_disabled():
                raise RuntimeError("[Lightx2v] Mock fallback is disabled and no API is configured")
            return None

        temp_path = None
        try:
            temp_path = self._save_temp_image(ref_image)
            payload = {
                "image_path": temp_path,
                "prompt": prompt,
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "seed": kwargs.get("seed", 42),
                "infer_steps": kwargs.get("infer_steps", int(os.getenv("LIGHTX2V_INFER_STEPS", "4"))),
                "guidance_scale": kwargs.get(
                    "guidance_scale", float(os.getenv("LIGHTX2V_GUIDANCE_SCALE", "1.0"))
                ),
            }
            _wait_service_ready(self._session, self.api_url)
            last_error = None
            max_attempts = int(os.getenv("LIGHTX2V_API_RETRIES", "2"))
            request_timeout = float(os.getenv("LIGHTX2V_API_TIMEOUT", "300"))
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = self._session.post(self.api_url, json=payload, timeout=request_timeout)
                    resp.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    body = ""
                    response = getattr(exc, "response", None)
                    if response is not None:
                        body = (response.text or "")[:500]
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"[Lightx2v] API call failed after attempts={max_attempts} "
                            f"status={status} body={body!r}: {exc}"
                        ) from exc
                    print(
                        f"[Lightx2v] API call failed; retrying {attempt}/{max_attempts}: "
                        f"status={status} error={exc}",
                        flush=True,
                    )
                    time.sleep(2)
            else:
                raise RuntimeError(f"[Lightx2v] API call failed: {last_error}")
            data = resp.json()
            result_path = data.get("result_path")
            if not result_path or not os.path.exists(result_path):
                raise RuntimeError(f"The API returned a missing result_path: {result_path}")
            result = Image.open(result_path).convert("RGB")
            if result.size != ref_image.size:
                result = result.resize(ref_image.size, Image.Resampling.LANCZOS)
            return result
        except (requests.RequestException, RuntimeError) as e:
            if _mock_disabled():
                raise RuntimeError(f"[Lightx2v] API call failed and strict mode forbids mock fallback: {e}") from e
            print(f"[Lightx2v] API call failed ({e})")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def generate_local(
        self,
        ref_image: Image.Image,
        prompt: str,
        mask: Image.Image = None,
        seed: int = 42,
        negative_prompt: str = "",
        **kwargs,
    ) -> Image.Image:
        result = self._call_api(ref_image, prompt, seed=seed, negative_prompt=negative_prompt, **kwargs)
        if result is not None:
            return result
        if _mock_disabled():
            raise RuntimeError("[Lightx2v] The API returned no result and strict mode forbids Local mock fallback")
        return self._mock_local(ref_image, prompt, mask)

    def generate_dual(
        self,
        ref_image: Image.Image,
        prompt: str,
        seed: int = 42,
        negative_prompt: str = "",
        **kwargs,
    ) -> Image.Image:
        result = self._call_api(ref_image, prompt, seed=seed, negative_prompt=negative_prompt, **kwargs)
        if result is not None:
            return result
        if _mock_disabled():
            raise RuntimeError("[Lightx2v] The API returned no result and strict mode forbids Dual mock fallback")
        return self._mock_dual(ref_image, prompt)

    def _mock_local(
        self,
        ref_image: Image.Image,
        prompt: str,
        mask: Image.Image = None,
    ) -> Image.Image:
        print(f"[Lightx2v][Mock Local] prompt='{prompt[:50]}...'")
        result = ref_image.copy()
        draw = ImageDraw.Draw(result)
        w, h = result.size
        draw.rectangle(
            [w // 3, h // 3, w * 2 // 3, h * 2 // 3],
            fill=(30, 30, 30)
        )
        return result

    def _mock_dual(self, ref_image: Image.Image, prompt: str) -> Image.Image:
        print(f"[Lightx2v][Mock Dual] prompt='{prompt[:50]}...'")
        import numpy as np
        img_array = np.array(ref_image).astype(np.float32)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)
        img_array = np.clip(img_array * 0.8, 0, 255).astype(np.uint8)
        result = Image.fromarray(img_array)
        draw = ImageDraw.Draw(result)
        w, h = result.size
        draw.rectangle([w // 4, h // 4, w * 3 // 4, h // 2], fill=(20, 20, 20))
        return result


if __name__ == "__main__":
    gen = Lightx2vGenerator()
    img = Image.new("RGB", (512, 512), color=(150, 150, 150))

    local_out = gen.generate_local(img, "A large truck partially blocking the road")
    local_out.save("/tmp/lightx2v_local_test.jpg")
    print("[Lightx2v] Local output saved to /tmp/lightx2v_local_test.jpg")

    dual_out = gen.generate_dual(img, "Heavy rain with a truck blocking the road, wet road surface")
    dual_out.save("/tmp/lightx2v_dual_test.jpg")
    print("[Lightx2v] Dual output saved to /tmp/lightx2v_dual_test.jpg")
