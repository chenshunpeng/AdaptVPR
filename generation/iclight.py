import os
import io
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_START_SCRIPT = "/path/to/IC-Light/start_server.sh"
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
        return (
            data.get("status") == "ok"
            and data.get("model_loaded") is True
            and data.get("generator_ready", data.get("model_loaded")) is True
        )
    except (requests.RequestException, ValueError):
        return False


def _start_local_service(session: requests.Session, api_url: str) -> None:
    if not api_url or not _is_local_url(api_url):
        return
    if os.getenv("ICLIGHT_AUTO_START", "1").lower() in {"0", "false", "no"}:
        return
    if _service_ready(session, api_url):
        return

    script = os.getenv("ICLIGHT_START_SCRIPT", DEFAULT_START_SCRIPT)
    SERVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SERVICE_LOG_DIR / "iclight.log"
    print(f"[ICLight] Service unavailable; starting automatically: {script}")
    log_file = open(log_path, "ab", buffering=0)
    subprocess.Popen(
        ["/bin/bash", script],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + int(os.getenv("ICLIGHT_START_TIMEOUT", "120"))
    while time.time() < deadline:
        if _service_ready(session, api_url):
            print(f"[ICLight] Service ready: {api_url}")
            return
        time.sleep(2)
    print(f"[ICLight] Service startup timed out; log: {log_path}")


def _mock_disabled() -> bool:
    return os.getenv("ADAPTVPR_DISABLE_MOCK", "0").lower() in {"1", "true", "yes"}


class ICLightGenerator:

    def __init__(self, api_url: str = None):
        self.api_url = os.getenv("ICLIGHT_API_URL", "") if api_url is None else api_url
        self._session = requests.Session()
        self._session.trust_env = False
        if self.api_url:
            print(f"[ICLight] API mode: {self.api_url}")
            _start_local_service(self._session, self.api_url)
        else:
            if _mock_disabled():
                raise RuntimeError("[ICLight] ADAPTVPR_DISABLE_MOCK=1 but ICLIGHT_API_URL is not configured")
            print("[ICLight] No API configured; using mock mode")

    def _save_temp_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        suffix = ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(buf.read())
        tmp.close()
        return tmp.name

    def generate(
        self,
        ref_image: Image.Image,
        prompt: str,
        **kwargs,
    ) -> Image.Image:
        if not self.api_url:
            if _mock_disabled():
                raise RuntimeError("[ICLight] Mock fallback is disabled and no API is configured")
            return self._mock_generate(ref_image, prompt)

        temp_path = None
        try:
            temp_path = self._save_temp_image(ref_image)
            payload = {
                "image_path": temp_path,
                "prompt": prompt,
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "seed": kwargs.get("seed", 42),
            }
            for key in [
                "highres_scale",
                "highres_denoise",
                "num_inference_steps",
                "highres_steps",
            ]:
                if key in kwargs:
                    payload[key] = kwargs[key]
            request_timeout = float(os.getenv("ICLIGHT_API_TIMEOUT", "300"))
            resp = self._session.post(self.api_url, json=payload, timeout=request_timeout)
            resp.raise_for_status()
            data = resp.json()
            result_path = data.get("result_path")
            if not result_path or not os.path.exists(result_path):
                raise RuntimeError(f"The API returned a missing result_path: {result_path}")
            return Image.open(result_path).convert("RGB")
        except requests.RequestException as e:
            if _mock_disabled():
                raise RuntimeError(f"[ICLight] API call failed and strict mode forbids mock fallback: {e}") from e
            print(f"[ICLight] API call failed ({e}); falling back to mock mode")
            return self._mock_generate(ref_image, prompt)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _mock_generate(self, ref_image: Image.Image, prompt: str) -> Image.Image:
        print(f"[ICLight][Mock] prompt='{prompt[:50]}...'")
        import numpy as np
        img_array = np.array(ref_image).astype(np.float32)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)
        img_array = np.clip(img_array * 0.85, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


if __name__ == "__main__":
    gen = ICLightGenerator()
    img = Image.new("RGB", (512, 512), color=(120, 120, 120))
    result = gen.generate(img, "Heavy rain, wet road, dark sky, realistic lighting")
    print(f"[ICLight] Output size: {result.size}")
    result.save("/tmp/iclight_test.jpg")
    print("[ICLight] Saved to /tmp/iclight_test.jpg")
