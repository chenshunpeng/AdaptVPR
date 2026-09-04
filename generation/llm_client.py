import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Cloud configuration remains an explicit opt-in.
if os.getenv("ADAPTVPR_ENABLE_CLOUD_LLM", "0").lower() in {"1", "true", "yes"}:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _mock_disabled() -> bool:
    return os.getenv("ADAPTVPR_DISABLE_MOCK", "0").lower() in {"1", "true", "yes"}


def build_llm_client():
    if os.getenv("ADAPTVPR_FORCE_MOCK_LLM", "0").lower() in {"1", "true", "yes"}:
        print("[LLM] Explicit mock client")
        return _MockLLMClient()
    cloud_enabled = os.getenv("ADAPTVPR_ENABLE_CLOUD_LLM", "0").lower() in {
        "1", "true", "yes"
    }
    if cloud_enabled:
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        model_name = os.getenv("OPENAI_MODEL_NAME", "")
    else:
        # Keep planner configuration independent from the legacy generator
        # .env, which may still contain paid/cloud OPENAI_* variables.
        api_key = os.getenv("ADAPTVPR_PLANNER_API_KEY", "local-placeholder")
        api_base = os.getenv(
            "ADAPTVPR_PLANNER_API_BASE",
            "http://127.0.0.1:23002/v1",
        )
        model_name = os.getenv(
            "ADAPTVPR_PLANNER_MODEL",
            "qwen3-vl-4b-instruct-remote",
        )
    timeout = float(os.getenv("OPENAI_TIMEOUT", "20"))
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "0"))

    if api_key and api_key != "sk-xxx":
        try:
            from openai import OpenAI
        except ImportError as exc:
            if _mock_disabled():
                raise RuntimeError(
                    "Real planner execution requires the OpenAI Python client. "
                    "Install the dependencies in requirements.txt."
                ) from exc
            print("[LLM] API key detected but the openai package is unavailable; using the mock client")
            return _MockLLMClient()
        client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )
        print(f"[LLM] Real client: base_url={api_base}, model={model_name}, timeout={timeout}s")
        return _RealLLMClient(client, model_name)

    if _mock_disabled():
        raise RuntimeError(
            "Real planner execution requires ADAPTVPR_PLANNER_API_KEY "
            "or an explicitly enabled cloud API key."
        )
    print("[LLM] No valid API key detected; using the mock client")
    return _MockLLMClient()


class _RealLLMClient:
    def __init__(self, client, model_name: str):
        self._client = client
        self._model = model_name

    def chat(self, system="", user="", json_mode=False, temperature=0.7, model=None):
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content

    def chat_with_images(self, system="", user="", images=None, json_mode=False, temperature=0.3, model=None):
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        content = [{"type": "text", "text": user}]
        if images:
            for b64 in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        resp = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content


class _MockLLMClient:
    def chat(self, system="", user="", json_mode=False, temperature=0.7, model=None):
        if json_mode:
            return json.dumps({
                "concrete_visible_failures": ["the requested appearance change is too weak"],
                "corrective_directives": ["make the authorized transient edit clearly visible"],
            })
        if model == "qwen3-vl-flash":
            return json.dumps({"scene_summary": "mock street scene"})
        return "Heavy rain at night, wet reflective road surface, preserve lane markings and road structure, realistic atmospheric lighting, high detail"

    def chat_with_images(self, system="", user="", images=None, json_mode=False, temperature=0.3, model=None):
        request = f"{system}\n{user}".lower()
        if "augmentation capabilities" in request or "capability scorer" in request:
            return json.dumps({
                "weather_score": 0.9,
                "occlusion_score": 0.9,
                "bad_image": False,
                "weather": "rain",
                "occlusion": "vehicle",
                "position": "center of the visible traffic lane",
                "prompt": "Apply realistic rain and add one vehicle on the visible traffic lane while preserving the exact scene geometry.",
                "reason": "mock scene supports both authorized edits",
                "street_scene_quality": "good",
                "occlusion_feasibility": "high",
                "weather_feasibility": "high",
                "road_visibility": "clear",
                "sky_visibility": "clear",
                "vegetation_level": "low",
                "facade_density": "low",
                "close_building": "no",
                "distant_landmarks_readable": "high",
                "global_weather_risk": "low",
                "safe_global_weathers": ["rain", "overcast"],
            })
        if "permanent scene structure" in request or "permanent-structure" in request:
            return json.dumps({
                "permanent_change_detected": False,
                "permanent_changes": [],
                "confidence": 0.99,
                "reason": "mock candidate preserves permanent structure",
            })
        if "border integrity" in request:
            return json.dumps({
                "accepted": True,
                "cropped": False,
                "edge_attached": False,
                "violations": [],
            })
        if "authorized transient edits" in request:
            count = 2 if "exactly 2" in request else 1
            return json.dumps({"accepted": True, "new_vehicle_count": count, "violations": []})
        if "obviously different" in request:
            return json.dumps({"different": True, "reason": "mock candidates are visibly different"})
        return self.chat(system=system, user=user, json_mode=json_mode, temperature=temperature, model=model)
