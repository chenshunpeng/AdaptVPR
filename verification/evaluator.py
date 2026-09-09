# evaluator.py - Dual-Trait Verification
# Compute s_geo (geometric consistency) and s_div (diversity score).

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from prompts.rules import normalize_weather


VISMATCH_ROOT = Path(os.getenv("VISMATCH_ROOT", "../vismatch"))
if VISMATCH_ROOT.exists() and str(VISMATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VISMATCH_ROOT))


LOCAL_MIN_GEO = 0.82
LOCAL_MIN_DIV = 0.09
DUAL_MIN_GEO = 0.72
DUAL_MIN_DIV = 0.20
DUAL_RAINY_VEHICLE_RELAXED_MIN_DIV = 0.12

ROUTE_THRESHOLDS = {
    "local": {"TAU_GEO": LOCAL_MIN_GEO, "TAU_DIV": LOCAL_MIN_DIV},
    "global": {"TAU_GEO": 0.78, "TAU_DIV": 0.15},
    "dual": {"TAU_GEO": DUAL_MIN_GEO, "TAU_DIV": DUAL_MIN_DIV},
}

ROUTE_ALIASES = {
    "occlusion": "local",
    "occlusion_only": "local",
    "weather": "global",
    "weather_only": "global",
    "both": "dual",
    "weather_and_occlusion": "dual",
    "skip": "pass",
}


@dataclass
class EvalResult:
    s_geo: float = 1.0
    s_div: float = 1.0
    geo_ok: bool = True
    div_ok: bool = True
    passed: bool = True
    feedback: dict = field(default_factory=dict)
    skipped: bool = False


class DualTraitEvaluator:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        matcher_name: str = "superpoint-lightglue",
        img_size: int = 512,
        n_kpts: int = 2048,
        mock: bool = False,
    ):
        self.mock = mock
        self.matcher_name = os.getenv("ADAPTVPR_MATCHER_NAME", matcher_name)
        self.img_size = img_size
        self.n_kpts = n_kpts
        self.matcher = None
        clip_model_name = os.getenv("ADAPTVPR_CLIP_MODEL_NAME", clip_model_name)

        # Scoped caching state (cleared upon completion or error)
        self._cached_ref_id = None
        self._cached_image0 = None
        self._cached_ref_feat = None

        if mock:
            self.device = "cpu"
            self.torch = None
            self.functional = None
            print("[Evaluator] Mock mode; skipping CLIP and vismatch model loading")
            self.model = None
            self.processor = None
            return

        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError(
                "Real verification requires PyTorch. Install the dependencies in requirements.txt."
            ) from exc

        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.functional = functional
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Evaluator] Loading CLIP: {clip_model_name}")
        local_files_only = os.getenv("ADAPTVPR_CLIP_LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes"}
        self.model = CLIPModel.from_pretrained(
            clip_model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            clip_model_name,
            local_files_only=local_files_only,
        )
        self.model.eval()
        print(f"[Evaluator] CLIP loaded on {self.device}")

    def clear_cache(self) -> None:
        """Clears cached tensors to free VRAM after a sample is completed or on error."""
        self._cached_ref_id = None
        self._cached_image0 = None
        self._cached_ref_feat = None

    def _normalize_route(self, entry: Optional[dict[str, Any]] = None, route: Optional[str] = None) -> str:
        raw_route = route
        if raw_route is None and entry is not None:
            raw_route = entry.get("route")
        if hasattr(raw_route, "value"):
            raw_route = raw_route.value
        route_name = str(raw_route or "dual").lower().strip()
        return ROUTE_ALIASES.get(route_name, route_name)

    def _thresholds_for(self, route: str) -> tuple[float, float]:
        thresholds = ROUTE_THRESHOLDS.get(route, ROUTE_THRESHOLDS["dual"])
        return thresholds["TAU_GEO"], thresholds["TAU_DIV"]

    def _load_matcher(self):
        if self.matcher is None:
            try:
                from vismatch import get_matcher
            except ImportError as exc:
                raise RuntimeError(
                    "Could not import vismatch. Activate its environment with "
                    "`source ../vismatch/.venv/bin/activate` before running the agent."
                ) from exc

            print(f"[Evaluator] Loading matcher: {self.matcher_name} on {self.device}")
            try:
                self.matcher = get_matcher(
                    self.matcher_name,
                    device=self.device,
                    max_num_keypoints=self.n_kpts,
                )
            except Exception as exc:
                fallback_matcher = "sift-nn"
                if self.matcher_name == fallback_matcher:
                    raise
                print(
                    f"[Evaluator] Matcher '{self.matcher_name}' failed: {exc!r}. "
                    f"Falling back to '{fallback_matcher}'."
                )
                self.matcher_name = fallback_matcher
                self.matcher = get_matcher(
                    fallback_matcher,
                    device=self.device,
                    max_num_keypoints=self.n_kpts,
                )
        return self.matcher

    def _save_temp_image(self, image: Image.Image, path: Path) -> None:
        image.convert("RGB").save(path, format="JPEG", quality=95)

    def _compute_s_geo(self, ref_image: Image.Image, gen_image: Image.Image) -> float:
        matcher = self._load_matcher()
        with tempfile.TemporaryDirectory(prefix="adaptvpr_eval_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Retrieve cached reference tensor, or process via tempfile on first pass
            if self._cached_ref_id == id(ref_image) and self._cached_image0 is not None:
                image0 = self._cached_image0
            else:
                ref_path = tmp_path / "ref.jpg"
                self._save_temp_image(ref_image, ref_path)
                image0 = matcher.load_image(ref_path, resize=self.img_size)
                self._cached_image0 = image0
                self._cached_ref_id = id(ref_image)

            # Always process and serialize the new candidate image
            gen_path = tmp_path / "gen.jpg"
            self._save_temp_image(gen_image, gen_path)
            image1 = matcher.load_image(gen_path, resize=self.img_size)
            
            # Retain the public matcher call for compatibility
            result = matcher(image0, image1)

        matched_kpts = result.get("matched_kpts0", [])
        num_matched = len(matched_kpts)
        num_inliers = int(result.get("num_inliers", 0))
        if num_matched <= 0:
            return 0.0
        return max(0.0, min(1.0, num_inliers / num_matched))

    def _extract_clip_feature_batch(self, images: list[Image.Image]) -> Any:
        inputs = self.processor(images=[img.convert("RGB") for img in images], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feat = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            
            # Compatibility fix for newer Hugging Face transformers versions returning objects instead of Tensors
            if not isinstance(feat, self.torch.Tensor):
                if hasattr(feat, "image_embeds"):
                    feat = feat.image_embeds
                elif hasattr(feat, "pooler_output"):
                    feat = feat.pooler_output
                else:
                    feat = feat[0]
                    
        return self.functional.normalize(feat, dim=-1)

    def _compute_s_div(self, ref_image: Image.Image, gen_image: Image.Image) -> float:
        # Utilize batching (size=2) for initial pass, then cache the reference feature
        if self._cached_ref_id == id(ref_image) and self._cached_ref_feat is not None:
            feat_ref = self._cached_ref_feat
            feat_gen = self._extract_clip_feature_batch([gen_image])[0:1] # Retain 2D shape [1, dim]
        else:
            feats = self._extract_clip_feature_batch([ref_image, gen_image])
            feat_ref = feats[0:1]
            feat_gen = feats[1:2]
            self._cached_ref_feat = feat_ref
            self._cached_ref_id = id(ref_image)

        cosine_sim = self.functional.cosine_similarity(feat_ref, feat_gen).item()
        return max(0.0, min(1.0, 1.0 - cosine_sim))

    def evaluate(
        self,
        ref_image: Image.Image,
        gen_image: Image.Image,
        entry: Optional[dict[str, Any]] = None,
        route: Optional[str] = None,
    ) -> EvalResult:
        route_name = self._normalize_route(entry=entry, route=route)
        if route_name == "pass":
            return EvalResult(
                passed=False,
                skipped=True,
                feedback={"status": "Pass route; evaluation skipped."},
            )

        tau_geo, tau_div = self._thresholds_for(route_name)

        if self.mock:
            import random

            s_geo = random.uniform(0.6, 0.9)
            s_div = random.uniform(0.05, 0.35)
        else:
            s_geo = self._compute_s_geo(ref_image, gen_image)
            s_div = self._compute_s_div(ref_image, gen_image)

        weather = normalize_weather((entry or {}).get("weather"))
        occlusion = str((entry or {}).get("occlusion", "") or "").lower()
        if route_name == "dual" and weather in {"rain", "rainy_night"} and occlusion == "vehicle":
            tau_div = DUAL_RAINY_VEHICLE_RELAXED_MIN_DIV
        geo_ok = s_geo >= tau_geo
        div_ok = s_div >= tau_div
        passed = geo_ok and div_ok

        feedback = {}
        if not geo_ok:
            feedback["geo_issue"] = {
                "prompt_instruction": (
                    "Strictly preserve the original road layout, lane markings, building "
                    "outlines, and perspective; do not alter any structural element."
                ),
                "param_adjustment": {"denoising_strength": "decrease by 0.05–0.10"},
            }

        if not div_ok:
            feedback["div_issue"] = {
                "prompt_instruction": (
                    "Strengthen realistic weather effects with more visible but natural "
                    "rain, fog, wet roads, or soft lighting changes, or increase the "
                    "occluder's visual salience with natural scale, clear texture, and "
                    "plausible placement. Do not use black shadows or pasted-looking "
                    "occlusions."
                ),
                "param_adjustment": {"style_strength": "increase by 0.05–0.10"},
            }

        if passed:
            feedback["status"] = "Passed; no revision required."

        return EvalResult(
            s_geo=s_geo,
            s_div=s_div,
            geo_ok=geo_ok,
            div_ok=div_ok,
            passed=passed,
            feedback=feedback,
        )


if __name__ == "__main__":
    evaluator = DualTraitEvaluator(mock=True)
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    result = evaluator.evaluate(img, img, entry={"route": "global"})
    print(result)