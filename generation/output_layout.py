"""Output-path policy for accepted and rejected generation candidates."""

from pathlib import Path


def final_output_path(
    output_root: Path,
    stem: str,
    route: str,
    weather: str | None,
    occlusion: str | None,
    *,
    passed: bool,
) -> Path:
    """Keep rejected images outside route directories consumed by training."""
    suffix_parts = [route]
    if weather:
        suffix_parts.append(weather)
    if occlusion:
        suffix_parts.append(occlusion)

    if passed:
        output_dir = output_root / route
        status_suffix = "final"
    else:
        output_dir = output_root / "rejected" / route
        status_suffix = "rejected"

    return output_dir / f"{stem}__{'_'.join(suffix_parts)}__{status_suffix}.jpg"
