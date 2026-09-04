# Released Prompt JSONL input

Use `--mode prompt` to bypass Qwen planning and send a released initial prompt
directly to the selected image generator. Each non-empty JSONL line must contain:

| Field | Type | Meaning |
|---|---|---|
| `sample_id` | string | Unique output and resume identifier |
| `source_id` | string | Source image filename |
| `city` | string | Subdirectory below `--image-root` |
| `route` | string | `global`, `local`, or `dual` |
| `condition` | string | Weather, occlusion, or `weather+occlusion` |
| `prompt` | string | Exact initial generation prompt |

Valid route/condition combinations are:

| Route | Condition |
|---|---|
| `global` | one supported weather |
| `local` | `vehicle` or `person` |
| `dual` | supported weather plus `vehicle` or `person` |

Example:

```json
{"sample_id":"adapt_000001","source_id":"Bangkok_example.jpg","city":"Bangkok","route":"dual","condition":"rain+vehicle","prompt":"Preserve the exact scene geometry. Add realistic rain and exactly one vehicle on the visible traffic lane."}
```

The source is resolved first as `IMAGE_ROOT/CITY/source_id`, then as
`IMAGE_ROOT/source_id`. Duplicate sample IDs, missing images, empty prompts, and
route/condition mismatches are rejected.

With `--reflection off`, AdaptVPR does not initialize or call Qwen. With
`--reflection on`, the initial prompt and seed remain identical; Qwen is called
only when an initial Local/Dual candidate fails verification. Global never
reflects.
