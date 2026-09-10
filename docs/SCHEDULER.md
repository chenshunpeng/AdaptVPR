# Route scheduler specification

This document specifies the public planning-time route scheduler implemented in
`generation/agent.py`. The released prompt JSONL files already freeze each
sample's route, condition, and prompt. This scheduler is used only when running
`run.py --mode plan` on source images.

## Effective configuration

All configurable values are read from `.env`, have public defaults in
`configs/default.env.example`, and are copied into `experiment.json` under the
`scheduler` key.

| Environment variable | Released default | Meaning |
| --- | ---: | --- |
| `ADAPTVPR_TARGET_ROUTE_RATIOS` | `skip:0.25,global:0.25,local:0.25,dual:0.25` | Dataset-level route targets; values are normalized to sum to one |
| `ADAPTVPR_WEATHER_THRESHOLD` | `0.55` | Minimum weather capability for Global or Dual |
| `ADAPTVPR_OCCLUSION_THRESHOLD` | `0.58` | Minimum occlusion capability for Local or Dual |
| `ADAPTVPR_DEFICIT_WEIGHT` | `1.0` | Base multiplier on route deficit |
| `ADAPTVPR_CAPABILITY_WEIGHT` | `0.25` | Multiplier on route capability |
| `ADAPTVPR_MIN_ROUTE_RATIO` | `0.20` | Ratio below which the deficit multiplier is increased |
| `ADAPTVPR_MAX_ROUTE_RATIO` | `0.30` | Soft upper filter for an eligible route |
| `ADAPTVPR_MIN_ROUTE_DEFICIT_MULTIPLIER` | `2.0` | Deficit multiplier below the minimum ratio |
| `ADAPTVPR_GLOBAL_WEATHER_TARGET_RATIOS` | five weather types at `0.20` each | Global-route weather targets; values are normalized |
| `ADAPTVPR_GLOBAL_SAFE_WEATHERS` | `overcast,fog` | Fallback pool for risky Global scenes |
| `ADAPTVPR_GLOBAL_MAX_WEATHER_RATIO` | `0.50` | Soft cap for one generated Global weather |

## Route eligibility and score

Let `w` be the VLM weather score, `o` the occlusion score, `n_r` the number of
previously scheduled samples on route `r`, and `N = 1 + sum_r(n_r)` the planned
sample count including the current sample.

An image is eligible for Global when `w >= 0.55`, for Local when `o >= 0.58`,
and for Dual when both conditions hold. If the VLM marks the image as bad, it is
sent directly to Skip. If no generation route is eligible, Skip is the only
candidate. Skip does not compete with an eligible generation route merely to
meet its target ratio.

For every route, the deficit is

```text
d_r = target_ratio_r * N - n_r
```

Eligible routes whose current ratio `n_r / N` exceeds the configured maximum
are removed when at least one other eligible route remains. The remaining route
score is

```text
score_r = effective_deficit_weight_r * d_r
          + capability_weight * capability_r
```

`effective_deficit_weight_r` is the base deficit weight, multiplied by the
configured low-ratio multiplier when `n_r / N` is below the minimum. Route
capability is `w` for Global, `o` for Local, and `min(w, o)` for Dual. The route
with the largest score wins. Exact ties follow the stable eligibility order
Global, Local, Dual.

Counts are updated after the post-scheduler safety policies have finalized the
route. Therefore the targets guide the distribution but cannot force a route
that is infeasible for the available images. With a fixed sequence of samples
eligible for all three generation routes and the released defaults, the golden
sequence is `Global, Local, Dual`, repeated.

## Global weather quota

The Global-only scene policy chooses from `overcast`, `fog`, `rain`, `snow`, and
`night`. It first restricts the pool using scene risk and the VLM's
`safe_global_weathers`, applies the per-weather soft cap, and then selects among
the weather types with the largest deficit relative to the configured weather
targets. Exact deficit ties use Python's seeded random choice. `run.py` seeds
that choice from `--seed` and records the seed in `experiment.json`.

Dual weather is finalized by the route scheduler and is not passed through this
Global-only quota policy.

## Ordering, state, batching, and parallel planning

The released scheduler is an online dataset-level quota scheduler. One
`SceneAugmentAgent` owns one set of counters, initialized to zero for each
`run.py` invocation. Planner mode recursively sorts paths lexicographically;
prompt mode preserves JSONL line order. `--limit` is applied after that ordering.

Because route scores use previous counts, changing the planning order or
planning separate subsets with separately initialized agents can change an
individual sample's route. This is expected for the released online algorithm,
not hidden state. The manifest records:

- the effective scheduler parameters and zero-state definition;
- the ordering rule, ordered sample IDs, and their SHA-256 digest;
- the limit, worker count, batching semantics, and planning seed.

To reproduce one planning run, use the same source set, ordered sample IDs,
planner outputs, scheduler configuration, and seed. To parallelize downstream
generation without changing route assignments, first complete one planning run,
freeze its route/condition/prompt records, and shard those frozen records. Do not
run independent online schedulers on each shard and expect the same global
quota trajectory.

The fixed capability-sequence regression in `tests/test_agent_planning.py`
protects the released defaults and the balanced Global/Local/Dual trajectory.
