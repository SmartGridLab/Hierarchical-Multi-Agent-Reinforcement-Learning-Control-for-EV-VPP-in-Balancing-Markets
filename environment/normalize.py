"""
Observation normalization helpers.

This module converts raw EVEnv observations into the scale expected by the
actor/critic networks, and provides small inverse helpers for diagnostics.

Input:
- `normalize_observation(obs)` receives a station-major observation with shape
  roughly `(num_stations, state_dim)`. `obs` may be a NumPy array from EVEnv or
  a torch Tensor from replay/evaluation utilities.
- Per-station layout is:
  `[EV slot 1 features, ..., EV slot N features][demand lookahead][current step]`

Output:
- `normalize_observation()` returns a torch Tensor with the same shape as input.
- The function clones the input tensor before editing, so callers keep their raw
  observation unchanged.

Normalization rules:
- `presence`: unchanged. Usually 0/1.
- `soc`: raw percent SoC `[0, 100]` -> `[0, 1]`.
- `remaining_time`: raw remaining episode steps `[0, EPISODE_STEPS]` -> `[0, 1]`.
- `needed_soc`: raw percent `[0, 100]` -> `[0, 1]`, then clamped to `[-1, 1]`.
- `switch_count`: raw count -> `[0, 1]` by dividing by `MAX_SWITCH_COUNT`.
- `last_direction`: clamped to `[-1, 1]`; expected values are charge/discharge
  direction-like indicators.
- `demand lookahead`: centered by `(TARGET_MIN + TARGET_MAX) / 2` and divided by
  `(TARGET_MAX - TARGET_MIN) / 2`, then clamped to `[-1, 1]`.
- `current step`: raw step `[0, EPISODE_STEPS]` -> `[0, 1]`.

Dependencies / layout contract:
- `environment.observation_config` defines EV feature order and tail layout.
  If `EV_FEATURE_NAMES`, `LOCAL_DEMAND_STEPS`, or `LOCAL_USE_STEP` changes,
  this file must remain consistent with that layout.
- `Config` supplies episode length, max EV slots per station, and switch-count
  scale.
- `environment.readcsv` supplies demand-adjustment min/max used for AG request
  scaling.

Notes:
- `denormalize_observation()` is a lightweight debug/plotting helper for the
  first station only. Training should use normalized tensors directly.
- Unknown EV features are passed through unchanged so adding already-normalized
  features does not require extra code here.
"""

import numpy as np
import torch

from Config import EPISODE_STEPS, MAX_EV_PER_STATION, MAX_SWITCH_COUNT
from environment.readcsv import TARGET_MIN, TARGET_MAX
from environment.observation_config import (
    EV_FEAT_DIM,
    EV_FEATURE_NAMES,
    LOCAL_DEMAND_STEPS,
    LOCAL_USE_STEP,
)


AG_REQUEST_CENTER = float((TARGET_MIN + TARGET_MAX) / 2.0)
AG_REQUEST_SCALE = float((TARGET_MAX - TARGET_MIN) / 2.0)


def _to_tensor(obs):
    # EVEnv usually emits NumPy arrays, while replay/test paths may already use
    # tensors. The rest of this module uses torch indexing and torch.clamp.
    if isinstance(obs, np.ndarray):
        return torch.from_numpy(obs).float()
    return obs


def _feature_index(name):
    # Optional lookup: diagnostic code can keep working even if a feature is
    # removed from EV_FEATURE_NAMES.
    try:
        return EV_FEATURE_NAMES.index(name)
    except ValueError:
        return None


def _normalize_ev_feature(name, value):
    # Each EV slot column has a different physical unit, so normalization is
    # feature-name based rather than position-only.
    if name == "presence":
        return value
    if name == "soc":
        return value / 100.0
    if name == "remaining_time":
        return value / EPISODE_STEPS
    if name == "needed_soc":
        return torch.clamp(value / 100.0, -1.0, 1.0)
    if name == "switch_count":
        return torch.clamp(value / max(float(MAX_SWITCH_COUNT), 1.0), 0.0, 1.0)
    if name == "last_direction":
        return torch.clamp(value, -1.0, 1.0)
    return value


def _denormalize_ev_feature(name, value):
    # Inverse mapping used for readable debug outputs. Features that are already
    # unitless or direction-like are returned unchanged.
    if name == "soc":
        return value * 100.0
    if name == "remaining_time":
        return value * EPISODE_STEPS
    if name == "needed_soc":
        return value * 100.0
    if name == "switch_count":
        return value * max(float(MAX_SWITCH_COUNT), 1.0)
    return value


def normalize_observation(obs):
    """
    Normalize all stations in a raw observation.

    The returned tensor keeps the original station/feature layout; only numeric
    scale changes. This is the main entry point used by training and evaluation.
    """
    obs = _to_tensor(obs)
    normalized_obs = obs.clone()
    ev_block_dim = EV_FEAT_DIM * MAX_EV_PER_STATION

    for station_idx in range(obs.shape[0]):
        state_dim = obs[station_idx].shape[-1]
        ev_block_end = min(ev_block_dim, state_dim)
        num_evs = ev_block_end // EV_FEAT_DIM

        for ev_idx in range(num_evs):
            base = ev_idx * EV_FEAT_DIM
            for feat_offset, feat_name in enumerate(EV_FEATURE_NAMES):
                idx = base + feat_offset
                if idx < ev_block_end:
                    normalized_obs[station_idx, idx] = _normalize_ev_feature(
                        feat_name, obs[station_idx, idx]
                    )

        tail_idx = ev_block_end
        if LOCAL_DEMAND_STEPS > 0 and tail_idx < state_dim:
            # The demand lookahead tail is normalized against the full observed
            # AG-request range, so positive and negative requests are balanced
            # around zero for the networks.
            lookahead = int(LOCAL_DEMAND_STEPS)
            end_ag = min(tail_idx + lookahead, state_dim - (1 if LOCAL_USE_STEP else 0))
            if end_ag > tail_idx:
                ag_slice = slice(tail_idx, end_ag)
                normalized_obs[station_idx, ag_slice] = torch.clamp(
                    (obs[station_idx, ag_slice] - AG_REQUEST_CENTER) / AG_REQUEST_SCALE,
                    -1.0,
                    1.0,
                )
                tail_idx = end_ag

        if LOCAL_USE_STEP and tail_idx < state_dim:
            # Current time is represented as episode progress. Clamp prevents a
            # malformed step index from leaking out-of-range values.
            normalized_obs[station_idx, tail_idx] = torch.clamp(
                obs[station_idx, tail_idx] / EPISODE_STEPS, 0.0, 1.0
            )

    return normalized_obs


def denormalize_observation(normalized_obs, archive_dir=None):
    """
    Convert station 1 from normalized tensor values back to readable units.

    This function is intentionally partial: it supports debug summaries and
    visualizers, not reconstruction of the full multi-station environment state.
    """
    del archive_dir
    normalized_obs = _to_tensor(normalized_obs)

    station_idx = 0
    state_dim = normalized_obs[station_idx].shape[-1]
    ev_block_dim = EV_FEAT_DIM * MAX_EV_PER_STATION
    ev_block_end = min(ev_block_dim, state_dim)
    num_evs = ev_block_end // EV_FEAT_DIM

    presence_idx = _feature_index("presence")
    soc_idx = _feature_index("soc")
    remaining_idx = _feature_index("remaining_time")
    needed_idx = _feature_index("needed_soc")

    station_data = {"station_id": station_idx + 1, "evs": [], "ag_requests": {}}

    for ev_idx in range(num_evs):
        base = ev_idx * EV_FEAT_DIM
        presence = 1.0
        if presence_idx is not None and base + presence_idx < ev_block_end:
            presence = normalized_obs[station_idx, base + presence_idx].item()
        if presence < 0.5:
            # Padded empty EV slots are omitted from debug output.
            continue

        ev_data = {"ev_index": ev_idx + 1}
        if soc_idx is not None:
            ev_data["soc"] = _denormalize_ev_feature(
                "soc", normalized_obs[station_idx, base + soc_idx]
            ).item()
        if remaining_idx is not None:
            ev_data["remaining_time"] = _denormalize_ev_feature(
                "remaining_time", normalized_obs[station_idx, base + remaining_idx]
            ).item()
        if needed_idx is not None:
            ev_data["needed_soc"] = _denormalize_ev_feature(
                "needed_soc", normalized_obs[station_idx, base + needed_idx]
            ).item()
        station_data["evs"].append(ev_data)

    tail_idx = ev_block_end
    ag_list = []
    if LOCAL_DEMAND_STEPS > 0 and tail_idx < state_dim:
        lookahead = int(LOCAL_DEMAND_STEPS)
        end_ag = min(tail_idx + lookahead, state_dim - (1 if LOCAL_USE_STEP else 0))
        if end_ag > tail_idx:
            ag_norm = normalized_obs[station_idx, tail_idx:end_ag]
            ag_list = (ag_norm * AG_REQUEST_SCALE + AG_REQUEST_CENTER).cpu().tolist()
            tail_idx = end_ag

    step_val = 0.0
    if LOCAL_USE_STEP and tail_idx < state_dim:
        step_val = normalized_obs[station_idx, tail_idx].item() * EPISODE_STEPS

    if ag_list:
        station_data["ag_requests"] = {
            "lookahead": ag_list,
            "current_step": step_val,
        }

    return {"station_1": station_data}


def denormalize_soc(normalized_soc):
    return normalized_soc * 100.0


def normalize_soc(raw_soc):
    return raw_soc / 100.0


def denormalize_remaining_time(normalized_time):
    return normalized_time * EPISODE_STEPS


def normalize_remaining_time(raw_time):
    return raw_time / EPISODE_STEPS


def denormalize_ag_request(normalized_ag):
    return normalized_ag * AG_REQUEST_SCALE + AG_REQUEST_CENTER


def normalize_ag_request(raw_ag):
    return torch.clamp((raw_ag - AG_REQUEST_CENTER) / AG_REQUEST_SCALE, -1.0, 1.0)
