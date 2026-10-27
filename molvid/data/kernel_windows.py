"""Fixed TRAIN-only Q1 windows with explicit history ages and physical clocks.

Preparation selects one plan independently of all model/source RNG streams.
All four kernel arms consume the same plan; changing the arm never changes
the system, replica, current frame, atom order, or multi-frame geometry support.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch

from .batch import collate_clip_records, validate_clip_record
from .store import ClipMMapDataset


MODES = ("B1", "B4", "B4_dense", "B4_lag1")
MODE_ALIASES = {"h1": "B1", "history_mix": "B4", "dense_history": "B4_dense", "lag1": "B4_lag1"}
ATOM_FIELDS = ("atom_identity", "atom_source_index", "atype", "btype", "block_id",
               "component_id", "edge_mask", "loss_mask", "align_mask", "bond_index")


def history_ages(lag: int, kind: str, stage: int = 0) -> list[int]:
    """Oldest-to-newest ages in base delta units, never fabricated frames.

    Rolling starts from [-3,-2,-1,0], appends stage real lag-spaced states,
    and retains the last four. Stage 3 is the stationary coarse history.
    """
    if lag not in (1, 2, 4):
        raise ValueError("kernel lag must be 1, 2, or 4 base intervals")
    if kind == "H1":
        return [0]
    if kind == "dense":
        return [3, 2, 1, 0]
    if kind == "multiscale":
        return [8, 4, 1, 0]
    if kind != "rolling" or stage not in range(4):
        raise ValueError("unknown history recipe or rolling stage")
    times = ([-3, -2, -1, 0] + [lag * s for s in range(1, stage + 1)])[-4:]
    return [stage * lag - t for t in times]


def native_indices(record) -> np.ndarray:
    """Resolve existing ordinary cache IDs using their original builders.

    ATLAS w is a 16-frame window ordinal; MISATO w is the native start index.
    Fixed-history/evaluation records are deliberately not accepted here.
    """
    source = record["source"]
    pattern = (r"atlas_(.+)_(R[123])_dt_(100)ps_w(\d+)" if source == "atlas"
               else r"misato_(.+)_(MD)_dt_(80)ps_w(\d+)")
    match = re.fullmatch(pattern, record["sample_id"])
    if match is None or record["split"] != "train" or len(record["x"]) != 16:
        raise ValueError("expected an ordinary finest-clock 16-frame TRAIN record")
    system, replica, delta, window = match.groups()
    if system != record["system_id"] or replica != record["replica"]:
        raise ValueError("record system/replica differs from its sample identifier")
    delta, window = int(delta), int(window)
    if not np.allclose(np.diff(record["time_ps"]), delta, rtol=0, atol=1e-4):
        raise ValueError("cache timestamps disagree with its finest physical clock")
    stride = 10 if source == "atlas" else 1
    start = window * 16 * stride if source == "atlas" else window
    result = start + np.arange(16, dtype=np.int64) * stride
    if "native_frame_indices" in record and not np.array_equal(result, record["native_frame_indices"]):
        raise ValueError("explicit native indices disagree with ordinary window convention")
    return result


def plan_digest(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_plan(groups, seed: int = 71) -> list[dict]:
    """256 predeclared Q1 visits, uniform system cycles within each domain.

    Integer allocations approximate the requested probabilities: lag128/77/51,
    H1/H4=51/205; within H4 rolling/dense/multiscale=103/51/51. Independent
    shuffles prevent history/lag selection from changing system/window draws.
    """
    data_rng = random.Random(seed + 27001)
    recipe_rng = random.Random(seed + 27002)
    domains = ["protein"] * 128 + ["complex"] * 128
    data_rng.shuffle(domains)
    lags = [1] * 128 + [2] * 77 + [4] * 51
    recipes = [("H1", 0)] * 51 + [("dense", 0)] * 51 + [("multiscale", 0)] * 51
    recipes += [("rolling", i % 4) for i in range(103)]
    recipe_rng.shuffle(lags)
    recipe_rng.shuffle(recipes)
    orders = {}
    rows = []
    for position, (domain, lag, (kind, stage)) in enumerate(zip(domains, lags, recipes)):
        if not orders.get(domain):
            orders[domain] = sorted(groups[domain])
            data_rng.shuffle(orders[domain])
        system = orders[domain].pop()
        bucket = "dt_100ps" if domain == "protein" else "dt_80ps"
        candidates = groups[domain][system][bucket]
        store, index = data_rng.choice(candidates)
        ages = history_ages(lag, kind, stage)
        # The dense-history ablation also needs three earlier real frames.
        lower = max(max(ages), 3)
        upper = 15 - lag
        t0 = data_rng.randint(lower, upper) if lower <= upper else lower
        rows.append(dict(plan_index=position, domain=domain, system=system,
                         store=str(store.root), store_index=int(index), visit=0,
                         history_recipe=kind, rolling_stage=stage, history_ages_delta=ages,
                         planned_lag_delta=lag, t0_index=t0,
                         required_frames=max(16, t0 + lag + 1)))
    return rows


def select_kernel_record(record, row, mode):
    """Select real history and one target without centering/alignment changes."""
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"unsupported kernel mode {mode!r}; choose {MODES}")
    if record["split"] != "train":
        raise ValueError("kernel training accepts TRAIN records only")
    t0, lag = row["t0_index"], row["planned_lag_delta"]
    ages = list(row["history_ages_delta"])
    if mode == "B1":
        ages = [0]
    elif mode == "B4_dense" and len(ages) > 1:
        ages = [3, 2, 1, 0]
    target_lag = 1 if mode == "B4_lag1" else lag
    indices = [t0 - age for age in ages] + [t0 + target_lag]
    if min(indices) < 0 or max(indices) >= len(record["x"]):
        raise ValueError("real source window does not cover requested history/target")
    if any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("history/query indices must be strictly increasing")
    result = dict(record)
    for field in ("x", "bpos", "time_ps"):
        result[field] = np.asarray(record[field])[indices].copy()
    result["delta_time_ps"] = np.diff(result["time_ps"])
    result["frame_mask"] = np.ones(len(indices), dtype=bool)
    result["time_bucket_id"] = "kernel_q1_actual_timestamps"
    result["sample_id"] = f"{row['sample_id']}_kernel{row['plan_index']:03d}_{mode}"
    result["native_frame_indices"] = [row["native_frame_indices"][i] for i in indices]
    result["sampled_delta_time_ps"] = None  # Nonuniform history has no single dt.
    # Retain the original jointly aligned coordinates; do not independently
    # align the selected subset (or align/center protein and ligand separately).
    result.pop("alignment", None)
    result["alignment_provenance"] = "unchanged source window joint alignment"
    identity = {key: deepcopy(row[key]) for key in (
        "plan_index", "domain", "system", "visit", "sample_id", "replica",
        "store", "store_index", "planned_lag_delta", "history_recipe", "rolling_stage",
        "delta_ps", "t0_index", "absolute_t0_ps", "band_source_sample_id")}
    identity.update(task="dynamic", kernel_mode=mode, history_count=len(ages),
                    history_ages_delta=ages, target_lag_delta=target_lag,
                    target_lag_ps=float(target_lag * row["delta_ps"]),
                    clock_sampling="fixed_train_kernel_plan", time_bucket_id=result["time_bucket_id"],
                    native_frame_indices=result["native_frame_indices"],
                    absolute_time_ps=[row["absolute_time_ps"][i] for i in indices],
                    band_frame_count=16)
    return validate_clip_record(result), len(ages), identity


class KernelWindows:
    """Read a prepared plan with an exactly resumable, RNG-free cursor.

    settings must contain ``manifest``. ``data_config`` is checked against
    preparation; it is never used to silently select a different inventory.
    next(mode) -> (ClipBatch[H+1,N], H, JSON identity, FloatTensor[16,N,3]).
    The last tensor is loss-only TRAIN geometry support, never model context.
    """
    def __init__(self, settings, data_config, seed):
        self.path = Path(settings["manifest"])
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("status") != "complete" or self.manifest["seed"] != seed:
            raise ValueError("kernel manifest is incomplete or has a different seed")
        if self.manifest["data_config"] != data_config:
            raise ValueError("kernel preparation inventory differs from training config")
        self.rows = self.manifest["rows"]
        self.digest = plan_digest(self.rows)
        if self.digest != self.manifest["plan_digest"] or len(self.rows) != 256:
            raise ValueError("kernel plan integrity/count mismatch")
        self.cursor, self.stores, self.last_record = 0, {}, None

    def next(self, mode="B4"):
        if self.cursor >= len(self.rows):
            raise StopIteration("all 256 authorized kernel visits were consumed")
        row = self.rows[self.cursor]
        path = row["store"]
        if path not in self.stores:
            self.stores[path] = ClipMMapDataset(path)
        original = self.stores[path][row["store_index"]]
        if original["sample_id"] != row["sample_id"] or original["topology_fingerprint"] != row["topology_fingerprint"]:
            raise ValueError("TRAIN source identity changed since preparation")
        record = original
        if row.get("extended_record"):
            record = torch.load(row["extended_record"], map_location="cpu", weights_only=False)["record"]
        chosen, history, identity = select_kernel_record(record, row, mode)
        self.last_record = chosen
        batch = collate_clip_records([chosen])
        support = torch.from_numpy(np.asarray(original["x"], dtype=np.float32).copy())
        self.cursor += 1
        return batch, history, identity, support

    def state_dict(self):
        return dict(cursor=self.cursor, plan_digest=self.digest)

    def load_state_dict(self, state):
        if state["plan_digest"] != self.digest or not 0 <= int(state["cursor"]) <= len(self.rows):
            raise ValueError("kernel state differs from frozen plan")
        self.cursor = int(state["cursor"])

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()
