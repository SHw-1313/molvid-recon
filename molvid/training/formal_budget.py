"""Durable paired-arm GPU-slot accounting for formal learned-memory runs."""

from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable, Mapping
import uuid

from ..runtime import atomic_write_json, sha256_file


SCHEMA = "molvid.learned_memory.paired_cost_ledger.v1"
CHECKPOINT_SCHEMA = "molvid.learned_memory.formal_budget_checkpoint.v1"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PairedFormalBudget:
    """Append and enforce wall-clock GPU-slot cost across both formal arms."""

    def __init__(
        self,
        *,
        path: Path,
        invocation_id: str,
        pair_contract_hash: str,
        arm: str,
        config_sha256: str,
        cap_gpu_hours: float,
        prior_seconds: float,
        started_unix: float,
        clock: Callable[[], float],
    ) -> None:
        self.path = Path(path).resolve()
        self.invocation_id = str(invocation_id)
        self.pair_contract_hash = str(pair_contract_hash)
        self.arm = str(arm)
        self.config_sha256 = str(config_sha256)
        self.cap_gpu_hours = float(cap_gpu_hours)
        self.prior_seconds = float(prior_seconds)
        self.started_unix = float(started_unix)
        self._clock = clock
        self._finished = False

    @property
    def lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    @classmethod
    def start(
        cls,
        path: Path,
        *,
        pair_contract_hash: str,
        arm: str,
        config_sha256: str,
        code_commit: str,
        cap_gpu_hours: float,
        planned_updates: int,
        resume_checkpoint_sha256: str | None,
        started_unix: float | None = None,
        clock: Callable[[], float] = time.time,
        pid: int | None = None,
        hostname: str | None = None,
    ) -> "PairedFormalBudget":
        if arm not in {"M-short", "M-long"}:
            raise ValueError("formal budget arm is invalid")
        if cap_gpu_hours <= 0 or planned_updates < 1:
            raise ValueError("formal budget cap/planned updates are invalid")
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        now = float(clock())
        started = now if started_unix is None else float(started_unix)
        if started > now + 1e-6:
            raise ValueError("formal invocation start time is in the future")
        local_pid = os.getpid() if pid is None else int(pid)
        local_host = socket.gethostname() if hostname is None else str(hostname)
        invocation_id = uuid.uuid4().hex
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                ledger = json.loads(path.read_text(encoding="utf-8"))
                cls._validate_ledger(
                    ledger,
                    pair_contract_hash=pair_contract_hash,
                    cap_gpu_hours=cap_gpu_hours,
                )
            else:
                ledger = {
                    "schema": SCHEMA,
                    "pair_contract_hash": str(pair_contract_hash),
                    "formal_cap_gpu_hours": float(cap_gpu_hours),
                    "pair_status": "active",
                    "invocations": [],
                }
            for record in ledger["invocations"]:
                if record["status"] != "running":
                    continue
                if (
                    record.get("hostname") != local_host
                    or _process_alive(int(record.get("pid", -1)))
                ):
                    raise RuntimeError("another formal paired-budget invocation is active")
                record.update({
                    "status": "abandoned",
                    "ended_unix": now,
                    "charged_seconds": max(0.0, now - float(record["started_unix"])),
                    "failure": {
                        "type": "AbandonedInvocation",
                        "message": "process ended without finalizing its durable cost record",
                    },
                })
            completed_arms = {
                str(record["arm"])
                for record in ledger["invocations"]
                if record["status"] == "completed"
                and int(record.get("successful_updates", -1))
                == int(record.get("planned_updates", -2))
            }
            if ledger["pair_status"] in {"budget_exhausted", "completed"}:
                raise RuntimeError(f"formal paired budget is already {ledger['pair_status']}")
            if arm == "M-short" and "M-short" in completed_arms:
                raise RuntimeError("formal M-short already completed")
            if arm == "M-long" and "M-short" not in completed_arms:
                raise RuntimeError("formal M-long requires completed M-short")
            if arm == "M-long" and "M-long" in completed_arms:
                raise RuntimeError("formal M-long already completed")
            prior_seconds = sum(
                float(record.get("charged_seconds", 0.0))
                for record in ledger["invocations"]
                if record["status"] != "running"
            )
            if prior_seconds >= float(cap_gpu_hours) * 3600.0:
                ledger["pair_status"] = "budget_exhausted"
                atomic_write_json(path, ledger)
                raise RuntimeError("formal paired GPU-hour cap is already exhausted")
            ledger["invocations"].append({
                "invocation_id": invocation_id,
                "arm": arm,
                "config_sha256": str(config_sha256),
                "code_commit": str(code_commit),
                "resume_checkpoint_sha256": resume_checkpoint_sha256,
                "started_unix": started,
                "ended_unix": None,
                "charged_seconds": 0.0,
                "status": "running",
                "pid": local_pid,
                "hostname": local_host,
                "planned_updates": int(planned_updates),
                "successful_updates": 0,
                "failure": None,
            })
            atomic_write_json(path, ledger)
        return cls(
            path=path,
            invocation_id=invocation_id,
            pair_contract_hash=pair_contract_hash,
            arm=arm,
            config_sha256=config_sha256,
            cap_gpu_hours=cap_gpu_hours,
            prior_seconds=prior_seconds,
            started_unix=started,
            clock=clock,
        )

    @staticmethod
    def _validate_ledger(
        ledger: Mapping[str, Any], *, pair_contract_hash: str, cap_gpu_hours: float
    ) -> None:
        if (
            ledger.get("schema") != SCHEMA
            or ledger.get("pair_contract_hash") != str(pair_contract_hash)
            or float(ledger.get("formal_cap_gpu_hours", -1.0)) != float(cap_gpu_hours)
            or ledger.get("pair_status") not in {"active", "budget_exhausted", "completed"}
            or not isinstance(ledger.get("invocations"), list)
        ):
            raise ValueError("formal paired cost ledger contract differs")
        identifiers: set[str] = set()
        for record in ledger["invocations"]:
            if (
                not isinstance(record, Mapping)
                or record.get("arm") not in {"M-short", "M-long"}
                or record.get("status")
                not in {"running", "completed", "failed_or_interrupted", "abandoned", "budget_exhausted"}
                or not isinstance(record.get("invocation_id"), str)
                or record["invocation_id"] in identifiers
                or float(record.get("charged_seconds", -1.0)) < 0.0
            ):
                raise ValueError("formal paired cost ledger invocation is invalid")
            identifiers.add(record["invocation_id"])

    def elapsed_seconds(self, *, now: float | None = None) -> float:
        value = float(self._clock()) if now is None else float(now)
        return max(0.0, value - self.started_unix)

    def cumulative_seconds(self, *, now: float | None = None) -> float:
        return self.prior_seconds + self.elapsed_seconds(now=now)

    def exhausted(self, *, now: float | None = None) -> bool:
        return self.cumulative_seconds(now=now) >= self.cap_gpu_hours * 3600.0

    def checkpoint_state(self, *, now: float | None = None) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "pair_contract_hash": self.pair_contract_hash,
            "arm": self.arm,
            "config_sha256": self.config_sha256,
            "invocation_id": self.invocation_id,
            "cumulative_gpu_seconds": self.cumulative_seconds(now=now),
            "formal_cap_gpu_hours": self.cap_gpu_hours,
        }

    def validate_resume_state(self, state: Mapping[str, Any]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("schema") != CHECKPOINT_SCHEMA
            or state.get("pair_contract_hash") != self.pair_contract_hash
            or state.get("arm") != self.arm
            or state.get("config_sha256") != self.config_sha256
            or float(state.get("formal_cap_gpu_hours", -1.0)) != self.cap_gpu_hours
        ):
            raise ValueError("resume checkpoint formal budget contract differs")
        prior_id = state.get("invocation_id")
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            ledger = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate_ledger(
                ledger,
                pair_contract_hash=self.pair_contract_hash,
                cap_gpu_hours=self.cap_gpu_hours,
            )
        accounted = 0.0
        found = False
        for record in ledger["invocations"]:
            if record["status"] != "running":
                accounted += float(record["charged_seconds"])
            if record["invocation_id"] == prior_id:
                found = record["status"] != "running"
                break
        if not found or accounted + 1e-6 < float(state.get("cumulative_gpu_seconds", float("inf"))):
            raise ValueError("resume checkpoint GPU cost is not covered by the durable ledger")

    def finish(
        self,
        *,
        status: str,
        successful_updates: int,
        failure: Mapping[str, str] | None,
        now: float | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed_or_interrupted", "budget_exhausted"}:
            raise ValueError("formal budget final status is invalid")
        ended = float(self._clock()) if now is None else float(now)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            ledger = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate_ledger(
                ledger,
                pair_contract_hash=self.pair_contract_hash,
                cap_gpu_hours=self.cap_gpu_hours,
            )
            matches = [
                record for record in ledger["invocations"]
                if record["invocation_id"] == self.invocation_id
            ]
            if len(matches) != 1:
                raise ValueError("formal budget invocation identity is missing or duplicated")
            record = matches[0]
            if record["status"] == "running":
                record.update({
                    "status": status,
                    "ended_unix": ended,
                    "charged_seconds": max(0.0, ended - float(record["started_unix"])),
                    "successful_updates": int(successful_updates),
                    "failure": None if failure is None else dict(failure),
                })
            elif record["status"] != status:
                raise ValueError("formal budget invocation was already finalized differently")
            completed_arms = {
                str(value["arm"])
                for value in ledger["invocations"]
                if value["status"] == "completed"
                and int(value.get("successful_updates", -1))
                == int(value.get("planned_updates", -2))
            }
            cumulative = sum(
                float(value.get("charged_seconds", 0.0))
                for value in ledger["invocations"]
                if value["status"] != "running"
            )
            if status == "budget_exhausted" or cumulative >= self.cap_gpu_hours * 3600.0:
                ledger["pair_status"] = "budget_exhausted"
            elif completed_arms == {"M-short", "M-long"}:
                ledger["pair_status"] = "completed"
            else:
                ledger["pair_status"] = "active"
            atomic_write_json(self.path, ledger)
        self._finished = True
        result = copy.deepcopy(ledger)
        result["cumulative_gpu_hours"] = cumulative / 3600.0
        result["ledger_sha256"] = sha256_file(self.path)
        return result

    @property
    def finished(self) -> bool:
        return self._finished
