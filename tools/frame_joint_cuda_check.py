"""Targeted real-artifact CUDA checks for Frame Joint v1."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import torch

from molvid.checkpoints import frame_target_provenance, load_codec_artifact
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from molvid.data.store import ClipMMapDataset
from molvid.generation import sample_frame_joint
from molvid.flow.objective import FrameRectifiedFlowObjective
from molvid.geometry.coordinates import center_coordinates
from molvid.latent.statistics import FrameLatentStatistics
from molvid.latent.types import ObservedContext
from molvid.model import FrameJointModel
from molvid.training.batches import prepare_batch_then_to_device, prepare_frame_joint_batch
from molvid.training.dit import module_state_hash
from molvid.training.joint import FrameJointTrainer, JointLossConfig, frame_joint_loss


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--train-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _proper_rotation(device: torch.device) -> torch.Tensor:
    matrix = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    if not torch.allclose(matrix.T @ matrix, torch.eye(3, device=device), atol=1e-6):
        raise RuntimeError("check rotation is not orthogonal")
    return matrix


def _vector_rotate(value: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ic,ij->...jc", value, rotation)


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()


def main() -> int:
    args = _arguments()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Frame Joint checks require an explicit CUDA device")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(20260920)
    torch.cuda.manual_seed_all(20260920)
    store = ClipMMapDataset(args.train_store)
    try:
        cpu_batch = collate_clip_records([store[0]])
        artifact = load_codec_artifact(
            args.codec,
            expected_sha256=args.codec_sha256,
            device=device,
        )
        codec = artifact.model.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)

        # Both modules register the same coordinate-independent topology while
        # the batch is still on CPU.
        codec.prepare_batch(cpu_batch)
        coordinate = prepare_batch_then_to_device(codec, cpu_batch, device)
        centered, origin = center_coordinates(
            coordinate.x,
            frame_mask=coordinate.frame_mask,
            abid=coordinate.abid,
            atom_mask=coordinate.loss_mask,
        )
        direct = codec.frame_encoder(replace(coordinate, x=centered))
        direct_v = direct.v + codec.coordinate_stem(centered).to(direct.v.dtype)

        from molvid.codec.frame import FrozenFrameTeacher

        frozen = FrozenFrameTeacher.from_codec(codec).to(device)
        frozen.prepare_batch(cpu_batch)
        frozen_latent, frozen_origin = frozen(coordinate)
        teacher_h_error = float((frozen_latent.h - direct.h).abs().max())
        teacher_v_error = float((frozen_latent.v - direct_v).abs().max())
        origin_error = float((frozen_origin - origin).abs().max())
        if max(teacher_h_error, teacher_v_error) > 1e-4 or origin_error > 1e-5:
            raise RuntimeError(
                "copied frozen teacher differs from verified codec output: "
                f"h={teacher_h_error}, v={teacher_v_error}, origin={origin_error}"
            )

        statistics = FrameLatentStatistics.fit(
            [frozen_latent],
            provenance={
                "scope": "cuda_check_single_train_clip_only",
                "codec_checkpoint_sha256": artifact.report.source_sha256,
            },
        ).to(device)
        model = FrameJointModel.from_codec(codec, statistics).to(device)
        teacher_hash_before = module_state_hash(model.target_teacher)
        prepared = prepare_frame_joint_batch(
            model.target_teacher,
            cpu_batch,
            device=device,
            normalizer=model,
            history_frames=4,
        )

        # A repeated static trajectory must have no history detail even when
        # its physical timestamps differ.
        observed = prepared.observed_context
        repeated_latent = replace(
            observed.latent,
            h=observed.latent.h[:1].expand(4, -1, -1).clone(),
            v=observed.latent.v[:1].expand(4, -1, -1, -1).clone(),
            time_ps=torch.tensor([[0.0, 100.0, 250.0, 600.0]], device=device),
            frame_mask=torch.ones((1, 4), device=device, dtype=torch.bool),
        )
        repeated_context = ObservedContext(
            latent=repeated_latent,
            coordinates=observed.coordinates[:1].expand(4, -1, -1).clone(),
            sample_origin=observed.sample_origin,
            loss_mask=observed.loss_mask,
        )
        repeated_memory = model.history_encoder(repeated_context)
        static_detail_max = float(torch.maximum(
            repeated_memory.detail_h.abs().max(), repeated_memory.detail_v.abs().max()
        ))
        if static_detail_max > 2e-6:
            raise RuntimeError("repeated static history produced nonzero detail")

        irregular_latent = replace(
            observed.latent,
            time_ps=torch.tensor([[0.0, 70.0, 230.0, 500.0]], device=device),
            frame_mask=torch.tensor([[True, True, True, False]], device=device),
        )
        irregular_context = replace(observed, latent=irregular_latent)
        irregular_memory = model.history_encoder(irregular_context)
        irregular_finite = bool(torch.isfinite(irregular_memory.h).all() and torch.isfinite(irregular_memory.v).all())
        if not irregular_finite:
            raise RuntimeError("irregular masked physical time produced NaN/Inf")

        # T=1 uses the uncompressed history tail and the observed-only decoder.
        one_cpu = slice_clip_frames(cpu_batch, 0, 1)
        model.target_teacher.prepare_batch(one_cpu)
        one_context = model.target_teacher.observed_context(one_cpu.to(device))
        one_memory = model.history_encoder(one_context)
        one_decode = model.decoder(one_context)
        t1_ok = one_memory.tokens == 1 and bool(torch.isfinite(one_decode.coordinates).all())
        if not t1_ok:
            raise RuntimeError("T=1 static path failed")

        # Rotation+translation covariance of the frozen target and full flow
        # velocity.  Row-vector coordinate convention is x' = x R + a.
        rotation = _proper_rotation(torch.device("cpu"))
        translation = torch.tensor([2.0, -3.0, 0.5])
        rotated_cpu = replace(
            cpu_batch,
            x=cpu_batch.x @ rotation + translation,
            bpos=cpu_batch.bpos @ rotation + translation,
        )
        rotated = prepare_frame_joint_batch(
            model.target_teacher,
            rotated_cpu,
            device=device,
            normalizer=model,
            history_frames=4,
        )
        rotation_cuda = rotation.to(device)
        teacher_rotation_h = float((rotated.target_future.h - prepared.target_future.h).abs().max())
        teacher_rotation_v = float((rotated.target_future.v - _vector_rotate(prepared.target_future.v, rotation_cuda)).abs().max())
        flow_time = torch.tensor([0.37], device=device)
        with torch.no_grad():
            base_output = model(
                prepared.normalized_target,
                context=prepared.observed_context,
                query=prepared.query,
                flow_time=flow_time,
            )
            rotated_output = model(
                rotated.normalized_target,
                context=rotated.observed_context,
                query=rotated.query,
                flow_time=flow_time,
            )
        flow_rotation_h = float((rotated_output.velocity.h - base_output.velocity.h).abs().max())
        flow_rotation_v = float((
            rotated_output.velocity.v - _vector_rotate(base_output.velocity.v, rotation_cuda)
        ).abs().max())
        if max(teacher_rotation_h, flow_rotation_h) > 3e-4 or max(teacher_rotation_v, flow_rotation_v) > 7e-4:
            raise RuntimeError(
                "Frame Joint rotation/translation covariance exceeded tolerance: "
                f"teacher_h={teacher_rotation_h}, teacher_v={teacher_rotation_v}, "
                f"flow_h={flow_rotation_h}, flow_v={flow_rotation_v}"
            )

        loss_on = JointLossConfig.resolve({"bond_enabled": True, "generated_bond": 0.01})
        loss_off = JointLossConfig.resolve({
            "bond_enabled": False,
            "generated_bond": 9.0,
            "clean_bond": 9.0,
            "near_bond": 9.0,
        })
        bond_switch_ok = (
            loss_off.generated_bond == 0.0
            and loss_off.clean_bond == 0.0
            and loss_off.near_bond == 0.0
        )
        if not bond_switch_ok:
            raise RuntimeError("bond-off did not resolve every explicit bond weight to zero")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16_decoder = model.decoder(
                prepared.observed_context,
                prepared.target_future,
                prepared.query,
            )
        bf16_decoder_finite = bool(torch.isfinite(bf16_decoder.coordinates).all())
        if not bf16_decoder_finite:
            raise RuntimeError("BF16 decoder produced NaN/Inf")

        torch.cuda.reset_peak_memory_stats(device)
        trainer = FrameJointTrainer(
            model,
            loss_config=loss_on,
            learning_rates={"history_encoder": 1e-4, "dit": 2e-4, "decoder": 5e-5},
            stage="joint",
            amp=True,
            teacher_artifact_sha256=artifact.report.source_sha256,
        )
        generator = torch.Generator(device=device).manual_seed(20260921)
        first_step = trainer.train_step(prepared, generator=generator)
        local_gate_gradient_norms = []
        local_gate_max_abs = []
        for block in model.decoder.blocks:
            gate_gradients = [
                block.local_scalar_gate.grad,
                block.local_vector_gate.grad,
            ]
            local_gate_gradient_norms.append(float(torch.stack([
                gradient.float().square().sum()
                for gradient in gate_gradients
                if gradient is not None
            ]).sum().sqrt()))
            local_gate_max_abs.append(float(torch.maximum(
                block.local_scalar_gate.detach().abs().max(),
                block.local_vector_gate.detach().abs().max(),
            )))
        if any(value <= 0 for value in local_gate_gradient_norms + local_gate_max_abs):
            raise RuntimeError("decoder local residual gates did not start from their first gradient")
        second_step = trainer.train_step(prepared, generator=generator)
        local_message_gradient_norms = []
        for block in model.decoder.blocks:
            gradients = [
                parameter.grad
                for parameter in block.local.parameters()
                if parameter.grad is not None
            ]
            local_message_gradient_norms.append(float(torch.stack([
                gradient.float().square().sum() for gradient in gradients
            ]).sum().sqrt()))
        if any(value <= 0 for value in local_message_gradient_norms):
            raise RuntimeError("decoder local message parameters remained gradient-locked")
        third_step = trainer.train_step(prepared, generator=generator)
        gradient_counts = {
            name: sum(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and bool(torch.any(parameter.grad != 0))
                for parameter in getattr(model, name).parameters()
            )
            for name in ("history_encoder", "dit", "decoder")
        }
        if not all(count > 0 for count in gradient_counts.values()):
            raise RuntimeError(f"trainable component has no finite nonzero gradient after three steps: {gradient_counts}")
        if any(parameter.grad is not None for parameter in model.target_teacher.parameters()):
            raise RuntimeError("frozen target teacher received gradients")
        teacher_hash_after = module_state_hash(model.target_teacher)
        if teacher_hash_after != teacher_hash_before:
            raise RuntimeError("frozen target teacher changed during optimizer steps")
        peak_memory = int(torch.cuda.max_memory_allocated(device))

        # Force the thresholded branches on and compare bond release against
        # literally omitting every explicit bond term from the same graph.
        forced_time = torch.full((prepared.normalized_target.batch_size,), 0.95, device=device)
        attribution_flow = FrameRectifiedFlowObjective()
        attribution_sample = attribution_flow.sample(
            prepared.normalized_target,
            prepared.source_center,
            generator=torch.Generator(device=device).manual_seed(20260922),
            flow_time=forced_time,
        )
        attribution_output = model(
            attribution_sample.interpolated,
            context=prepared.observed_context,
            query=prepared.query,
            flow_time=forced_time,
            clean_future=prepared.target_future,
            decode_generated=True,
            decode_clean=True,
            decode_near=True,
        )
        attribution_rf = attribution_flow.loss(
            attribution_output.velocity, attribution_sample.target_velocity
        )
        on_losses = frame_joint_loss(
            attribution_output,
            attribution_rf,
            prepared,
            forced_time,
            stage="joint",
            config=loss_on,
        )
        off_losses = frame_joint_loss(
            attribution_output,
            attribution_rf,
            prepared,
            forced_time,
            stage="joint",
            config=loss_off,
        )
        no_explicit_bonds = (
            off_losses.raw["flow"]
            + off_losses.raw["clean_coordinate"] * loss_off.clean_coordinate
            + off_losses.raw["near_coordinate"] * loss_off.near_coordinate
        )
        active_parameters = [
            parameter
            for component in (model.history_encoder, model.dit, model.decoder)
            for parameter in component.parameters()
            if parameter.requires_grad
        ]
        release_gradients = torch.autograd.grad(
            off_losses.total, active_parameters, retain_graph=True, allow_unused=True
        )
        removed_gradients = torch.autograd.grad(
            no_explicit_bonds, active_parameters, retain_graph=True, allow_unused=True
        )
        release_gradient_max_abs = 0.0
        for release, removed in zip(release_gradients, removed_gradients):
            if release is None and removed is None:
                continue
            if release is None or removed is None:
                release_gradient_max_abs = float("inf")
                break
            release_gradient_max_abs = max(
                release_gradient_max_abs, float((release - removed).abs().max())
            )
        dit_parameters = [parameter for parameter in model.dit.parameters() if parameter.requires_grad]
        generated_gradients = torch.autograd.grad(
            on_losses.raw["generated_bond"],
            dit_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        generated_bond_dit_gradient_norm = float(torch.stack([
            gradient.float().square().sum()
            for gradient in generated_gradients
            if gradient is not None
        ]).sum().sqrt())
        near_gradients = torch.autograd.grad(
            on_losses.raw["near_coordinate"] + on_losses.raw["near_bond"],
            dit_parameters,
            allow_unused=True,
        )
        near_dit_gradient_norm = float(sum(
            gradient.float().square().sum()
            for gradient in near_gradients
            if gradient is not None
        ).sqrt()) if any(gradient is not None for gradient in near_gradients) else 0.0
        if release_gradient_max_abs != 0.0:
            raise RuntimeError("bond release gradient differs from literal bond-term removal")
        if generated_bond_dit_gradient_norm <= 0:
            raise RuntimeError("generated bond loss has no DiT gradient at s=0.95")
        if near_dit_gradient_norm != 0.0:
            raise RuntimeError("near-endpoint decoder-only branch leaked gradient into DiT")

        # Evaluator truth mutation must not alter observed-only generation.
        prefix = cpu_batch.x[:4].clone()
        prediction_a, _ = sample_frame_joint(
            model,
            template=cpu_batch,
            prefix_coordinates=prefix,
            history_frames=4,
            steps=2,
            seed=11,
        )
        mutated = replace(
            cpu_batch,
            x=torch.cat((cpu_batch.x[:4], cpu_batch.x[4:] + 123.0), dim=0),
            bpos=torch.cat((cpu_batch.bpos[:4], cpu_batch.bpos[4:] - 91.0), dim=0),
        )
        prediction_b, _ = sample_frame_joint(
            model,
            template=mutated,
            prefix_coordinates=prefix,
            history_frames=4,
            steps=2,
            seed=11,
        )
        future_mutation_max_abs = float((prediction_a - prediction_b).abs().max())
        if future_mutation_max_abs != 0.0:
            raise RuntimeError("future target coordinate mutation changed generation")

        checks = {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "sample_id": cpu_batch.sample_id[0],
            "atoms": cpu_batch.atom_count,
            "frames": cpu_batch.frames,
            "teacher_h_max_abs": teacher_h_error,
            "teacher_v_max_abs": teacher_v_error,
            "origin_max_abs": origin_error,
            "static_history_detail_max_abs": static_detail_max,
            "irregular_time_finite": irregular_finite,
            "t1_static_ok": t1_ok,
            "teacher_rotation_h_max_abs": teacher_rotation_h,
            "teacher_rotation_v_max_abs": teacher_rotation_v,
            "flow_rotation_h_max_abs": flow_rotation_h,
            "flow_rotation_v_max_abs": flow_rotation_v,
            "bond_switch_all_explicit_zero": bond_switch_ok,
            "bf16_decoder_finite": bf16_decoder_finite,
            "bf16_joint_steps": 3,
            "decoder_local_gate_gradient_norms_after_first_step": local_gate_gradient_norms,
            "decoder_local_gate_max_abs_after_first_step": local_gate_max_abs,
            "decoder_local_message_gradient_norms_after_second_step": local_message_gradient_norms,
            "gradient_nonzero_parameter_counts": gradient_counts,
            "teacher_state_unchanged": teacher_hash_after == teacher_hash_before,
            "forced_generated_bond_raw": float(on_losses.raw["generated_bond"].detach()),
            "forced_near_bond_raw": float(on_losses.raw["near_bond"].detach()),
            "bond_release_weighted_terms": {
                name: float(off_losses.weighted[name].detach())
                for name in ("generated_bond", "clean_bond", "near_bond")
            },
            "bond_release_removed_gradient_max_abs": release_gradient_max_abs,
            "generated_bond_dit_gradient_norm": generated_bond_dit_gradient_norm,
            "near_endpoint_dit_gradient_norm": near_dit_gradient_norm,
            "future_mutation_max_abs": future_mutation_max_abs,
            "peak_memory_bytes_three_joint_steps": peak_memory,
            "first_step": first_step,
            "second_step": second_step,
            "third_step": third_step,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "cuda_checks.json").write_text(
            json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        provenance = frame_target_provenance(
            artifact,
            source_path=args.codec,
            code_commit=_commit(),
            cuda_check=checks,
        )
        (args.output / "target_encoder_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(checks, sort_keys=True))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
