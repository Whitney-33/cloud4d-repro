#!/usr/bin/env python3
"""Launch the official three-stage Cloud4D training with a fixed global batch."""

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGES = ("pretrain", "finetune", "refine")
STEPS = {"pretrain": 20000, "finetune": 40000, "refine": 30000}
LR = {"pretrain": "1e-4", "finetune": "1e-5", "refine": "1e-5"}
WEIGHT_HASHES = {
    "dinov2_vits14_pretrain.pth":
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "loftup_dinov2s_pytorch_model.bin":
        "715d71f171ecaa557bf2f1b6f317b1285805f738aae84a3cd52b1a2d92f0d867",
}


def check_runtime():
    for package in ("accelerate", "tensorboard", "xformers", "spconv"):
        if importlib.util.find_spec(package) is None:
            raise SystemExit(f"Missing Python package: {package}. Run python -m pip install -r requirements.txt")
    source = ROOT / "vendor" / "dinov2" / "dinov2" / "models" / "vision_transformer.py"
    if not source.is_file():
        raise SystemExit(f"Missing bundled DINOv2 source: {source}")
    for filename, expected in WEIGHT_HASHES.items():
        path = ROOT / "vendor" / "weights" / filename
        if not path.is_file():
            raise SystemExit(f"Missing bundled pretrained weights: {path}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise SystemExit(f"Pretrained weights checksum mismatch: {path}")


def image_dir(data_root, subset, override):
    if override:
        return Path(override).expanduser().resolve()
    return data_root / "synthetic" / subset


def volume_dir(image_path, override):
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        image_path / "volumes",
        image_path / "5km_crop",
        image_path / "volumes" / "5km_crop",
    ):
        if candidate.is_dir():
            return candidate
    return image_path / "volumes"


def check_data(images, volumes, exr_images=False):
    for name in ("perspective_1", "perspective_2", "perspective_3"):
        view = images / name
        for item in ("left_images", "right_images", "left_cam2world.npy",
                     "right_cam2world.npy", "left_intrinsic.npy", "right_intrinsic.npy"):
            if not (view / item).exists():
                raise SystemExit(f"Missing data: {view / item}")
        for side in ("left_images", "right_images"):
            if not any((view / side).iterdir()):
                raise SystemExit(f"No images in {view / side}")
    if not volumes.is_dir() or not any(volumes.glob("*.npz")):
        raise SystemExit(
            f"No .npz volumes in {volumes}. Inspect the extracted archive and "
            "pass --terragen-volumes or --les-volumes if needed."
        )
    if exr_images:
        sample = next((images / "perspective_1" / "left_images").glob("*.exr"), None)
        if sample is None:
            raise SystemExit(f"No EXR images in {images / 'perspective_1' / 'left_images'}")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2
        if cv2.imread(str(sample), cv2.IMREAD_UNCHANGED) is None:
            raise SystemExit(
                f"Cannot decode sample EXR: {sample} (OpenCV {cv2.__version__}). "
                "Install opencv-python==4.11.0.86 or inspect the file."
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--global-batch", type=int, default=4,
                        help="Effective batch over all GPUs and accumulation steps")
    parser.add_argument("--micro-batch", type=int, default=1,
                        help="Per-GPU batch, constrained by GPU memory")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--terragen-images", type=Path)
    parser.add_argument("--terragen-volumes", type=Path)
    parser.add_argument("--les-images", type=Path)
    parser.add_argument("--les-volumes", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default=None,
                        help="TensorBoard run name (default: launch timestamp)")
    parser.add_argument("--smoke-steps", type=int, default=None,
                        help="Run every selected stage for N optimizer steps, with separate checkpoints")
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--finetune-steps", type=int, default=None)
    parser.add_argument("--refine-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without launching or checking data")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate GPUs and extracted data without training")
    args = parser.parse_args()

    if args.gpus < 1 or args.micro_batch < 1 or args.global_batch < 1:
        parser.error("GPU count and batch sizes must be positive")
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error("--smoke-steps must be positive")
    custom_steps = {
        "pretrain": args.pretrain_steps,
        "finetune": args.finetune_steps,
        "refine": args.refine_steps,
    }
    if any(value is not None and value < 1 for value in custom_steps.values()):
        parser.error("Stage step counts must be positive")
    if args.smoke_steps is not None and any(value is not None for value in custom_steps.values()):
        parser.error("--smoke-steps cannot be combined with per-stage step counts")
    steps = {
        phase: args.smoke_steps or custom_steps[phase] or STEPS[phase]
        for phase in STAGES
    }
    per_micro_step = args.gpus * args.micro_batch
    if args.global_batch % per_micro_step:
        parser.error(
            "--global-batch must be divisible by --gpus * --micro-batch; "
            f"received {args.global_batch} and {per_micro_step}"
        )
    accumulation = args.global_batch // per_micro_step
    data_root = args.data_root.expanduser().resolve()
    terragen_images = image_dir(data_root, "terragen", args.terragen_images)
    terragen_volumes = volume_dir(terragen_images, args.terragen_volumes)
    les_images = image_dir(data_root, "large_eddy_simulations", args.les_images)
    les_volumes = volume_dir(les_images, args.les_volumes)
    phases = STAGES if args.phase == "all" else (args.phase,)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    if Path(run_name).name != run_name or run_name in (".", ".."):
        parser.error("--run-name must be one directory name")
    if args.smoke_steps is not None:
        checkpoint_root = ROOT / "checkpoints" / "smoke" / run_name
    elif any(value is not None for value in custom_steps.values()):
        checkpoint_root = ROOT / "checkpoints" / "experiments" / run_name
    else:
        checkpoint_root = ROOT / "checkpoints"

    if not args.dry_run:
        check_runtime()
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("Install dependencies first; see README_复现.md") from exc
        available = torch.cuda.device_count()
        if available < args.gpus:
            raise SystemExit(f"Requested {args.gpus} GPU(s), CUDA sees {available}")

    print(f"GPUs={args.gpus}, per-GPU microbatch={args.micro_batch}, "
          f"accumulation={accumulation}, effective batch={args.global_batch}", flush=True)
    for phase in phases:
        if phase == "pretrain":
            images, volumes = terragen_images, terragen_volumes
            stage, checkpoint = "1", None
            destination = checkpoint_root / "stage1_pretrain"
            extra = ["--pretrain"]
        elif phase == "finetune":
            images, volumes = les_images, les_volumes
            stage = "1"
            checkpoint = checkpoint_root / "stage1_pretrain" / f"stage1_model_{steps['pretrain']}.pth"
            destination = checkpoint_root / "stage1_les"
            extra = []
        else:
            images, volumes = les_images, les_volumes
            stage = "2"
            checkpoint = checkpoint_root / "stage1_les" / f"stage1_model_{steps['finetune']}.pth"
            destination = checkpoint_root / "stage2_les"
            extra = []

        if not args.dry_run:
            check_data(images, volumes, exr_images=(phase != "pretrain"))
            if checkpoint and not checkpoint.is_file() and not (args.check_only and args.phase == "all"):
                raise SystemExit(f"Missing previous-stage checkpoint: {checkpoint}")
            if not args.check_only:
                destination.mkdir(parents=True, exist_ok=True)

        command = (
            [sys.executable, "src/train.py"] if args.gpus == 1 else
            ["accelerate", "launch", "--multi_gpu", "--num_processes", str(args.gpus),
             "--mixed_precision", "no", "src/train.py"]
        )
        command += [
            "--stage", stage, "--lr", LR[phase],
            "--steps", str(steps[phase]),
            "--batch_size", str(args.micro_batch),
            "--gradient_accumulation_steps", str(accumulation),
            "--num-workers", str(args.num_workers),
            "--data_dir", str(images), "--volume_dir", str(volumes),
            "--checkpoint_path", str(destination),
            "--log_dir", str(ROOT / "runs" / run_name / phase),
            "--save-every", str(args.smoke_steps or 1000),
            "--validation_frequency", str(1 if args.smoke_steps else 25),
        ]
        if checkpoint:
            command += ["--stage1_checkpoint", str(checkpoint)]
        command += extra
        print(f"\n[{phase}] {' '.join(command)}", flush=True)
        if not args.dry_run and not args.check_only:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
