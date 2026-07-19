#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEVFORMER = ROOT / "third_party" / "BEVFormer"
REPORTS = ROOT / "reports"

sys.path.insert(0, str(ROOT / "integrations"))
sys.path.insert(0, str(BEVFORMER))

from bevformer_instrumentation import (
    import_bevformer_plugins,
    install_module_stage_hooks,
    install_msda_tensor_capture,
    write_stage_shape_manifest,
)
from bevformer_stage_hooks import BRIDGE, bev_stage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run real BEVFormer inference with optional CUPTI hooks."
    )
    parser.add_argument(
        "--config",
        default=str(
            BEVFORMER / "projects/configs/bevformer/bevformer_tiny.py"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "bevformer_tiny_epoch_24.pth"),
    )
    parser.add_argument(
        "--data-root", default=str(ROOT / "data" / "nuscenes")
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=0)
    parser.add_argument(
        "--output", default=str(REPORTS / "bevformer_outputs.pt")
    )
    parser.add_argument(
        "--stage-shapes", default=str(REPORTS / "stage_shapes.json")
    )
    parser.add_argument(
        "--capture-msda",
        default=str(
            ROOT
            / "benchmarks/MSDA/captured_tensors"
            / "MSDA_sample.pt"
        ),
    )
    parser.add_argument(
        "--capture-stage",
        default="ms_deformable_attention",
        choices=[
            "temporal_self_attention",
            "ms_deformable_attention",
            "detection_head",
        ],
    )
    parser.add_argument("--overwrite-capture", action="store_true")
    parser.add_argument("--skip-msda-capture", action="store_true")
    return parser.parse_args()


def validate_inputs(args):
    required = [
        BEVFORMER,
        Path(args.config),
        Path(args.checkpoint),
        Path(args.data_root) / "v1.0-mini" / "sample.json",
        Path(args.data_root) / "nuscenes_infos_temporal_val.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Missing real BEVFormer inputs:\n  - " + "\n  - ".join(missing)
        )
    if args.samples < 1:
        raise ValueError("--samples must be at least 1")


def set_dataset_root(dataset_cfg, data_root):
    dataset_cfg.data_root = data_root
    ann_file = dataset_cfg.get("ann_file")
    if isinstance(ann_file, str):
        dataset_cfg.ann_file = str(Path(data_root) / Path(ann_file).name)


def rebase_nuscenes_path(path, data_root):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    parts = path.parts
    if "nuscenes" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("nuscenes")
        relative = Path(*parts[index + 1 :])
    else:
        relative = path
    return str(Path(data_root) / relative)


def rebase_dataset_paths(dataset, data_root):
    for info in getattr(dataset, "data_infos", []):
        if info.get("lidar_path"):
            info["lidar_path"] = rebase_nuscenes_path(
                info["lidar_path"], data_root
            )
        for camera in info.get("cams", {}).values():
            if camera.get("data_path"):
                camera["data_path"] = rebase_nuscenes_path(
                    camera["data_path"], data_root
                )
        for sweep in info.get("sweeps", []):
            if sweep.get("data_path"):
                sweep["data_path"] = rebase_nuscenes_path(
                    sweep["data_path"], data_root
                )


def run_real(args):
    validate_inputs(args)
    REPORTS.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stage_shapes).parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from mmcv import Config
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint
        from mmdet.datasets import replace_ImageToTensor
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model
    except Exception as exc:
        raise RuntimeError(
            "The pinned BEVFormer Python environment is not importable. "
            "Run scripts/verify_real_pipeline.py --inputs for details."
        ) from exc

    cfg = Config.fromfile(args.config)
    import_bevformer_plugins(cfg)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.workers_per_gpu = args.workers_per_gpu
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if cfg.get("close_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        set_dataset_root(cfg.data.test, args.data_root)
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline
            )
    else:
        for dataset_cfg in cfg.data.test:
            set_dataset_root(dataset_cfg, args.data_root)
            dataset_cfg.test_mode = True
        samples_per_gpu = max(
            dataset_cfg.pop("samples_per_gpu", 1)
            for dataset_cfg in cfg.data.test
        )
        if samples_per_gpu > 1:
            for dataset_cfg in cfg.data.test:
                dataset_cfg.pipeline = replace_ImageToTensor(
                    dataset_cfg.pipeline
                )

    dataset = build_dataset(cfg.data.test)
    rebase_dataset_paths(dataset, args.data_root)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    wrapped_modules = install_module_stage_hooks(model)
    capture_installed = False
    if not args.skip_msda_capture:
        capture_installed = install_msda_tensor_capture(
            args.capture_msda,
            target_stage=args.capture_stage,
            overwrite=args.overwrite_capture,
        )

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    outputs = []
    data_iterator = iter(data_loader)
    with torch.no_grad():
        for request_id in range(args.samples):
            BRIDGE.begin_request(request_id)
            try:
                with bev_stage("data_preprocess"):
                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        break
                result = model(return_loss=False, rescale=True, **data)
                with bev_stage("result_collection"):
                    outputs.append(result)
            finally:
                BRIDGE.end_request()

    if not outputs:
        raise RuntimeError("The dataset produced no inference samples")

    torch.save(outputs, args.output)
    write_stage_shape_manifest(args.stage_shapes)

    capture_path = Path(args.capture_msda)
    if not args.skip_msda_capture and not capture_path.exists():
        raise RuntimeError(
            "No MSDA call matched capture stage "
            f"{args.capture_stage!r}; capture file was not created"
        )

    return {
        "samples": len(outputs),
        "output": str(Path(args.output)),
        "stage_shapes": str(Path(args.stage_shapes)),
        "wrapped_modules": wrapped_modules,
        "msda_capture_installed": capture_installed,
        "msda_capture_path": (
            None if args.skip_msda_capture else str(capture_path)
        ),
        "msda_capture_stage": (
            None if args.skip_msda_capture else args.capture_stage
        ),
        "workers_per_gpu": args.workers_per_gpu,
        "device": torch.cuda.get_device_name(),
    }


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = run_real(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
