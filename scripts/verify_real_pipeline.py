#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = (
    ROOT
    / "benchmarks/MSDA/captured_tensors"
    / "MSDA_sample.pt"
)
REPORTS = ROOT / "reports"
REQUIRED_METRICS = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.avg.per_cycle_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
}


class Audit:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail):
        self.rows.append((name, bool(ok), str(detail)))

    def finish(self):
        for name, ok, detail in self.rows:
            print(f"{'PASS' if ok else 'MISS':4} {name}: {detail}")
        missing = [name for name, ok, _ in self.rows if not ok]
        if missing:
            print("\nMissing checks:")
            for name in missing:
                print(f"  - {name}")
            raise SystemExit(1)
        print(f"\nAll {len(self.rows)} checks passed.")


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def check_inputs(audit):
    data_root = ROOT / "data/nuscenes"
    required_paths = [
        ROOT / "third_party/BEVFormer/projects/configs/bevformer/bevformer_tiny.py",
        ROOT / "third_party/mmdetection3d-0.17.1/mmdet3d/__init__.py",
        ROOT / "checkpoints/bevformer_tiny_epoch_24.pth",
        data_root / "v1.0-mini/sample.json",
        data_root / "nuscenes_infos_temporal_train.pkl",
        data_root / "nuscenes_infos_temporal_val.pkl",
        ROOT / "data/can_bus",
    ]
    for path in required_paths:
        ok = path.is_dir() if path.suffix == "" else nonempty(path)
        audit.add(f"input:{path.relative_to(ROOT)}", ok, path)

    cameras = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    for camera in cameras:
        directory = data_root / "samples" / camera
        audit.add(
            f"camera:{camera}",
            directory.is_dir() and any(directory.iterdir()),
            directory,
        )

    obsolete = [
        ROOT / "profilers/range_profiler",
        ROOT / "scripts/run_range_profile.sh",
        ROOT / "scripts/generate_sample_artifacts.py",
        ROOT / "scripts/run_cupti_range_nvpw_verifier.sh",
        ROOT / "integrations/mmdet3d_minimal_ops_compat.py",
    ]
    audit.add(
        "source:no_obsolete_paths",
        not any(path.exists() for path in obsolete),
        "legacy derived-range, sample, verifier, and dynamic-op shim removed",
    )

    sys.path.insert(0, str(ROOT / "third_party/BEVFormer"))
    try:
        import torch
        import mmcv
        import mmdet
        import mmdet3d
        import mmseg
        from mmcv.ops.multi_scale_deform_attn import (
            MultiScaleDeformableAttnFunction,
        )
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model
        import projects.mmdet3d_plugin

        _ = (
            MultiScaleDeformableAttnFunction,
            build_dataset,
            build_model,
            projects.mmdet3d_plugin,
        )
        versions = (
            f"torch={torch.__version__}, mmcv={mmcv.__version__}, "
            f"mmdet={mmdet.__version__}, mmseg={mmseg.__version__}, "
            f"mmdet3d={mmdet3d.__version__}"
        )
        audit.add("python:imports", True, versions)
        audit.add(
            "cuda:available",
            torch.cuda.is_available(),
            torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        )
    except Exception as exc:
        audit.add("python:imports", False, repr(exc))


def check_outputs(audit):
    required = [
        REPORTS / "bevformer_outputs.pt",
        REPORTS / "stage_shapes.json",
        REPORTS / "activity_timeline.csv",
        REPORTS / "activity_timeline.json",
        REPORTS / "msda_latency_sweep.csv",
        REPORTS / "range_metrics_msda_nvpw.csv",
        REPORTS / "range_metrics_msda_nvpw.json",
        REPORTS / "msda_nvpw_run.json",
        CAPTURE,
    ]
    for path in required:
        audit.add(
            f"output:{path.relative_to(ROOT)}",
            nonempty(path),
            path,
        )
    if not all(nonempty(path) for path in required):
        return

    import torch

    capture = torch.load(CAPTURE, map_location="cpu")
    audit.add(
        "capture:real_spatial_msda",
        capture.get("capture_source") == "real_bevformer_forward"
        and capture.get("capture_stage") == "ms_deformable_attention",
        {
            "source": capture.get("capture_source"),
            "stage": capture.get("capture_stage"),
            "value": tuple(capture["value"].shape),
            "sampling": tuple(capture["sampling_locations"].shape),
        },
    )

    with (REPORTS / "activity_timeline.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        activity = list(csv.DictReader(handle))
    required_fields = {
        "cpu_launch_us",
        "cpu_launch_end_us",
        "cpu_launch_overhead_us",
        "gpu_start_us",
        "gpu_end_us",
        "gpu_duration_us",
        "scheduling_delay_us",
        "launch_to_start_us",
        "correlation_id",
    }
    audit.add(
        "activity:real_rows_and_fields",
        bool(activity) and required_fields.issubset(activity[0]),
        f"rows={len(activity)}",
    )

    with (REPORTS / "range_metrics_msda_nvpw.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        ranges = list(csv.DictReader(handle))
    metrics = {
        row.get("metric_name")
        for row in ranges
        if row.get("metric_source") == "cupti_range_nvpw_msda_replay"
    }
    audit.add(
        "nvpw:five_real_metrics",
        REQUIRED_METRICS.issubset(metrics),
        sorted(metrics),
    )

    with (REPORTS / "msda_latency_sweep.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        benchmark = list(csv.DictReader(handle))
    audit.add(
        "benchmark:captured_real",
        any(
            row.get("sample_source") == "captured_real"
            and row.get("capture_stage") == "ms_deformable_attention"
            for row in benchmark
        ),
        f"rows={len(benchmark)}",
    )

    manifest = json.loads(
        (REPORTS / "msda_nvpw_run.json").read_text(encoding="utf-8")
    )
    audit.add(
        "nvpw:manifest",
        manifest.get("capture_stage") == "ms_deformable_attention"
        and manifest.get("passes", 0) > 0,
        {
            "passes": manifest.get("passes"),
            "device": manifest.get("device"),
            "capture_stage": manifest.get("capture_stage"),
        },
    )


def check_clean_state(audit):
    build_dirs = [
        ROOT / "profilers/activity_profiler/build",
        ROOT / "profilers/msda_nvpw_replay/build",
    ]
    generated_reports = [
        path
        for path in REPORTS.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    generated_capture = [
        path
        for path in CAPTURE.parent.glob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    caches = list(ROOT.rglob("__pycache__")) + list(ROOT.rglob("*.pyc"))
    audit.add("clean:no_build_dirs", not any(path.exists() for path in build_dirs), build_dirs)
    audit.add("clean:no_generated_reports", not generated_reports, generated_reports)
    audit.add("clean:no_captured_tensors", not generated_capture, generated_capture)
    audit.add("clean:no_python_cache", not caches, f"count={len(caches)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", action="store_true")
    parser.add_argument("--outputs", action="store_true")
    parser.add_argument("--clean-state", action="store_true")
    args = parser.parse_args()
    if not (args.inputs or args.outputs or args.clean_state):
        args.inputs = True

    audit = Audit()
    if args.inputs:
        check_inputs(audit)
    if args.outputs:
        check_outputs(audit)
    if args.clean_state:
        check_clean_state(audit)
    audit.finish()


if __name__ == "__main__":
    main()
