#!/usr/bin/env python3
import argparse
import ctypes
import datetime
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ",".join(
    [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__inst_executed.avg.per_cycle_active",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Replay captured MSDA under custom CUPTI/NVPW user-range profiler.")
    parser.add_argument("--input", default=str(ROOT / "benchmarks/MSDA/captured_tensors/MSDA_sample.pt"))
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--csv", default=str(ROOT / "reports/range_metrics_msda_nvpw.csv"))
    parser.add_argument("--json", default=str(ROOT / "reports/range_metrics_msda_nvpw.json"))
    parser.add_argument("--run-manifest", default=str(ROOT / "reports/msda_nvpw_run.json"))
    parser.add_argument(
        "--build-dir",
        default=str(ROOT / "profilers/msda_nvpw_replay/build"),
    )
    parser.add_argument("--max-passes", type=int, default=16)
    parser.add_argument("--build", action="store_true")
    return parser.parse_args()


def build_library(build_dir):
    build_dir = Path(build_dir)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT / "profilers/msda_nvpw_replay"),
            "-B",
            str(build_dir),
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "-j"], cwd=str(ROOT), check=True)


def import_op():
    import mmcv.ops.multi_scale_deform_attn as mod

    return mod.MultiScaleDeformableAttnFunction.apply


def load_case(path):
    import torch

    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"Captured real MSDA tensor is missing: {path}. "
            "Run scripts/capture_real_msda_sample.sh first."
        )
    captured = torch.load(path, map_location="cuda")
    if captured.get("capture_source") != "real_bevformer_forward":
        raise RuntimeError("NVPW replay input is not a real BEVFormer capture")
    if captured.get("capture_stage") != "ms_deformable_attention":
        raise RuntimeError(
            "NVPW replay requires the spatial MSDA capture, got "
            f"{captured.get('capture_stage')!r}"
        )
    case = {
        "value": captured["value"].contiguous(),
        "spatial_shapes": captured["spatial_shapes"].contiguous(),
        "level_start_index": captured["level_start_index"].contiguous(),
        "sampling_locations": captured["sampling_locations"].contiguous(),
        "attention_weights": captured["attention_weights"].contiguous(),
        "im2col_step": int(captured.get("im2col_step", 64)),
        "capture_source": captured["capture_source"],
        "capture_stage": captured["capture_stage"],
        "capture_call_index": captured.get("capture_call_index"),
    }
    return case


def configure_lib(lib_path):
    lib = ctypes.CDLL(str(lib_path))
    lib.bev_msda_nvpw_last_error.restype = ctypes.c_char_p
    lib.bev_msda_nvpw_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.bev_msda_nvpw_init.restype = ctypes.c_int
    for name in [
        "bev_msda_nvpw_begin_pass",
        "bev_msda_nvpw_push_range",
        "bev_msda_nvpw_pop_range",
        "bev_msda_nvpw_end_pass",
        "bev_msda_nvpw_finalize",
    ]:
        getattr(lib, name).restype = ctypes.c_int
    lib.bev_msda_nvpw_push_range.argtypes = [ctypes.c_char_p]
    return lib


def check(lib, ok, action):
    if ok:
        return
    err = lib.bev_msda_nvpw_last_error()
    msg = err.decode("utf-8", errors="replace") if err else "unknown"
    raise RuntimeError(f"{action} failed: {msg}")


def call_op(op, case):
    return op(
        case["value"],
        case["spatial_shapes"],
        case["level_start_index"],
        case["sampling_locations"],
        case["attention_weights"],
        case["im2col_step"],
    )


def main():
    args = parse_args()
    if args.build:
        build_library(args.build_dir)

    import torch

    for output in [args.csv, args.json, args.run_manifest]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("[msda_nvpw] initializing PyTorch CUDA context", flush=True)
    torch.cuda.init()
    print("[msda_nvpw] loading captured real BEVFormer MSDA tensors", flush=True)
    case = load_case(args.input)
    op = import_op()

    lib_path = Path(args.build_dir) / "libbevformer_msda_nvpw_replay.so"
    if not lib_path.exists():
        build_library(args.build_dir)
    lib = configure_lib(lib_path)

    print("[msda_nvpw] configuring CUPTI Range Profiling + NVPW", flush=True)
    check(
        lib,
        lib.bev_msda_nvpw_init(
            args.metrics.encode(),
            str(args.csv).encode(),
            str(args.json).encode(),
            1,
        ),
        "init",
    )

    pass_count = 0
    all_passes = False
    try:
        while not all_passes and pass_count < args.max_passes:
            check(lib, lib.bev_msda_nvpw_begin_pass(), "begin_pass")
            check(lib, lib.bev_msda_nvpw_push_range(b"range_ms_deformable_attention"), "push_range")
            out = call_op(op, case)
            torch.cuda.synchronize()
            check(lib, lib.bev_msda_nvpw_pop_range(), "pop_range")
            all_passes = bool(lib.bev_msda_nvpw_end_pass())
            pass_count += 1
        if not all_passes:
            raise RuntimeError(f"CUPTI requested more than {args.max_passes} replay passes")
    finally:
        check(lib, lib.bev_msda_nvpw_finalize(), "finalize")

    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metric_source": "cupti_range_nvpw_msda_replay",
        "range_mode": "CUPTI_UserRange",
        "replay_mode": "CUPTI_UserReplay",
        "range_name": "range_ms_deformable_attention",
        "profiled_target": "msda_operator_range",
        "passes": pass_count,
        "csv": args.csv,
        "json": args.json,
        "output_shape": tuple(int(x) for x in out.shape),
        "input": args.input,
        "capture_source": case["capture_source"],
        "capture_stage": case["capture_stage"],
        "capture_call_index": case["capture_call_index"],
        "input_shapes": {
            "value": list(case["value"].shape),
            "spatial_shapes": case["spatial_shapes"].detach().cpu().tolist(),
            "sampling_locations": list(case["sampling_locations"].shape),
            "attention_weights": list(case["attention_weights"].shape),
        },
        "metrics": args.metrics.split(","),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
    }
    Path(args.run_manifest).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
