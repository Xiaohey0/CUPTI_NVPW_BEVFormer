#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DEFAULT_INPUT = (
    ROOT
    / "benchmarks/MSDA/captured_tensors"
    / "MSDA_sample.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the MMCV CUDA MSDA op with captured BEVFormer tensors."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--output-csv", default=str(REPORTS / "msda_latency_sweep.csv")
    )
    parser.add_argument(
        "--output-json", default=str(REPORTS / "msda_latency_sweep.json")
    )
    parser.add_argument(
        "--output-report", default=str(REPORTS / "msda_latency_report.txt")
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--sweep-real-shapes", action="store_true")
    return parser.parse_args()


def import_cuda_op():
    try:
        import mmcv.ops.multi_scale_deform_attn as module

        return module.MultiScaleDeformableAttnFunction.apply
    except Exception as exc:
        raise RuntimeError(
            "The MMCV CUDA multi-scale deformable attention extension is "
            "not importable in this environment."
        ) from exc


def load_case(path):
    import torch

    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"Captured real MSDA tensor is missing: {path}. "
            "Run scripts/capture_real_msda_sample.sh first."
        )
    case = torch.load(path, map_location="cuda")
    required = {
        "value",
        "spatial_shapes",
        "level_start_index",
        "sampling_locations",
        "attention_weights",
    }
    missing = sorted(required.difference(case))
    if missing:
        raise RuntimeError(f"Captured tensor is missing keys: {missing}")
    if case.get("capture_source") != "real_bevformer_forward":
        raise RuntimeError("MSDA input was not captured from a real BEVFormer forward")
    if case.get("capture_stage") != "ms_deformable_attention":
        raise RuntimeError(
            "Expected the spatial MSDA stage, got "
            f"{case.get('capture_stage')!r}. Recapture with the default runner."
        )
    case["sample_source"] = "captured_real"
    return case


def clone_case(case):
    return {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in case.items()
    }


def real_shape_sweep_cases(case):
    cases = []
    base_bs = int(case["value"].shape[0])
    base_queries = int(case["sampling_locations"].shape[1])
    base_points = int(case["sampling_locations"].shape[4])
    batch_options = sorted({1, base_bs})
    query_options = sorted(
        {
            max(1, base_queries // 4),
            max(1, base_queries // 2),
            base_queries,
        }
    )
    point_options = sorted({max(1, base_points // 2), base_points})

    for batch_size in batch_options:
        for num_query in query_options:
            for num_points in point_options:
                sliced = clone_case(case)
                sliced["value"] = case["value"][:batch_size].contiguous()
                sliced["sampling_locations"] = case["sampling_locations"][
                    :batch_size, :num_query, :, :, :num_points, :
                ].contiguous()
                weights = case["attention_weights"][
                    :batch_size, :num_query, :, :, :num_points
                ].contiguous()
                denominator = weights.sum(
                    (-1, -2), keepdim=True
                ).clamp_min(1e-12)
                sliced["attention_weights"] = (
                    weights / denominator
                ).contiguous()
                is_base = (
                    batch_size == base_bs
                    and num_query == base_queries
                    and num_points == base_points
                )
                sliced["sample_source"] = (
                    "captured_real" if is_base else "captured_real_slice"
                )
                sliced["sweep_config"] = {
                    "batch_size": batch_size,
                    "num_query": num_query,
                    "num_points": num_points,
                }
                cases.append(sliced)
    return cases


def call_op(op, case):
    return op(
        case["value"],
        case["spatial_shapes"],
        case["level_start_index"],
        case["sampling_locations"],
        case["attention_weights"],
        int(case.get("im2col_step", 64)),
    )


def shape_config(case):
    return {
        "value": [int(x) for x in case["value"].shape],
        "sampling_locations": [
            int(x) for x in case["sampling_locations"].shape
        ],
        "attention_weights": [
            int(x) for x in case["attention_weights"].shape
        ],
        "spatial_shapes": case["spatial_shapes"].detach().cpu().tolist(),
    }


def benchmark(op, case, warmup, repeats):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if warmup < 0 or repeats < 1:
        raise ValueError("--warmup must be >= 0 and --repeats must be >= 1")

    for _ in range(warmup):
        call_op(op, case)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = call_op(op, case)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return output, times


def percentile(values, pct):
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((pct / 100.0) * (len(ordered) - 1)))),
    )
    return ordered[index]


def write_outputs(args, rows):
    for path in [args.output_csv, args.output_json, args.output_report]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    Path(args.output_json).write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    primary = next(
        row for row in rows if row["sample_source"] == "captured_real"
    )
    lines = [
        "# Real MSDA Benchmark",
        "",
        "- backend: `MMCV CUDA extension`",
        "- source: spatial MSDA tensors captured from a real BEVFormer forward",
        f"- rows: `{len(rows)}`",
        f"- full-shape mean ms: `{primary['latency_mean_ms']}`",
        f"- full-shape P50/P95 ms: `{primary['latency_p50_ms']}` / `{primary['latency_p95_ms']}`",
        "",
        "| source | sweep | mean ms | P50 ms | P95 ms | shape |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sample_source']}` | `{row['sweep_config']}` | "
            f"{row['latency_mean_ms']} | {row['latency_p50_ms']} | "
            f"{row['latency_p95_ms']} | `{row['shape_config']}` |"
        )
    Path(args.output_report).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    op = import_cuda_op()
    base_case = load_case(args.input)
    cases = (
        real_shape_sweep_cases(base_case)
        if args.sweep_real_shapes
        else [base_case]
    )

    rows = []
    for case in cases:
        output, times = benchmark(op, case, args.warmup, args.repeats)
        rows.append(
            {
                "operator": "mmcv.cuda.multi_scale_deformable_attn",
                "sample_source": case["sample_source"],
                "capture_stage": case.get("capture_stage", ""),
                "capture_call_index": case.get("capture_call_index", ""),
                "sweep_config": json.dumps(
                    case.get("sweep_config", {}), sort_keys=True
                ),
                "repeats": args.repeats,
                "latency_mean_ms": round(statistics.mean(times), 6),
                "latency_p50_ms": round(percentile(times, 50), 6),
                "latency_p95_ms": round(percentile(times, 95), 6),
                "latency_min_ms": round(min(times), 6),
                "latency_max_ms": round(max(times), 6),
                "shape_config": json.dumps(
                    shape_config(case), sort_keys=True
                ),
                "output_shape": str(tuple(int(x) for x in output.shape)),
            }
        )
    write_outputs(args, rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
