#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NVPW_SOURCE = "cupti_range_nvpw_msda_replay"
NVPW_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed.avg.per_cycle_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
]


def require(path):
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Required real-run artifact is missing: {path}")
    return path


def read_csv(path):
    require(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_ncu_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith('"ID"'):
            return list(csv.DictReader(lines[index:]))
    return []


def activity_tables(rows):
    stage_ms = defaultdict(float)
    kernel_ms = defaultdict(float)
    overhead_us = defaultdict(list)
    scheduling_us = defaultdict(list)
    launch_to_start_us = defaultdict(list)
    for row in rows:
        stage = row.get("stage_name") or "unattributed"
        kernel = row.get("kernel_name") or "unknown"
        duration_ms = float(row.get("gpu_duration_us") or 0.0) / 1000.0
        stage_ms[stage] += duration_ms
        kernel_ms[kernel] += duration_ms
        if row.get("activity_type") == "kernel":
            overhead_us[stage].append(
                float(row.get("cpu_launch_overhead_us") or 0.0)
            )
            scheduling_us[stage].append(
                float(row.get("scheduling_delay_us") or 0.0)
            )
            launch_to_start_us[stage].append(
                float(row.get("launch_to_start_us") or 0.0)
            )
    return (
        stage_ms,
        kernel_ms,
        overhead_us,
        scheduling_us,
        launch_to_start_us,
    )


def nvpw_values(rows):
    values = {}
    for row in rows:
        if row.get("metric_source") != NVPW_SOURCE:
            raise RuntimeError(
                "Unexpected NVPW metric source: "
                f"{row.get('metric_source')!r}"
            )
        values[row["metric_name"]] = float(row["metric_value"])
    missing = [metric for metric in NVPW_METRICS if metric not in values]
    if missing:
        raise RuntimeError(f"NVPW output is missing metrics: {missing}")
    return values


def ncu_values(rows):
    values = defaultdict(list)
    for row in rows:
        if "ms_deformable_im2col_gpu_kernel" not in row.get(
            "Kernel Name", ""
        ):
            continue
        try:
            values[row["Metric Name"]].append(float(row["Metric Value"]))
        except (KeyError, ValueError):
            continue
    return {
        metric: sum(samples) / len(samples)
        for metric, samples in values.items()
        if samples
    }


def average(values):
    return sum(values) / len(values) if values else 0.0


def write_activity_summary(activity_rows):
    (
        stage_ms,
        kernel_ms,
        overhead_us,
        scheduling_us,
        launch_to_start_us,
    ) = activity_tables(activity_rows)
    lines = [
        "# CUPTI Activity Summary",
        "",
        f"- activity rows: `{len(activity_rows)}`",
        "- stage duration is the sum of GPU activity rows assigned to that stage",
        "- launch averages use kernel rows only",
        "",
        "| stage | cumulative GPU ms | CPU launch overhead us | scheduling delay us | launch-to-start us |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, duration in sorted(
        stage_ms.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(
            f"| {stage} | {duration:.3f} | "
            f"{average(overhead_us[stage]):.3f} | "
            f"{average(scheduling_us[stage]):.3f} | "
            f"{average(launch_to_start_us[stage]):.3f} |"
        )
    lines += [
        "",
        "## Top GPU activities",
        "",
        "| kernel or activity | cumulative ms |",
        "|---|---:|",
    ]
    for name, duration in sorted(
        kernel_ms.items(), key=lambda item: item[1], reverse=True
    )[:15]:
        lines.append(f"| `{name}` | {duration:.3f} |")
    (REPORTS / "activity_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return stage_ms, kernel_ms


def write_range_summary(custom, official):
    lines = [
        "# MSDA Hardware Metrics",
        "",
        "- custom boundary: complete spatial-MSDA operator user range",
        "- official NCU boundary: core CUDA kernel only",
        "- percentages from these two boundaries are not numerically interchangeable",
        "",
        "| source | SM throughput % | IPC | DRAM % | active warps % | Tensor Core % |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| custom CUPTI/NVPW range | "
            f"{custom[NVPW_METRICS[0]]:.6f} | "
            f"{custom[NVPW_METRICS[1]]:.6f} | "
            f"{custom[NVPW_METRICS[2]]:.6f} | "
            f"{custom[NVPW_METRICS[3]]:.6f} | "
            f"{custom[NVPW_METRICS[4]]:.6f} |"
        ),
    ]
    if official:
        lines.append(
            "| official NCU kernel | "
            f"{official.get(NVPW_METRICS[0], 0.0):.6f} | "
            f"{official.get(NVPW_METRICS[1], 0.0):.6f} | "
            f"{official.get(NVPW_METRICS[2], 0.0):.6f} | "
            "not collected | not collected |"
        )
    (REPORTS / "range_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_final_report(
    activity_rows,
    stage_ms,
    kernel_ms,
    custom,
    benchmark_rows,
    stage_shapes,
    run_manifest,
    official,
):
    full_shape = next(
        row
        for row in benchmark_rows
        if row.get("sample_source") == "captured_real"
    )
    top_stage = max(stage_ms, key=stage_ms.get)
    top_activity = max(kernel_ms, key=kernel_ms.get)
    lines = [
        "# BEVFormer Real-Run Profiling Report",
        "",
        "## Evidence inputs",
        "",
        "- model input: nuScenes mini camera data",
        "- model: BEVFormer tiny checkpoint",
        "- Activity source: custom CUPTI Activity shared library",
        "- hardware-counter source: custom CUPTI user-range replay + NVPW",
        "- MSDA replay input: spatial cross-attention tensors captured from the real forward",
        "",
        "## Forward and shapes",
        "",
        f"- recorded stage shape variants: `{len(stage_shapes)}`",
        f"- captured MSDA stage: `{run_manifest.get('capture_stage')}`",
        f"- captured MSDA output shape: `{run_manifest.get('output_shape')}`",
        "",
        "## Activity observations",
        "",
        f"- activity rows: `{len(activity_rows)}`",
        f"- largest cumulative stage: `{top_stage}` ({stage_ms[top_stage]:.3f} ms)",
        f"- largest cumulative GPU activity: `{top_activity}` ({kernel_ms[top_activity]:.3f} ms)",
        "",
        "These are cumulative activity durations, not end-to-end wall-clock stage durations. "
        "Overlapping GPU work can make summed values differ from stage wall time.",
        "",
        "## Spatial MSDA benchmark",
        "",
        f"- full-shape mean: `{full_shape['latency_mean_ms']}` ms",
        f"- full-shape P50/P95: `{full_shape['latency_p50_ms']}` / `{full_shape['latency_p95_ms']}` ms",
        f"- full-shape tensors: `{full_shape['shape_config']}`",
        "",
        "## Hardware-counter observations",
        "",
        f"- custom range IPC: `{custom[NVPW_METRICS[1]]:.6f}`",
        f"- custom range active warps: `{custom[NVPW_METRICS[3]]:.6f}%`",
        f"- custom range Tensor Core activity: `{custom[NVPW_METRICS[4]]:.6f}%`",
        "",
        "Zero Tensor Core activity is consistent with MSDA's irregular bilinear sampling "
        "and weighted aggregation rather than dense matrix multiplication. A definitive "
        "memory-stall diagnosis would require additional L1/L2 traffic and warp-stall metrics.",
        "",
        "## Official comparison",
        "",
    ]
    if official:
        lines += [
            "- NCU core-kernel metrics were found under `reports/official_tools/`.",
            "- Compare IPC and qualitative tendencies; do not directly equate range-level and kernel-level elapsed percentages.",
        ]
    else:
        lines.append(
            "- Official NCU output is optional and has not been collected in this run."
        )
    (REPORTS / "bevformer_operator_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    activity_rows = read_csv(REPORTS / "activity_timeline.csv")
    nvpw_rows = read_csv(REPORTS / "range_metrics_msda_nvpw.csv")
    benchmark_rows = read_csv(REPORTS / "msda_latency_sweep.csv")
    stage_shapes = json.loads(
        require(REPORTS / "stage_shapes.json").read_text(encoding="utf-8")
    )
    run_manifest = json.loads(
        require(REPORTS / "msda_nvpw_run.json").read_text(encoding="utf-8")
    )
    official_rows = read_ncu_csv(
        REPORTS / "official_tools" / "bevformer_ncu.csv"
    )

    stage_ms, kernel_ms = write_activity_summary(activity_rows)
    custom = nvpw_values(nvpw_rows)
    official = ncu_values(official_rows)
    write_range_summary(custom, official)
    write_final_report(
        activity_rows,
        stage_ms,
        kernel_ms,
        custom,
        benchmark_rows,
        stage_shapes,
        run_manifest,
        official,
    )
    print(f"Wrote real-run reports to {REPORTS}")


if __name__ == "__main__":
    main()
