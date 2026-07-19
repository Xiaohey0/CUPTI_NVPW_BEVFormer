#!/usr/bin/env python3
import csv
import struct
import zlib
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_bar(path, labels, values, title, ylabel):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        save_simple_bar_png(path, values)
        return

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    ax.bar(labels, values, color="#2f6f8f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_kernel_bar(path, labels, values):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        save_simple_bar_png(path, values)
        return

    display = [label if len(label) <= 56 else label[:53] + "..." for label in labels]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(list(reversed(display)), list(reversed(values)), color="#2f6f8f")
    ax.set_title("Top kernels by duration")
    ax.set_xlabel("ms")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_simple_bar_png(path, values, width=900, height=420):
    max_value = max(values) if values else 1.0
    bar_count = max(1, len(values))
    bar_w = max(10, width // (bar_count * 2))
    pixels = bytearray()
    for y in range(height):
        pixels.append(0)
        for x in range(width):
            r, g, b = 250, 250, 250
            if y > height - 40 or x < 40:
                r, g, b = 40, 40, 40
            for i, value in enumerate(values):
                left = 60 + i * (bar_w * 2)
                right = left + bar_w
                bar_h = int((height - 80) * float(value) / max_value)
                if left <= x <= right and height - 40 - bar_h <= y <= height - 40:
                    r, g, b = 47, 111, 143
            pixels.extend([r, g, b])
    raw = bytes(pixels)
    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    activity = read_csv(REPORTS / "activity_timeline.csv")
    ranges = read_csv(REPORTS / "range_metrics_msda_nvpw.csv")

    stage_ms = defaultdict(float)
    kernel_ms = defaultdict(float)
    for row in activity:
        ms = float(row.get("gpu_duration_us") or 0.0) / 1000.0
        stage_ms[row.get("stage_name", "unknown")] += ms
        kernel_ms[row.get("kernel_name", "unknown")] += ms
    if stage_ms:
        labels, values = zip(*sorted(stage_ms.items(), key=lambda item: item[1], reverse=True))
        save_bar(
            FIGURES / "stage_latency_bar.png",
            labels,
            values,
            "Stage cumulative GPU activity duration",
            "summed ms",
        )
    if kernel_ms:
        labels, values = zip(*sorted(kernel_ms.items(), key=lambda item: item[1], reverse=True)[:8])
        save_kernel_bar(FIGURES / "top_kernels_by_duration.png", labels, values)

    by_metric = defaultdict(dict)
    for row in ranges:
        by_metric[row.get("metric_name", "")][row.get("stage_name", "unknown")] = float(row.get("metric_value") or 0.0)
    metric_to_file = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_utilization_by_range.png",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_by_range.png",
        "smsp__inst_executed.avg.per_cycle_active": "ipc_by_range.png",
    }
    for metric, filename in metric_to_file.items():
        values = by_metric.get(metric)
        if values:
            labels, nums = zip(*sorted(values.items()))
            save_bar(FIGURES / filename, labels, nums, metric, "value")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
