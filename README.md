# CUPTI/NVPW BEVFormer Profiler

Run BEVFormer-tiny on nuScenes mini and profile its CUDA workload with CUPTI Activity, CUPTI Range Profiling, and NVPW.

## Requirements

- Ubuntu 20.04 or WSL2 with an NVIDIA GPU
- Python 3.8 and CMake 3.18+
- PyTorch 1.9.1 with CUDA 11.1 runtime
- `mmcv-full==1.4.0`
- `mmdet==2.14.0` and `mmsegmentation==0.14.1`
- CUDA Toolkit with CUPTI and NVPW

Install the vendored MMDetection3D package:

```bash
pip install -e third_party/mmdetection3d-0.17.1 --no-deps
```

## Data

Datasets and checkpoints are not included. Prepare this layout:

```text
checkpoints/
`-- bevformer_tiny_epoch_24.pth

data/
|-- can_bus/
`-- nuscenes/
    |-- maps/
    |-- samples/
    |-- sweeps/
    |-- v1.0-mini/
    |-- nuscenes_infos_temporal_train.pkl
    `-- nuscenes_infos_temporal_val.pkl
```

Download the checkpoint with:

```bash
bash scripts/download_bevformer_checkpoint.sh
```

Download nuScenes mini and CAN bus data from the official nuScenes website. The BEVFormer temporal train and validation PKL files must also be available before running inference.

## Usage

```bash
conda activate bevformer-prof
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CPU_NAME=generic
export PYTHONDONTWRITEBYTECODE=1

# Verify the environment and local inputs.
python3 scripts/verify_real_pipeline.py --inputs

# Run real BEVFormer inference and capture Spatial MSDA tensors.
bash scripts/capture_real_msda_sample.sh

# Measure the MMCV MSDA CUDA latency baseline.
bash scripts/run_msda_benchmark.sh

# Collect the full BEVFormer CUPTI Activity timeline.
bash scripts/run_activity_profile.sh

# Collect CUPTI Range Profiling and NVPW metrics for MSDA.
bash scripts/run_msda_nvpw.sh

# Aggregate reports and render figures.
python3 scripts/collect_reports.py
python3 scripts/render_figures.py

# Verify generated outputs.
python3 scripts/verify_real_pipeline.py --outputs
```

Generated files are written to `reports/`. The captured MSDA input is written to `benchmarks/MSDA/captured_tensors/MSDA_sample.pt`.

## Optional

```bash
# Visualize cameras, ground truth, and predictions.
python3 scripts/visualize_nuscenes.py

# Compare with Nsight Systems and Nsight Compute.
bash scripts/run_official_tool_comparison.sh
```

Common overrides:

```bash
SAMPLES=2 bash scripts/capture_real_msda_sample.sh
WARMUP=20 REPEATS=100 bash scripts/run_msda_benchmark.sh
CUDA_HOME=/usr/local/cuda MAX_PASSES=16 bash scripts/run_msda_nvpw.sh
```

## Cleanup

```bash
bash scripts/clean_generated.sh
python3 scripts/verify_real_pipeline.py --clean-state
```
