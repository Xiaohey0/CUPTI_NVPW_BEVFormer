# BEVFormer CUPTI/NVPW Profiler

基于真实 nuScenes mini 输入和 BEVFormer-tiny 前向推理的 CUDA 性能分析项目。项目实现了两条互补的性能分析路径：

- 使用 CUPTI Activity API 关联 CPU Runtime API 与 GPU activity，生成完整 BEVFormer 推理时间线。
- 使用 CUPTI Range Profiling API 与 NVPW 对真实捕获的 MSDA 输入执行 User Replay，采集 GPU 硬件指标。

项目不使用随机 Tensor、模拟计数器或派生的伪硬件指标。延迟、时间线和 NVPW 指标都来自真实 CUDA 执行；生成结果和本地数据不会提交到 Git。

## 功能

### CUPTI Activity 时间线

`profilers/activity_profiler` 实现了一个通过 `LD_PRELOAD` 注入 Python 进程的 C++ profiler：

- 注册 CUPTI activity buffer request/completion callback。
- 采集 CUDA Runtime、kernel、memcpy 和 memset activity。
- 使用 CUPTI correlation ID 关联 CPU launch 与 GPU execution。
- 统计 CPU API 开销、排队延迟、launch-to-start 和 GPU 执行时间。
- 结合 Python stage hook，将 kernel 归属到 image backbone、FPN、temporal attention、spatial attention、MSDA、detection head 等阶段。
- 输出 `reports/activity_timeline.csv` 和 `reports/activity_timeline.json`。

### CUPTI Range Profiling 与 NVPW

`profilers/msda_nvpw_replay` 使用 CUPTI UserRange 和 UserReplay：

- 从真实 BEVFormer Spatial Cross-Attention 前向中捕获 MSDA 输入 Tensor。
- 以 `range_ms_deformable_attention` 为用户范围重复执行同一个 MSDA 算子。
- 由 NVPW 将高层 metric 名称转换为芯片对应的原始计数器配置。
- 根据硬件要求自动执行多 pass replay。
- 使用 NVPW 计算 SM throughput、IPC、DRAM throughput、active warps 和 Tensor Pipe 活跃率等指标。
- 输出 `reports/range_metrics_msda_nvpw.csv`、JSON 和运行清单。

### MSDA 延迟基线

`benchmarks/MSDA/benchmark_msda_latency.py` 使用 CUDA Event 测量官方 MMCV MSDA CUDA 扩展：

- 完整真实捕获 Shape 作为主结果。
- 可选的真实 Tensor 裁剪 Shape sweep 用于观察 batch、query 和 sampling point 变化。
- 输出 CSV、JSON 和纯文本摘要。

## 数据流

```text
nuScenes 六路相机 + 标定 + ego pose + CAN bus
                         |
                         v
             BEVFormer-tiny 官方模型
                         |
       +-----------------+-----------------+
       |                                   |
       v                                   v
CUPTI Activity 全模型时间线       捕获真实 Spatial MSDA Tensor
                                           |
                              +------------+------------+
                              |                         |
                              v                         v
                       CUDA Event 延迟基线      CUPTI Range + NVPW
```

## 项目结构

```text
benchmarks/MSDA/              MSDA Tensor replay、延迟基线和 NVPW Python 编排
checkpoints/                  本地 checkpoint，不提交
data/                         本地 nuScenes 和 CAN bus，不提交
integrations/                 BEVFormer runner、stage hook、NVTX 和 Tensor capture
profilers/activity_profiler/  CUPTI Activity C++ 实现
profilers/msda_nvpw_replay/   CUPTI Range Profiling + NVPW C++ 实现
reports/                      本地生成结果，不提交
scripts/                      构建、运行、验证、可视化和清理入口
third_party/                  固定版本的 BEVFormer 与 mmdetection3d 源码
```

## 环境

已验证的软件组合：

| 组件 | 版本 |
|---|---|
| Ubuntu | 20.04 / WSL2 |
| Python | 3.8 |
| PyTorch | 1.9.1 + CUDA 11.1 runtime |
| MMCV Full | 1.4.0 |
| MMDetection | 2.14.0 |
| MMSegmentation | 0.14.1 |
| MMDetection3D | 0.17.1，仓库内 camera-only 补丁版本 |
| CUDA Toolkit | 需要包含 CUPTI、NVPW 和 CUPTI profiler host utilities |
| CMake | 3.18 或更高 |

GPU、驱动和 CUDA Toolkit 必须支持 CUPTI Range Profiling。PyTorch 自带的 CUDA runtime 版本可以与用于构建 profiler 的系统 CUDA Toolkit 分开，但组合兼容性需要由实际驱动验证。

初始化环境：

```bash
conda create -n bevformer-prof python=3.8 -y
conda activate bevformer-prof

# 安装与 CUDA runtime 匹配的 PyTorch 和 mmcv-full 后，再安装以下框架版本。
pip install mmdet==2.14.0 mmsegmentation==0.14.1
pip install -e third_party/mmdetection3d-0.17.1 --no-deps

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NUMBA_CPU_NAME=generic
export PYTHONDONTWRITEBYTECODE=1
```

MSDA 必须来自带 CUDA 扩展的 `mmcv-full`，不能使用仅包含 Python 实现的 `mmcv`。可通过下面的命令确认：

```bash
python3 -c "import mmcv._ext, torch; print(mmcv._ext.__file__); print(torch.cuda.is_available())"
```

## 本地输入

仓库不包含 nuScenes、CAN bus、checkpoint 或捕获 Tensor。运行前应准备：

```text
checkpoints/
└── bevformer_tiny_epoch_24.pth

data/
├── can_bus/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-mini/
    ├── nuscenes_infos_temporal_train.pkl
    └── nuscenes_infos_temporal_val.pkl
```

Checkpoint 可使用：

```bash
bash scripts/download_bevformer_checkpoint.sh
```

nuScenes 数据受其官方许可约束，应从 nuScenes 官网获取。Temporal PKL 由 BEVFormer 数据准备流程将 nuScenes token 关系、相机标定、ego pose、图像路径和 3D 标注整理为模型可读取的信息文件。

## 运行

先验证环境和本地输入：

```bash
conda activate bevformer-prof
export NUMBA_CPU_NAME=generic
export PYTHONDONTWRITEBYTECODE=1

python3 scripts/verify_real_pipeline.py --inputs
```

按顺序执行完整流程：

```bash
# 1. 真实 BEVFormer 前向、预测结果、stage shape 和 Spatial MSDA Tensor capture
bash scripts/capture_real_msda_sample.sh

# 2. 官方 MMCV MSDA CUDA 算子的 CUDA Event 延迟基线
bash scripts/run_msda_benchmark.sh

# 3. 完整 BEVFormer 的 CUPTI Activity 时间线
bash scripts/run_activity_profile.sh

# 4. 真实 MSDA Tensor 的 CUPTI Range Profiling + NVPW 指标
bash scripts/run_msda_nvpw.sh

# 5. 汇总结果和绘图
python3 scripts/collect_reports.py
python3 scripts/render_figures.py

# 6. 验证生成结果
python3 scripts/verify_real_pipeline.py --outputs
```

可选功能：

```bash
# 六相机、GT 和预测框可视化
python3 scripts/visualize_nuscenes.py

# 使用 Nsight Systems / Nsight Compute 生成独立对照
bash scripts/run_official_tool_comparison.sh
```

常用运行参数通过环境变量覆盖，不依赖个人绝对路径：

```bash
SAMPLES=2 bash scripts/capture_real_msda_sample.sh
WARMUP=20 REPEATS=100 bash scripts/run_msda_benchmark.sh
CUDA_HOME=/usr/local/cuda MAX_PASSES=16 bash scripts/run_msda_nvpw.sh
```

## 输出说明

主要生成文件：

| 文件 | 含义 |
|---|---|
| `reports/bevformer_outputs.pt` | BEVFormer 预测结果 |
| `reports/stage_shapes.json` | 模型阶段与 MSDA Tensor Shape |
| `benchmarks/MSDA/captured_tensors/MSDA_sample.pt` | 真实 Spatial MSDA 输入 |
| `reports/msda_latency_sweep.csv` | CUDA Event 延迟结果 |
| `reports/activity_timeline.csv` | CPU/GPU 关联时间线 |
| `reports/range_metrics_msda_nvpw.csv` | CUPTI Range + NVPW 硬件指标 |

Activity 时间线中主要字段：

- `correlation_id`：CUPTI 生成的 Runtime API 与 GPU activity 关联键。
- `cpu_launch_overhead_us`：CUDA Runtime API enter 到 exit。
- `scheduling_delay_us`：CPU API exit 到 GPU start。
- `launch_to_start_us`：CPU API enter 到 GPU start。
- `gpu_duration_us`：GPU activity start 到 end。
- `stage_name`：由 BEVFormer stage hook 标记的语义阶段。

NVPW 指标可能需要多个 replay pass。不同指标受 GPU 架构、驱动、时钟状态和 metric availability 影响，不能把某台机器上的数值当作跨设备固定结论。

## 第三方源码与补丁

`third_party/BEVFormer` 来自官方 BEVFormer 仓库，固定基线提交：

```text
https://github.com/fundamentalvision/BEVFormer.git
66b65f3a1f58caf0507cb2a971b9c0e7f842376c
```

`third_party/mmdetection3d-0.17.1` 来自 OpenMMLab MMDetection3D 0.17.1。两者的 Apache-2.0 `LICENSE` 均保留在对应目录。

本项目仅运行 camera-only BEVFormer-tiny。为了避免 Python import 阶段强制加载未使用的 mmdetection3d 点云扩展，vendored source 做了以下最小修改：

- 缩小 mmdetection3d core、dataset、pipeline、model 和 detector 的 eager import 范围。
- 将 3D IoU、RoIAwarePool3D、voxelization 和多增强 NMS 等点云依赖延迟到实际调用路径。
- 只注册 BEVFormer-tiny 推理需要的 plugin、dataset、pipeline、head 和 detector。
- 将自定义评估依赖延迟到 `evaluate()`，并删除未使用的 IPython import。
- mmdetection3d `setup.py` 不构建点云 C++/CUDA 扩展。

这些修改不会替代实际执行的 CUDA 算子。MSDA 仍由环境中的 `mmcv-full` CUDA extension 执行。当前源码不支持直接启用 LiDAR voxelization、PointNet、sparse convolution、RoIAwarePool3D 或 rotated 3D IoU/NMS；使用这些功能前需要恢复对应 registry 并编译官方扩展。

## 清理

删除所有可重新生成的 build、report、capture 和 Python cache，同时保留源码、数据与 checkpoint：

```bash
bash scripts/clean_generated.sh
python3 scripts/verify_real_pipeline.py --clean-state
```

## 说明

- `integrations/` 和 `profilers/` 是本项目的主要实现。
- `third_party/` 保留上游版权与许可证，不应被描述为本项目原创代码。
- `data/`、`checkpoints/`、`reports/`、native build 和捕获 Tensor 均由 `.gitignore` 排除。
- Nsight Systems/Compute 结果仅作为官方工具对照，不参与自研 profiler 的数据生成。
