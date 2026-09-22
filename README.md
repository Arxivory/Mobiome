# Quantized Neural Operators for Biomechanical Inverse Dynamics

> **Research status:** This project is actively in development and is currently a research prototype. Results, interfaces, and deployment claims may change as the experiments and validation pipeline mature.

This repository explores whether a quantized neural operator can approximate biomechanical inverse dynamics from monocular 3D motion data with sufficiently low latency for edge deployment. The central research question is how much precision and structural constraint can be traded for real-time inference without making joint-torque estimates physically unusable.

## Overview

The proposed pipeline has three connected stages:

1. **Kinematic regularization:** anatomical bone-length constraints and higher-order temporal smoothness reduce noise amplification in position derivatives.
2. **Operator learning:** a Deep Operator Network (DeepONet) maps windowed kinematic trajectories, including position, velocity, and acceleration, to joint torque trajectories.
3. **Edge-oriented quantization:** the FP32 operator is exported to ONNX and evaluated after INT8 post-training quantization. The benchmark compares empirical INT8 torque drift with an analytical drift bound and simulated derivative-noise error.

The repository also contains an OpenSim-facing inverse-dynamics bridge, data preparation utilities, ONNX inference checks, and 3D visualization scripts.

## Problem Formulation

Let generalized joint position, velocity, and acceleration be $\mathbf{q}(t)$, $\dot{\mathbf{q}}(t)$, and $\ddot{\mathbf{q}}(t)$. Classical inverse dynamics models torque as

$$\boldsymbol{\tau}= \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}+ \mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}+ \mathbf{g}(\mathbf{q}),$$

where $\mathbf{M}$ is the mass and inertia matrix, $\mathbf{C}$ captures Coriolis and centripetal effects, and $\mathbf{g}$ is the gravitational load.

Monocular reconstruction produces noisy positions,

$$\hat{\mathbf{p}}_t = \mathbf{p}_t + \boldsymbol{\epsilon}_t,$$

and finite differences amplify that noise as derivative order increases:

$$\operatorname{Var}(\dot{\hat{\mathbf{p}}}) \propto \frac{\sigma^2}{\Delta t^2},\qquad\operatorname{Var}(\ddot{\hat{\mathbf{p}}}) \propto \frac{\sigma^2}{\Delta t^4},\qquad\operatorname{Var}(\dddot{\hat{\mathbf{p}}})\propto\frac{\sigma^2}{\Delta t^6}.$$

The project addresses this instability before torque prediction by combining anatomical and temporal regularization with a learned operator.

## Proposed Method

The regularized pose objective is represented by

$$\mathcal{L}_{\text{total}}= \mathcal{L}_{\text{MPJPE}}+ \lambda_1 \mathcal{L}_{\text{bone}}+ \lambda_2 \mathcal{L}_{\text{smooth}},$$

where the bone term preserves calibrated segment lengths and the smoothness term penalizes unstable higher-order temporal derivatives. A representative bone penalty is

$$\mathcal{L}_{\text{bone}}= \sum_{(i,j)\in\mathcal{B}}\left|\left\|\hat{\mathbf{p}}_i-\hat{\mathbf{p}}_j\right\|_2-L_{ij}\right|.$$

The DeepONet then learns the functional mapping

$$\mathcal{G}: u(t)=[\mathbf{q}(t),\dot{\mathbf{q}}(t),\ddot{\mathbf{q}}(t)]\longmapsto \boldsymbol{\tau}(t),$$

using a branch network for the sampled trajectory and a trunk network for the output query location:

$$\mathcal{G}(u)(y)=\mathbf{b}(u)^\mathsf{T}\mathbf{t}(y)+b_0.$$

For deployment, the FP32 model is quantized to INT8. The primary numerical quantity is the torque drift

$$\Delta\boldsymbol{\tau}=\left\|\boldsymbol{\tau}_{\mathrm{FP32}}-\boldsymbol{\tau}_{\mathrm{INT8}}\right\|_2,$$

which is compared with the error produced by unconstrained derivative noise.

## Research Goals

- Reduce monocular pose noise in velocity, acceleration, and jerk.
- Learn a fast surrogate for classical biomechanical inverse dynamics.
- Measure the accuracy, latency, and memory trade-offs of low-precision inference.
- Compare quantization drift with physically meaningful torque errors.
- Evaluate the feasibility of browser and edge deployment.
- Build toward validation against independent motion-capture and biomechanical measurements.

## Current Scope

Implemented research components include:

- PyTorch DeepONet branch, trunk, and operator modules.
- Biomechanical derivative and anatomical loss components.
- Processed kinematic trajectory loading and normalization-statistic generation.
- DeepONet training over windowed kinematic and torque sequences.
- PyTorch PTQ and QAT helpers, plus ONNX model artifacts.
- A benchmark covering low, medium, and high synthetic movement-velocity regimes.
- Publication-oriented plots and a Markdown benchmark summary in `paper_results/`.

The current benchmark uses ONNX Runtime's CPU execution provider. WebGPU and browser execution are intended deployment targets and remain part of the ongoing research and validation work. The included benchmark artifacts should therefore be treated as preliminary project results, not as a completed clinical or scientific validation.

## Repository Layout

```text
checkpoints/      Trained PyTorch and ONNX model artifacts, normalization statistics
data/             Raw, processed, and synthetic trajectory data
paper_results/    Generated benchmark summaries and figures
scripts/          Training, preparation, quantization, inference, and visualization entry points
src/operators/    DeepONet branch, trunk, and operator implementation
src/physics/      OpenSim/inverse-dynamics bridge
src/vision/       Pose backbone and regularization losses
src/analytics/    Biomechanical evaluation metrics
src/utils/        Derivative and numerical utilities
tests/            Unit and pipeline tests
```

## Setup

Create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency list includes PyTorch, OpenSim, ONNX, ONNX Runtime, MediaPipe, PyVista, and scientific Python tooling. OpenSim installation can be platform-specific; verify that it is available in the selected Python environment before running the dynamics workflows.

## Reproduce the Main Workflows

Prepare processed inputs and synthetic targets:

```powershell
python -m scripts.prepare_data
```

Train the FP32 DeepONet and save its normalization statistics:

```powershell
python -m scripts.train
```

Run the quantization workflow:

```powershell
python -m scripts.quantize
```

Check that the exported FP32 ONNX model agrees with the PyTorch reference:

```powershell
python -m scripts.test_onnx_inference
```

Generate the paper-oriented latency, torque-drift, and derivative-noise artifacts:

```powershell
python -m scripts.benchmark_paper_metrics
```

The benchmark writes figures and a summary table to `paper_results/`. Existing checkpoint files allow inference and benchmark scripts to be run without retraining, provided their expected dependencies and paths are available.

## Preliminary Benchmark Artifact

The checked-in benchmark summary currently reports the following recorded comparison:

| Model               | Backend                          |   Latency | Throughput | Footprint |
| ------------------- | -------------------------------- | --------: | ---------: | --------: |
| OpenSim RNEA solver | CPU single core                  | 118.40 ms |    8.4 FPS |  240.0 MB |
| DeepONet FP32       | ONNX CPU                         |  20.60 ms |   48.6 FPS |   48.2 MB |
| DeepONet INT8 PTQ   | CPU / recorded deployment target |   1.68 ms |  593.8 FPS |   12.1 MB |

These numbers are environment-dependent and should be regenerated before being used in a paper, report, or deployment decision. In particular, the current benchmark code measures CPU ONNX Runtime sessions even though the long-term target includes WebGPU.

## Limitations and Open Work

- The end-to-end monocular video-to-torque workflow is still being integrated and validated.
- Synthetic and processed data do not replace validation against independent experimental measurements.
- The analytical quantization bound requires broader validation across architectures, calibration data, and hardware.
- Browser/WebGPU latency and numerical behavior still need dedicated measurement.
- Contact-rich motion, multiple people, subject variation, and clinical use are outside the currently validated scope.

Planned work includes quantization-aware training, comparisons with Fourier Neural Operators, independent motion-capture validation, and direct browser or edge-runtime evaluation.

## License

No license has been declared yet. Until one is added, assume that reuse and redistribution require permission from the author.
