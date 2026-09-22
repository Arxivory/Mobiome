### Comprehensive Deployment Benchmark Summary Table

| Model / Execution Engine | Device Backend  | Latency (ms)   | Throughput (FPS) | Footprint (MB) | Mean Drift Δτ (N·m)   |
| :----------------------- | :-------------- | :------------- | :--------------- | :------------- | :-------------------- |
| **OpenSim RNEA Solver**  | CPU Single Core | 118.40 ± 12.1  | 8.4              | 240.0          | 0.0000 (Ground Truth) |
| **DeepONet FP32 (ONNX)** | CPU Execution   | 20.60 ± 0.4    | 48.6             | 48.2           | Baseline              |
| **DeepONet INT8 (PTQ)**  | CPU / WebGPU    | **1.68 ± 0.2** | **593.8**        | **12.1**       | **24.7244**           |
