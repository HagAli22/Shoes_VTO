# ONNX Operator Compatibility Report (ONNX Runtime Web)

## 1. Compliance Matrix
- **ONNX Opset:** 12 (Universal WebGPU / WASM / CoreML / NCNN compatibility).
- **Unsupported Operators:** 0 (Zero custom or non-standard kernels).
- **Graph Partitioning Risk:** None. Single execution graph with static shapes.

## 2. Operator List
- `Conv`, `BatchNormalization`, `SiLU`, `MaxPool`, `Concat`, `Reshape`, `Transpose`, `Sigmoid`, `Add`, `Mul`.
