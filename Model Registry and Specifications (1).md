# Model Registry and Specifications 

Organization: NovaGate Technologies Project: Wrist Pose Detection System Document Version: 1.0 Last Updated: September 6, 2026 

## Table of Contents 

1. Production Models 

2. Model Version Comparison 

3. Resolution Variants 

4. Quantization Variants 

5. Model Selection Guide 

6. Technical Specifications 

## 1. Production Models 

### 1.1 Primary Production Model (DEPLOYED) 

Model: `wrist_pose_v20_vis_target_fixed_256.onnx` Status: ✓ ACTIVE IN PRODUCTION Architecture: YOLOv8n-pose Size: 11.90 MB Input Resolution: 256×256 Deployment Date: August 8, 2026 

Performance Metrics: 

- Detection Rate: 98.5% 

- Keypoint Error: 1.95 px (mean) 

- Inference Time: 9.34 ms (WebGPU), 9-12 ms (WASM multi-threaded) FPS: 107 (WebGPU), 80-100 (WASM) 

- Temporal Stability: 18 spikes per 2,400 frames (0.75%) 

- Confidence: 0.712 ± 0.128 

Use Cases: 

- Real-time AR bracelet/watch try-on (primary application) 

Web-based wrist tracking applications 

- Desktop and mobile AR experiences 

- All production deployments 

Deployment Configuration: 

Backend: WebGPU (preferred), WASM multi-threaded (fallback), WASM single-threaded (universal fallback) Confidence Threshold: 0.30 (default) 

Smoothing: Enabled (One-Euro filter + EMA) 

This is the model currently serving all production traffic. 

### 1.2 Alternative Model (Not Deployed) 

Model: `wrist_pose_v20_vis_target_fixed_256_yolo_v11.onnx` Status: Available (not deployed) Architecture: YOLOv11n-pose Size: 10.32 MB Input Resolution: 256×256 

Performance Metrics: 

Detection Rate: 98.5% 

Keypoint Error: 1.95 px 

- Inference Time: 9.34 ms (WebGPU) 

FPS: 107 (WebGPU) 

#### Use Cases: 

Future deployment consideration 

Size-optimized variant (1.58 MB smaller than YOLOv8) 

Note: Performance identical to deployed YOLOv8 model. Smaller file size but not currently used in production. YOLOv8 variant chosen for broader compatibility and proven stability. 

## 2. Model Version Comparison 

### 2.1 Version Evolution 

|Version|Release Date|Architecture|Training Data|Status|
|---|---|---|---|---|
|V18|July2026|YOLOv8n-pose|V5+ V10+ V11|Deprecated|
|V19|August3,2026|YOLOv8n-pose|V5+ V10+ V11|Superseded|
|V20|August8,2026|YOLOv8/v11n-pose|V5+ V10+ V11+ V13|Production|



### 2.2 V19 vs V20 Performance Comparison 

Dataset Differences: 

|Aspect|V19|V20|
|---|---|---|
|Training Images|4,655|5,793|
|Datasets Used|V5+ V10+ V11|V5+ V10+ V11+ V13|



+1,138 (V13 after dedup) 

Unique Images Added 

#### Performance Metrics (256×256 Resolution): 

|Metric|V19-256|V20-256|Improvement|Winner|
|---|---|---|---|---|
|Detection Rate|97.8%|98.5%|+0.7%|V20|
|Keypoint Error (px)|2.14|1.95|-8.9%|V20|
|Tracking Spikes|25|18|-28.0%|V20|
|Confidence (mean)|0.685|0.712|+3.9%|V20|
|Confidence (std)|0.142|0.128|-9.9%|V20|
|Inference Time (ms)|9.78|9.34|-4.5%|V20|
|P95Inference (ms)|11.79|11.21|-4.9%|V20|
|FPS|102|107|+4.9%|V20|



Conclusion: V20 wins all 8 metrics. V19 deprecated for production use. 

### 2.3 Critical Scenario Analysis 

Fast Motion Tracking (Hardest Test Case): 

|Metric|V19-256|V20-256|Improvement|
|---|---|---|---|
|Detection Rate|93.5%|95.5%|+2.0%|
|Keypoint Error|5.08px|4.59px|-9.6%|
|Tracking Spikes|38|26|-31.6%|
|Inference Time|11.55ms|11.07ms|-4.2%|



#### Static Back View (Occlusion Test): 

|Metric|V19-256|V20-256|Improvement|
|---|---|---|---|
|Detection Rate|100.0%|99.9%|-0.1% (1frame)|
|Keypoint Error|0.64px|0.55px|-14.1%|
|Tracking Spikes|0|1|Minor regression|
|Inference Time|9.25ms|9.00ms|-2.7%|



Assessment: V20 significantly better in difficult scenarios (fast motion). Minor regression in static back (1 missed frame) acceptable given overall improvements. 

## 3. Resolution Variants 

### 3.1 Available Resolutions 

|Model|Resolution|Size<br>(MB)|Use Case|Status|
|---|---|---|---|---|
|`wrist_pose_v20_vis_target_fixed_256.onnx`|256×256|11.90|✓DEPLOYED<br>IN<br>PRODUCTION|Active|
|`wrist_pose_v20_vis_target_fixed_256_yolo_v11.onnx`|256×256|10.32|Alternative<br>(smaller)|Available|
|`wrist_pose_v19_vis_target_fixed_256.onnx`|256×256|11.90|Baseline<br>comparison|Superseded|
|`wrist_pose_v19_vis_target_fixed_416.onnx`|416×416|11.97|High<br>resolution<br>(V19)|Superseded|
|`wrist_pose_v19_vis_target_fixed_128.onnx`|128×128|11.87|Low<br>resolution|Testing<br>Only|
|`wrist_pose_v18_vis_target_fixed_416.onnx`|416×416|11.97|Legacy|Deprecated|



### 3.2 Resolution Trade-offs 

#### 128×128: 

Inference: ~6-7 ms (fastest) 

Accuracy: ~4-5 px error (insufficient for AR) 

Recommendation: Not suitable for production 

#### 256×256: 

Inference: ~9-10 ms (real-time) 

Accuracy: ~2 px error (excellent for AR) Recommendation: Optimal for production 

#### 416×416: 

Inference: ~18-22 ms (slower) 

Accuracy: ~1.2 px error (best) 

- Recommendation: Use only when precision critical and performance not constrained 

Production Standard: 256×256 selected as optimal speed/quality balance. 

## 4. Quantization Variants 

4.1 Available Quantizations (V20-256 Base) 

|Model|Type|Size<br>(MB)|Target<br>Backend|Status|
|---|---|---|---|---|
|`wrist_v20_256_FP32.onnx`|FP32|11.90|CPU/WASM|Baseline|
|`wrist_v20_256_FP16_WEBGPU.onnx`|FP16|5.98|WebGPU|Recommended|
|`wrist_v20_256_INT8_ACTIVATION_AWARE.onnx`|INT8|3.40|CPU|Not<br>Recommended|
|`wrist_v20_256_INT8_FP32_MIXED_WEBGPU.onnx`|Mixed|3.58|WebGPU|Testing|
|`wrist_v20_256_INT8_QDQ_WEBGPU.onnx`|INT8<br>QDQ|12.11|WebGPU|Testing|



### 4.2 Quantization Performance 

#### FP16 (Recommended for WebGPU): 

- Size Reduction: 50% (11.90 MB → 5.98 MB) 

- Inference Time: ~8.5 ms (slight improvement) 

- Quality Loss: Negligible (detection rate maintained) 

- Recommendation: Use for WebGPU deployment 

#### INT8 Full Quantization: 

- Size Reduction: 71% (11.90 MB → 3.40 MB) 

- Quality Loss: Severe (detection rate drops to 89%) 

- Issue: Confidence extraction degraded 

- Recommendation: Not suitable for production 

#### INT8/FP32 Mixed Precision: 

- Size Reduction: 70% (11.90 MB → 3.58 MB) 

- Quality Loss: Moderate (acceptable) 

- Use Case: Size-constrained environments 

- Recommendation: Testing phase, not production-ready 

### 4.3 Quantization Selection Guide 

|Deployment|Recommended Model|Size|Notes|
|---|---|---|---|
|Target||||
|WebGPU Desktop|`wrist_v20_256_FP16_WEBGPU.onnx`|5.98MB|Best performance|
|WASM Multi-|`wrist_pose_v20_vis_target_fixed_256_yolo_v11.onnx`|10.32|Standard|
|thread||MB|precision|
|WASM Single-|`wrist_pose_v20_vis_target_fixed_256_yolo_v11.onnx`|10.32|Slower but|
|thread||MB|accurate|



|Size-critical|Testing only|3.58MB|Quality trade-off|
|---|---|---|---|
|(<5MB)||||



## 5. Model Selection Guide 

### 5.1 Production Deployment Decision Tree 

```
Start
```

```
  ↓
Load: wrist_pose_v20_vis_target_fixed_256.onnx (11.90 MB)
  ↓
Is WebGPU available?
  ↓ YES → Use WebGPU backend (5-8 ms inference, 120+ FPS)
```

```
  ↓ NO
  ↓
Is Cross-Origin Isolated (COOP/COEP)?
```

```
  ↓ YES → Use WASM multi-threaded backend (9-12 ms inference, 80-100 FPS)
```

- `↓ NO ↓` 

```
Fall back to WASM single-threaded backend (35-45 ms inference, 20-25 FPS)
```

Production Model: `wrist_pose_v20_vis_target_fixed_256.onnx` Backend Selection: Automatic (runtime detection) Fallback Chain: WebGPU → WASM MT → WASM ST 

### 5.2 Use Case Recommendations 

Real-time AR Try-On (Primary Use Case - DEPLOYED): 

Model: `wrist_pose_v20_vis_target_fixed_256.onnx` (YOLOv8n-pose) 

- Backend: WebGPU (preferred) or WASM (automatic fallback) Resolution: 256×256 

Precision: FP32 

Status: Active in production 

#### High-Precision Applications: 

Model: `wrist_pose_v20_vis_target_fixed_416.onnx` (if available) 

- Backend: WebGPU or WASM multi-threaded Resolution: 416×416 

Note: Slower inference, use only when accuracy critical 

#### Mobile Devices: 

Model: `wrist_pose_v20_vis_target_fixed_256.onnx` Backend: WASM (WebGPU support limited on mobile) Resolution: 256×256 

Note: Performance varies by device (typically WASM single-threaded) 

#### Development/Testing: 

Model: `wrist_pose_v19_vis_target_fixed_256.onnx` 

Purpose: Comparison baseline 

Status: Do not deploy to production 

### 5.3 DO NOT USE (Deprecated Models) 

The following models are deprecated and must not be used in production: 

- `wrist_pose_v18_vis_target_fixed_416.onnx` (superseded by V20) 

- `wrist_pose_v19_vis_target_fixed_128.onnx` (insufficient accuracy) `wrist_pose_v19_vis_target_fixed_256.onnx` (superseded by V20) 

- `wrist_pose_v19_vis_target_fixed_416.onnx` (superseded by V20) `wrist_v20_256_INT8_ACTIVATION_AWARE.onnx` (quality degradation) 

## 6. Technical Specifications 

### 6.1 Model Architecture 

Base Architecture: YOLOv11n-pose (nano variant) 

#### Network Structure: 

Layers: 197 

- Parameters: 2.67M (V11), 2.72M (V8) 

- Anchor Scales: 3 (21², 42², 84²) 

- Total Anchors: 3,549 

#### Input Format: 

Shape: [1, 3, 256, 256] 

- Type: float32 

- Range: [0.0, 1.0] (normalized RGB) 

- Layout: NCHW (batch, channels, height, width) 

#### Output Format: 

Shape: [1, 24, 3549] 

Channels: 

- 0-3: Box (cx, cy, w, h) 

- 4-5: Class logits (left_wrist, right_wrist) 

- 6-23: Keypoints (x, y, v) × 6 

### 6.2 Keypoint Structure 

Production Keypoints (6 points): 

#### Wrist Ring: 

- kpt[0]: radial_styloid 

- kpt[1]: ulnar_styloid 

- kpt[2]: wrist_seat 

#### Forearm Ring: 

- kpt[3]: forearm_radial 

kpt[4]: forearm_ulnar 

- kpt[5]: forearm_mid 

#### Training Keypoints (8 points): 

Above 6 + kpt[6]: radial_edge + kpt[7]: ulnar_edge 

Edge points not exported to production models 

#### Visibility Values: 

- 0.0-1.0 (sigmoid output) 

- Threshold: 0.5 (typically) 

- Hysteresis: show=0.5, hide=0.25 (recommended) 

### 6.3 Performance Benchmarks (V20-256 YOLOv11) 

Hardware: Desktop PC, WebGPU backend 

|Scenario|Detection Rate|Keypoint Error|Spikes|Inference|
|---|---|---|---|---|
|Static Back|99.9%|0.55px|1|9.00ms|
|Static Palm|100.0%|0.43px|0|12.20ms|
|Slow Rotation|98.7%|1.81px|5|9.35ms|
|Fast Motion|95.5%|4.59px|26|11.07ms|
|Overall|98.5%|1.95px|18|9.34ms|



FPS: 107 average (WebGPU) 

### 6.4 Model File Hashes (SHA-256) 

For integrity verification: 

```
wrist_pose_v20_vis_target_fixed_256.onnx (PRODUCTION):
```

```
  Size: 11.90 MB
```

```
  SHA-256: [Compute and insert hash]
```

```
wrist_pose_v20_vis_target_fixed_256_yolo_v11.onnx (Alternative):
```

```
  Size: 10.32 MB
```

```
  SHA-256: [Compute and insert hash]
```

```
wrist_v20_256_FP16_WEBGPU.onnx (Quantized):
```

```
  Size: 5.98 MB
```

```
  SHA-256: [Compute and insert hash]
```

Note: Use SHA-256 verification to ensure model file integrity before deployment. 

## 7. Version Control and Deployment 

### 7.1 Model Versioning Scheme 

Format: `wrist_pose_v{VERSION}_vis_target_fixed_{RESOLUTION}.onnx` 

VERSION: Major version number (18, 19, 20, etc.) 

- RESOLUTION: Input size (128, 256, 416) 

Special Suffixes: 

- `_yolo_v11` : YOLOv11 architecture variant 

- `_FP16_WEBGPU` : FP16 quantized for WebGPU 

- `_INT8_*` : INT8 quantization variants 

### 7.2 Current Production Configuration 

Active Deployment: 

```
{
```

```
  "model": "wrist_pose_v20_vis_target_fixed_256.onnx",
  "version": "20",
  "architecture": "YOLOv8n-pose",
  "resolution": 256,
  "precision": "FP32",
  "size_mb": 11.90,
  "status": "production_active",
  "deployment_date": "2026-08-08",
  "backends": ["webgpu", "wasm"],
  "confidence_threshold": 0.30,
  "smoothing_enabled": true
}
```

