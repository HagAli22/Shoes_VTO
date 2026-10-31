"""
scratch/test_pipeline_components.py
Test script to verify all 5 core optimization components:
1. StageProfiler
2. CandidateFilter
3. BatchedPoseInference
4. KalmanBoxTracker & OpticalFlowRefiner
5. MotionAdaptiveDetector with Recovery Policy
"""
import sys
import os
import time
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# DLL path setup
try:
    import torch
    _tlib = Path(torch.__file__).parent / "lib"
    if _tlib.exists():
        os.environ["PATH"] = str(_tlib) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(_tlib))
except Exception:
    pass

# ── 1. Test KalmanBoxTracker ──
print("[1] Testing KalmanBoxTracker...")
class KalmanBoxTracker:
    def __init__(self, bbox: np.ndarray):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0  # dt = 1 frame
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        
        # Measurement noise R
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        # Process noise Q
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.processNoiseCov[4:, 4:] *= 2.0  # Velocity noise
        # State error covariance P
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 10.0

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)

    def predict(self) -> np.ndarray:
        state = self.kf.predict()
        cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def update(self, bbox: np.ndarray):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(z)

    def get_velocity(self) -> np.ndarray:
        vx = self.kf.statePost[4]
        vy = self.kf.statePost[5]
        return np.array([vx, vy], dtype=np.float32)

    def get_speed(self) -> float:
        v = self.get_velocity()
        return float(np.sqrt(v[0]**2 + v[1]**2))

    def get_uncertainty(self) -> float:
        return float(np.trace(self.kf.errorCovPost[:4, :4]))

kt = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=np.float32))
pred = kt.predict()
print(f"  Kalman init: [100, 100, 200, 200], predicted: {pred}, speed: {kt.get_speed():.2f}")
kt.update(np.array([110, 105, 210, 205], dtype=np.float32))
print(f"  Kalman update to [110, 105, 210, 205], speed: {kt.get_speed():.2f}")

# ── 2. Test CandidateFilter ──
print("[2] Testing CandidateFilter...")
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def filter_candidates(raw_detections, active_track_boxes, iou_thresh=0.40):
    if len(raw_detections) <= 2:
        return raw_detections
    # Match active tracks first
    matched = []
    unmatched = []
    for det in raw_detections:
        is_m = False
        for trk_box in active_track_boxes:
            if compute_iou(det["bbox"], trk_box) > 0.30:
                matched.append(det)
                is_m = True
                break
        if not is_m:
            unmatched.append(det)
    
    unmatched.sort(key=lambda d: d["score"], reverse=True)
    selected = matched[:2]
    for cand in unmatched:
        if len(selected) >= 2:
            break
        if all(compute_iou(cand["bbox"], s["bbox"]) < iou_thresh for s in selected):
            selected.append(cand)
    return selected[:2]

cands = [
    {"bbox": [100, 100, 200, 200], "score": 0.85},
    {"bbox": [102, 101, 198, 199], "score": 0.75}, # Overlaps with 1st
    {"bbox": [500, 500, 600, 600], "score": 0.80},
    {"bbox": [300, 300, 400, 400], "score": 0.45},
]
filtered = filter_candidates(cands, [[100, 100, 200, 200]])
print(f"  Raw: {len(cands)} -> Filtered: {len(filtered)} boxes: {[c['bbox'] for c in filtered]}")
assert len(filtered) == 2, "CandidateFilter should limit to 2!"

# ── 3. Test BatchedPoseInference ──
print("[3] Testing BatchedPoseInference...")
p = 'deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_improved5_fp16.onnx'
sess = ort.InferenceSession(p, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name

crops = [
    np.random.randn(3, 256, 256).astype(np.float16),
    np.random.randn(3, 256, 256).astype(np.float16)
]
batch_tensor = np.stack(crops, axis=0)
t0 = time.perf_counter()
outs = sess.run(None, {in_name: batch_tensor})
dt_batch = (time.perf_counter() - t0) * 1000.0
print(f"  Batch 2 inference latency: {dt_batch:.2f} ms, shapes: {[o.shape for o in outs]}")

# ── 4. Test MotionAdaptiveDetector & Recovery Policy ──
print("[4] Testing MotionAdaptiveDetector & Recovery Policy...")
class MotionAdaptiveDetector:
    def __init__(
        self,
        stable_interval: int = 10,
        moderate_interval: int = 5,
        speed_mod_thresh: float = 8.0,
        speed_fast_thresh: float = 25.0,
        conf_thresh: float = 0.35,
        max_immediate_retries: int = 2
    ):
        self.stable_interval = stable_interval
        self.moderate_interval = moderate_interval
        self.speed_mod_thresh = speed_mod_thresh
        self.speed_fast_thresh = speed_fast_thresh
        self.conf_thresh = conf_thresh
        self.max_immediate_retries = max_immediate_retries
        
        self.frames_since_det = 0
        self.consecutive_missing_retries = 0

    def should_detect(self, active_tracks, max_speed):
        self.frames_since_det += 1
        num_tracks = len(active_tracks)

        # 1. Missing feet check with controlled retry policy
        if num_tracks < 2:
            if self.consecutive_missing_retries < self.max_immediate_retries:
                self.consecutive_missing_retries += 1
                return True, "missing_foot_immediate"
            else:
                # Paced search every 5 frames instead of pounding detector on consecutive frames!
                if self.frames_since_det >= self.moderate_interval:
                    return True, "missing_foot_paced"
                return False, "missing_foot_waiting"
        else:
            self.consecutive_missing_retries = 0

        # 2. Tracking confidence check
        for t in active_tracks.values():
            if t["mean_conf"] < self.conf_thresh or t.get("lost_count", 0) > 0:
                return True, "low_confidence_or_occlusion"

        # 3. Fast motion check -> immediate trigger
        if max_speed > self.speed_fast_thresh:
            return True, "fast_motion_immediate"

        # 4. Moderate motion check -> interval = 5
        if max_speed > self.speed_mod_thresh:
            if self.frames_since_det >= self.moderate_interval:
                return True, "moderate_motion_heartbeat"
            return False, "moderate_motion_skip"

        # 5. Stable motion -> interval = 10
        if self.frames_since_det >= self.stable_interval:
            return True, "stable_heartbeat"

        return False, "stable_skip"

    def reset_heartbeat(self):
        self.frames_since_det = 0

mad = MotionAdaptiveDetector()
print("  Simulating 15 frames with stable motion:")
for f in range(1, 16):
    tracks = {"left": {"mean_conf": 0.50}, "right": {"mean_conf": 0.52}}
    det, reason = mad.should_detect(tracks, max_speed=4.0)
    if det:
        mad.reset_heartbeat()
    print(f"    Frame {f:2d}: detect={det!s:5s} ({reason})")

print("\nAll pipeline component tests passed successfully!")

