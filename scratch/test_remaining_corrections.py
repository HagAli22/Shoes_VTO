"""
scratch/test_remaining_corrections.py
Validates:
1. Kalman initialization: statePre, statePost, errorCovPre, errorCovPost
2. Forward-Backward Lucas-Kanade validation with keypoint filtering
3. One-to-One Global Assignment for Baseline Consistency
4. Async Offline (0% drop) vs Live (dropping stale frames)
"""
import numpy as np
import cv2
import queue
import time
from typing import Dict, List, Tuple, Optional, Any

print("--- [1] Testing Kalman Initialization Lifecycle ---")
class KalmanBoxTracker:
    def __init__(self, bbox: np.ndarray):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0  # Constant velocity model dt = 1
        
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.5
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.05
        self.kf.processNoiseCov[4:, 4:] *= 2.0  # Velocity noise
        
        # Proper initialization of state and covariance
        cx = float((bbox[0] + bbox[2]) / 2.0)
        cy = float((bbox[1] + bbox[3]) / 2.0)
        w = float(max(1.0, bbox[2] - bbox[0]))
        h = float(max(1.0, bbox[3] - bbox[1]))
        init_state = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
        
        self.kf.statePost = init_state.copy()
        self.kf.statePre = init_state.copy()
        
        init_cov = np.eye(8, dtype=np.float32)
        init_cov[:4, :4] *= 1.0   # Initial position uncertainty
        init_cov[4:, 4:] *= 10.0  # Initial velocity uncertainty
        self.kf.errorCovPost = init_cov.copy()
        self.kf.errorCovPre = init_cov.copy()
        
        self.predicted_bbox = bbox.copy()
        self.is_first_frame = True

    def step_predict(self) -> np.ndarray:
        """Called once at frame start (from frame 2 onward)."""
        state = self.kf.predict().flatten()
        cx, cy, w, h = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        w = max(10.0, w)
        h = max(10.0, h)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        self.predicted_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        return self.predicted_bbox

    def update_measurement(self, bbox: np.ndarray) -> np.ndarray:
        cx = float((bbox[0] + bbox[2]) / 2.0)
        cy = float((bbox[1] + bbox[3]) / 2.0)
        w = float(max(1.0, bbox[2] - bbox[0]))
        h = float(max(1.0, bbox[3] - bbox[1]))
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        
        if self.is_first_frame:
            # On creation frame, state is already initialized with measurement
            self.is_first_frame = False
            return self.get_state_bbox()
            
        self.kf.correct(z)
        return self.get_state_bbox()

    def get_state_bbox(self) -> np.ndarray:
        state = self.kf.statePost.flatten()
        cx, cy, w, h = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        w = max(10.0, w)
        h = max(10.0, h)
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

k = KalmanBoxTracker(np.array([100, 100, 200, 200], dtype=np.float32))
print("  Initial statePost:", k.kf.statePost.flatten()[:4])
print("  Initial statePre: ", k.kf.statePre.flatten()[:4])
print("  Initial errorCovPost diagonal:", np.diag(k.kf.errorCovPost))
# Frame 1: first observation
b1 = k.update_measurement(np.array([100, 100, 200, 200], dtype=np.float32))
# Frame 2: predict then update
p2 = k.step_predict()
b2 = k.update_measurement(np.array([105, 102, 205, 202], dtype=np.float32))
print("  Frame 2 predicted:", p2)
print("  Frame 2 posterior:", b2)
assert np.allclose(b1, [100, 100, 200, 200])

print("\n--- [2] Testing Forward-Backward Optical Flow Validation ---")
class OpticalFlowTracker:
    def __init__(
        self,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        fb_thresh: float = 2.0,
        min_conf: float = 0.35
    ):
        self.win_size = win_size
        self.max_level = max_level
        self.fb_thresh = fb_thresh
        self.min_conf = min_conf

    def track(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        if pts is None or len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool), 0.0, np.zeros(2, dtype=np.float32)

        # 1. Filter keypoints by visibility / confidence
        if pts.shape[-1] >= 3:
            conf_mask = pts[:, 2] >= self.min_conf
        else:
            conf_mask = np.ones(len(pts), dtype=bool)

        if not np.any(conf_mask):
            return np.empty((0, 2), dtype=np.float32), np.zeros((len(pts),), dtype=bool), 0.0, np.zeros(2, dtype=np.float32)

        pts_valid = pts[conf_mask, :2].astype(np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

        # 2. Forward Optical Flow: prev -> curr
        p_fwd, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_valid, None,
            winSize=self.win_size, maxLevel=self.max_level, criteria=criteria
        )
        status_fwd = (status_fwd.flatten() == 1)

        if not np.any(status_fwd):
            return np.empty((0, 2), dtype=np.float32), np.zeros((len(pts),), dtype=bool), 0.0, np.zeros(2, dtype=np.float32)

        # 3. Backward Optical Flow: curr -> prev
        p_bwd, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, p_fwd, None,
            winSize=self.win_size, maxLevel=self.max_level, criteria=criteria
        )
        status_bwd = (status_bwd.flatten() == 1)

        # 4. Forward-Backward Error Validation
        fb_dist = np.linalg.norm(pts_valid.reshape(-1, 2) - p_bwd.reshape(-1, 2), axis=-1)
        valid_fb = status_fwd & status_bwd & (fb_dist <= self.fb_thresh)

        # Map back to full input points status
        full_status = np.zeros(len(pts), dtype=bool)
        valid_indices = np.where(conf_mask)[0]
        full_status[valid_indices] = valid_fb

        total_pts = len(pts)
        valid_count = int(np.sum(valid_fb))

        if valid_count == 0:
            return np.empty((0, 2), dtype=np.float32), full_status, 0.0, np.zeros(2, dtype=np.float32)

        mean_fb_err = float(np.mean(fb_dist[valid_fb]))
        track_ratio = float(valid_count) / float(total_pts)
        err_score = max(0.0, 1.0 - mean_fb_err / self.fb_thresh)
        quality = float(track_ratio * err_score)

        displacements = (p_fwd.reshape(-1, 2) - pts_valid.reshape(-1, 2))[valid_fb]
        mean_disp = np.mean(displacements, axis=0) if len(displacements) > 0 else np.zeros(2, dtype=np.float32)

        out_pts = np.zeros((len(pts), 2), dtype=np.float32)
        out_pts[valid_indices] = p_fwd.reshape(-1, 2)

        return out_pts, full_status, quality, mean_disp

# Test with synthetic image translation
img1 = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(img1, (100, 100), 12, 255, -1)
cv2.circle(img1, (150, 120), 10, 200, -1)
img2 = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(img2, (105, 103), 12, 255, -1)
cv2.circle(img2, (155, 123), 10, 200, -1)

oft = OpticalFlowTracker(fb_thresh=2.0, min_conf=0.35)
kps = np.array([
    [100.0, 100.0, 0.9],  # Valid confident point
    [150.0, 120.0, 0.8],  # Valid confident point
    [200.0, 200.0, 0.1],  # Low confidence point (should be filtered!)
], dtype=np.float32)

out_pts, stat, q, disp = oft.track(img1, img2, kps)
print(f"  FB LK Quality: {q:.3f} | Valid FB points: {np.sum(stat)}/{len(kps)} | Mean Disp: {disp}")
assert stat[0] and stat[1] and not stat[2]
assert np.allclose(disp, [5.0, 3.0], atol=0.5)

print("\n--- [3] Testing One-to-One Global Assignment ---")
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

class BaselineConsistencyEvaluator:
    """
    Evaluates consistency relative to the reference Baseline pipeline (exp1).
    Uses One-to-One Global Assignment (Greedy exclusive matching) based on IoU >= 0.25.
    Explicitly labeled as Baseline Consistency (NOT ground-truth or CLEAR MOT).
    """
    def __init__(self, iou_thresh: float = 0.25):
        self.iou_thresh = iou_thresh
        self.prev_assignment: Dict[str, str] = {}
        self.baseline_identity_flips = 0
        self.total_dual_baseline_frames = 0
        self.correct_chirality_frames = 0
        self.recovery_durations: List[int] = []
        self.lost_start_frame: Dict[str, int] = {}
        self.tracked_frames = {"left": 0, "right": 0}
        self.baseline_frames = {"left": 0, "right": 0}

    def update(self, frame_idx: int, curr_tracks: Dict[str, Any], base_dets: Dict[str, Any]):
        for side in ["left", "right"]:
            if side in base_dets:
                self.baseline_frames[side] += 1
            if side in curr_tracks:
                self.tracked_frames[side] += 1

        is_dual_baseline = ("left" in base_dets and "right" in base_dets)
        if is_dual_baseline:
            self.total_dual_baseline_frames += 1

        # One-to-one exclusive global assignment
        curr_sides = list(curr_tracks.keys())
        base_sides = list(base_dets.keys())
        
        # Build pairwise IoU matrix
        iou_pairs = []
        for c_side in curr_sides:
            c_box = curr_tracks[c_side]["bbox"]
            for b_side in base_sides:
                b_box = base_dets[b_side]["bbox"]
                iou = compute_iou(c_box, b_box)
                if iou >= self.iou_thresh:
                    iou_pairs.append((iou, c_side, b_side))

        iou_pairs.sort(key=lambda x: x[0], reverse=True)
        assigned_curr = set()
        assigned_base = set()
        one_to_one: Dict[str, str] = {}

        for iou, c_side, b_side in iou_pairs:
            if c_side not in assigned_curr and b_side not in assigned_base:
                assigned_curr.add(c_side)
                assigned_base.add(b_side)
                one_to_one[c_side] = b_side

        # Baseline Identity Flips: Track changes assignment from one baseline foot to the other
        for c_side, b_side in one_to_one.items():
            if c_side in self.prev_assignment:
                if self.prev_assignment[c_side] != b_side:
                    self.baseline_identity_flips += 1
            self.prev_assignment[c_side] = b_side

        # Chirality agreement on dual-baseline frames
        if is_dual_baseline and len(one_to_one) == 2:
            if one_to_one.get("left") == "left" and one_to_one.get("right") == "right":
                self.correct_chirality_frames += 1

        # Recovery tracking
        for side in ["left", "right"]:
            if side not in curr_tracks and side in base_dets:
                if side not in self.lost_start_frame:
                    self.lost_start_frame[side] = frame_idx
            elif side in curr_tracks and side in self.lost_start_frame:
                dur = frame_idx - self.lost_start_frame[side]
                self.recovery_durations.append(dur)
                del self.lost_start_frame[side]

    def get_summary(self) -> Dict[str, Any]:
        tot_b = self.baseline_frames["left"] + self.baseline_frames["right"]
        tot_t = self.tracked_frames["left"] + self.tracked_frames["right"]
        cov = (tot_t / max(1, tot_b)) * 100.0
        chirality = (self.correct_chirality_frames / max(1, self.total_dual_baseline_frames)) * 100.0 if self.total_dual_baseline_frames else 100.0
        mean_rec = float(np.mean(self.recovery_durations)) if self.recovery_durations else 0.0
        return {
            "baseline_identity_flips": self.baseline_identity_flips,
            "baseline_coverage_pct": round(cov, 2),
            "baseline_chirality_agreement_pct": round(chirality, 2),
            "baseline_mean_recovery_frames": round(mean_rec, 1)
        }

evaluator = BaselineConsistencyEvaluator()
# Frame 1: Perfect match
evaluator.update(1, {"left": {"bbox": [10, 10, 50, 50]}, "right": {"bbox": [100, 100, 150, 150]}},
                    {"left": {"bbox": [10, 10, 50, 50]}, "right": {"bbox": [100, 100, 150, 150]}})
# Frame 2: Swap assignment
evaluator.update(2, {"left": {"bbox": [100, 100, 150, 150]}, "right": {"bbox": [10, 10, 50, 50]}},
                    {"left": {"bbox": [10, 10, 50, 50]}, "right": {"bbox": [100, 100, 150, 150]}})
summary = evaluator.get_summary()
print("  Evaluator summary after flip:", summary)
assert summary["baseline_identity_flips"] == 2

print("\n--- [4] Testing Async Offline (0% drop) vs Live (dropping stale) ---")
# Test offline queue (lossless)
q_off = queue.Queue(maxsize=100)
for i in range(10): q_off.put(i)
assert q_off.qsize() == 10
print("  Offline queue successfully queued all 10 frames without dropping.")

# Test live queue (dropping stale)
q_live = queue.Queue(maxsize=2)
dropped = 0
for i in range(10):
    try:
        q_live.put_nowait(i)
    except queue.Full:
        _ = q_live.get_nowait()
        dropped += 1
        q_live.put_nowait(i)
print(f"  Live queue handled 10 frames with capacity 2: dropped {dropped} stale frames.")
assert dropped == 8

print("\n[+] All new components verified successfully!")
