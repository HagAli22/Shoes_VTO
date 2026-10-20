import os, sys, json, cv2, numpy as np, onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = "deliverables/stage_a_256_13kp/models/stage-a-256-13kp-fp32.onnx"
FIXTURES_DIR = "deliverables/stage_a_256_13kp/fixtures"
SRC_FRAMES_DIR = os.path.join(FIXTURES_DIR, "source-frames")
RAW_TENSORS_DIR = os.path.join(FIXTURES_DIR, "raw-output-tensors")
EXPECTED_DECODED_PATH = os.path.join(FIXTURES_DIR, "expected-decoded.json")
TRANSFORM_CASES_PATH = os.path.join(FIXTURES_DIR, "crop-transform-cases.json")

os.makedirs(SRC_FRAMES_DIR, exist_ok=True)
os.makedirs(RAW_TENSORS_DIR, exist_ok=True)

# 13-Keypoint Schema matching model contract
KEYPOINT_NAMES_13 = [
    "toe_ground",        # Index 0
    "heel_back",         # Index 1
    "heel_ground",       # Index 2
    "ball_medial",       # Index 3
    "ball_lateral",      # Index 4
    "instep_top",        # Index 5
    "arch_medial",       # Index 6
    "midfoot_lateral",   # Index 7
    "malleolus_medial",  # Index 8
    "malleolus_lateral", # Index 9
    "toe_tip",           # Index 10
    "throat",            # Index 11
    "achilles"           # Index 12
]

def letterbox(img, target_size=256):
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, scale, pad_x, pad_y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def extract_sources():
    sources = {}
    
    # Video 1: test_video.mp4 (carpet)
    v1 = "data/test_video.mp4"
    if os.path.exists(v1):
        cap = cv2.VideoCapture(v1)
        for f_idx in [0, 150]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                name = f"case_0{len(sources)+1}_carpet_frame{f_idx}"
                sources[name] = frame
        cap.release()

    # Video 2: 6950859 (bench child hanging)
    v2 = "deliverables/stage_a_13kp/demo/collection_defferent_videos/original/6950859-hd_1920_1080_25fps.mp4"
    if os.path.exists(v2):
        cap = cv2.VideoCapture(v2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 151)
        ret, frame = cap.read()
        if ret:
            sources["case_03_bench_child_hanging"] = frame
        cap.release()

    # Video 3: 12227810 (outdoor sidewalk)
    v3 = "deliverables/stage_a_13kp/demo/collection_defferent_videos/original/12227810-hd_1080_1920_30fps.mp4"
    if os.path.exists(v3):
        cap = cv2.VideoCapture(v3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ret, frame = cap.read()
        if ret:
            sources["case_04_outdoor_sidewalk"] = frame
        cap.release()

    # Video 4: 12983463 (indoor wood)
    v4 = "deliverables/stage_a_13kp/demo/collection_defferent_videos/original/12983463-hd_720_1280_30fps.mp4"
    if os.path.exists(v4):
        cap = cv2.VideoCapture(v4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
        ret, frame = cap.read()
        if ret:
            sources["case_05_indoor_wood"] = frame
        cap.release()

    # Video 5: 19685579 (plain floor walking)
    v5 = "deliverables/stage_a_13kp/demo/collection_defferent_videos/original/19685579-hd_1080_1920_30fps.mp4"
    if os.path.exists(v5):
        cap = cv2.VideoCapture(v5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
        ret, frame = cap.read()
        if ret:
            sources["case_06_floor_walking"] = frame
        cap.release()

    # Model image 1: Side view reclined
    img_side = "C:/Users/Matrix Store/.gemini/antigravity/brain/18a8cf40-807b-41c5-bf4c-21f7a74062bb/.user_uploaded/media_1792491391731_e53427f3.png"
    if os.path.exists(img_side):
        sources["case_07_side_reclined"] = cv2.imread(img_side)

    # Model image 2: Back standing view
    img_back = "C:/Users/Matrix Store/.gemini/antigravity/brain/18a8cf40-807b-41c5-bf4c-21f7a74062bb/.user_uploaded/media_1792491306881_28e3e489.png"
    if os.path.exists(img_back):
        sources["case_08_back_standing"] = cv2.imread(img_back)

    return sources

def main():
    print("[*] Loading ONNX session for 256 Model:", MODEL_PATH)
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    sources = extract_sources()
    print(f"[*] Found {len(sources)} cross-scene source fixtures.")

    from deliverables.stage_a_13kp.demo.run_video_demo import (
        determine_geometric_chirality_13kp,
        nms_per_class,
        deduplicate_physical_feet
    )
    from src.models.stage_a.postprocess import derive_secondary_keypoints_15kp

    expected_decoded = {}
    crop_transforms = []

    for case_id, frame in sorted(sources.items()):
        orig_h, orig_w = frame.shape[:2]
        png_path = os.path.join(SRC_FRAMES_DIR, f"{case_id}.png")
        cv2.imwrite(png_path, frame)

        # Preprocess
        canvas, scale, pad_x, pad_y = letterbox(frame, 256)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        # Run ONNX inference
        raw_out = sess.run([out_name], {inp_name: blob})[0]  # shape [1, 45, 1344]
        npy_path = os.path.join(RAW_TENSORS_DIR, f"{case_id}.npy")
        np.save(npy_path, raw_out)

        preds = raw_out[0]  # [45, 1344]
        candidates = []
        for a in range(preds.shape[1]):
            raw_s0 = float(preds[4, a])
            raw_s1 = float(preds[5, a])
            s_left = raw_s0 if (0.0 <= raw_s0 <= 1.0) else sigmoid(raw_s0)
            s_right = raw_s1 if (0.0 <= raw_s1 <= 1.0) else sigmoid(raw_s1)
            max_conf = max(s_left, s_right)
            if max_conf < 0.25:
                continue

            cls_raw = 0 if s_left >= s_right else 1
            cx, cy, w, h = preds[0:4, a]
            x1 = (cx - w / 2.0 - pad_x) / scale
            y1 = (cy - h / 2.0 - pad_y) / scale
            x2 = (cx + w / 2.0 - pad_x) / scale
            y2 = (cy + h / 2.0 - pad_y) / scale

            kpts13 = []
            for k in range(13):
                kx = (preds[6 + k * 3, a] - pad_x) / scale
                ky = (preds[6 + k * 3 + 1, a] - pad_y) / scale
                raw_kv = float(preds[6 + k * 3 + 2, a])
                kv = raw_kv if (0.0 <= raw_kv <= 1.0) else sigmoid(raw_kv)
                kpts13.append({
                    "index": k,
                    "name": KEYPOINT_NAMES_13[k],
                    "x": float(round(kx, 2)),
                    "y": float(round(ky, 2)),
                    "visibility": float(round(kv, 4))
                })

            candidates.append({
                "bbox": [float(round(v, 2)) for v in [x1, y1, x2, y2]],
                "class_id": cls_raw,
                "confidence": float(round(max_conf, 4)),
                "keypoints": kpts13
            })

        # NMS and deduplication
        filtered = nms_per_class(candidates, iou_threshold=0.45)
        deduped = deduplicate_physical_feet(filtered, iou_threshold=0.60)

        # Geometric chirality & 15-KP derivation
        final_case_dets = []
        for det in deduped:
            geom_cls = determine_geometric_chirality_13kp(det["keypoints"])
            det["class_id"] = geom_cls
            det["class_name"] = "left_foot" if geom_cls == 0 else "right_foot"
            
            # Also compute derived 15-KP for SDK consumers
            kpts15 = derive_secondary_keypoints_15kp(det["keypoints"], bbox=det["bbox"])
            kpts15_clean = [{
                "index": kp["index"],
                "name": kp["name"],
                "x": float(round(kp["x"], 2)),
                "y": float(round(kp["y"], 2)),
                "visibility": float(round(kp["visibility"], 4))
            } for kp in kpts15]
            
            final_case_dets.append({
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "keypoints_13": det["keypoints"],
                "keypoints_15_derived": kpts15_clean
            })

        expected_decoded[case_id] = {
            "image_width": orig_w,
            "image_height": orig_h,
            "letterbox_scale": float(round(scale, 6)),
            "pad_x": float(round(pad_x, 2)),
            "pad_y": float(round(pad_y, 2)),
            "detections": final_case_dets
        }

        crop_transforms.append({
            "case_id": case_id,
            "orig_size": [orig_w, orig_h],
            "target_size": 256,
            "scale": float(round(scale, 6)),
            "pad": [float(round(pad_x, 2)), float(round(pad_y, 2))]
        })

        print(f"  [OK] Processed {case_id}: {orig_w}x{orig_h} -> {len(final_case_dets)} feet detected.")

    with open(EXPECTED_DECODED_PATH, "w", encoding="utf-8") as f:
        json.dump(expected_decoded, f, indent=2)
    print(f"\n[OK] Saved golden fixture: {EXPECTED_DECODED_PATH}")

    with open(TRANSFORM_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(crop_transforms, f, indent=2)
    print(f"[OK] Saved transform cases: {TRANSFORM_CASES_PATH}")

if __name__ == "__main__":
    main()
