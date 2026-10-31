import os
import sys
import numpy as np
import onnxruntime as ort
import urllib.request
import time

def verify_models():
    print("=" * 60)
    print("1. VERIFYING MODEL FILES IN tools/mobile_web_demo/public/models/")
    print("=" * 60)
    models = [
        ("nanodet-improved4-int8.onnx", 1.44),
        ("nanodet-improved4-fp32.onnx", 4.16),
        ("nanodet-improved4-fp16.onnx", 2.18),
        ("rtmpose-13kp-hand-fp16.onnx", 7.48),
        ("stage-a-256-13kp-int8.onnx", 3.46),
        ("stage-a-256-13kp-fp16.onnx", 6.53)
    ]
    base_dir = "tools/mobile_web_demo/public/models"
    for fname, exp_mb in models:
        fpath = os.path.join(base_dir, fname)
        assert os.path.exists(fpath), f"Missing model: {fpath}"
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  [+] {fname:<30}: {size_mb:.2f} MB (Expected ~{exp_mb} MB)")

    print("\n" + "=" * 60)
    print("2. VERIFYING TWO-STAGE ONNX SESSIONS & FORWARD PASSES")
    print("=" * 60)

    # 1. NanoDet INT8
    sess_int8 = ort.InferenceSession(os.path.join(base_dir, "nanodet-improved4-int8.onnx"))
    x_f32 = np.random.randn(1, 3, 256, 256).astype(np.float32)
    b, s = sess_int8.run(None, {"images": x_f32})
    print(f"  [+] NanoDet INT8: boxes {b.shape} {b.dtype}, scores {s.shape} {s.dtype}")

    # 2. NanoDet FP32
    sess_fp32 = ort.InferenceSession(os.path.join(base_dir, "nanodet-improved4-fp32.onnx"))
    b, s = sess_fp32.run(None, {"images": x_f32})
    print(f"  [+] NanoDet FP32: boxes {b.shape} {b.dtype}, scores {s.shape} {s.dtype}")

    # 3. NanoDet FP16
    sess_fp16 = ort.InferenceSession(os.path.join(base_dir, "nanodet-improved4-fp16.onnx"))
    x_f16 = np.random.randn(1, 3, 256, 256).astype(np.float16)
    b, s = sess_fp16.run(None, {"images": x_f16})
    print(f"  [+] NanoDet FP16: boxes {b.shape} {b.dtype}, scores {s.shape} {s.dtype}")

    # 4. RTMPose FP16
    sess_rtm = ort.InferenceSession(os.path.join(base_dir, "rtmpose-13kp-hand-fp16.onnx"))
    out_rtm = sess_rtm.run(None, {"images": x_f16})
    print(f"  [+] RTMPose FP16: outputs count {len(out_rtm)}, simcc_x {out_rtm[0].shape}, hand {out_rtm[5].shape}")

    print("\n" + "=" * 60)
    print("3. VERIFYING FULL TWO-STAGE PIPELINE ON ACTUAL FRAME")
    print("=" * 60)
    import cv2
    cap = cv2.VideoCapture("tools/mobile_web_demo/public/video.mp4")
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read tools/mobile_web_demo/public/video.mp4"
    h0, w0 = frame.shape[:2]
    print(f"  [+] Loaded video.mp4 frame: {w0}x{h0}")

    # Preprocess letterbox 256x256
    target_size = 256
    ratio = min(target_size / w0, target_size / h0)
    nw, nh = int(round(w0 * ratio)), int(round(h0 * ratio))
    pad_x, pad_y = (target_size - nw) // 2, (target_size - nh) // 2
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(frame, (nw, nh))
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

    inp = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # NanoDet FP32 inference
    t0 = time.perf_counter()
    b_out, s_out = sess_fp32.run(None, {"images": inp})
    t_det = (time.perf_counter() - t0) * 1000.0

    boxes = b_out[0]
    scores = s_out[0, :, 0]
    high_mask = scores >= 0.35
    print(f"  [+] Stage A NanoDet infer time: {t_det:.2f} ms | Candidates >= 0.35: {high_mask.sum()}")

    # NMS
    cand_boxes = boxes[high_mask]
    cand_scores = scores[high_mask]
    keep = []
    order = cand_scores.argsort()[::-1]
    for i in order:
        box_i = cand_boxes[i]
        overlap = False
        for k in keep:
            box_k = cand_boxes[k]
            # IoU
            x1 = max(box_i[0], box_k[0])
            y1 = max(box_i[1], box_k[1])
            x2 = min(box_i[2], box_k[2])
            y2 = min(box_i[3], box_k[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            a1 = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
            a2 = (box_k[2] - box_k[0]) * (box_k[3] - box_k[1])
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou > 0.45:
                overlap = True
                break
        if not overlap:
            keep.append(i)

    print(f"  [+] After NMS: {len(keep)} foot detections")
    keep = keep[:2]

    # Stage B
    for rank, idx in enumerate(keep):
        bx = cand_boxes[idx]
        sc = cand_scores[idx]
        # Map to original frame
        x1_orig = max(0.0, min(float(w0), (bx[0] - pad_x) / ratio))
        y1_orig = max(0.0, min(float(h0), (bx[1] - pad_y) / ratio))
        x2_orig = max(0.0, min(float(w0), (bx[2] - pad_x) / ratio))
        y2_orig = max(0.0, min(float(h0), (bx[3] - pad_y) / ratio))

        cx = (x1_orig + x2_orig) / 2.0
        cy = (y1_orig + y2_orig) / 2.0
        bw = max(1.0, x2_orig - x1_orig)
        bh = max(1.0, y2_orig - y1_orig)
        ar = max(bw / bh, bh / bw)
        extra = 0.15 if ar > 1.3 else 0.05
        size = max(bw, bh) * (1.0 + 0.25 + extra)

        # Crop
        crop_x1 = int(round(cx - size / 2.0))
        crop_y1 = int(round(cy - size / 2.0))
        crop_x2 = int(round(cx + size / 2.0))
        crop_y2 = int(round(cy + size / 2.0))

        crop_canvas = np.full((256, 256, 3), 114, dtype=np.uint8)
        sx1 = max(0, crop_x1)
        sy1 = max(0, crop_y1)
        sx2 = min(w0, crop_x2)
        sy2 = min(h0, crop_y2)

        scale_c = 256.0 / size
        dx1 = int(round((sx1 - crop_x1) * scale_c))
        dy1 = int(round((sy1 - crop_y1) * scale_c))
        dx2 = dx1 + int(round((sx2 - sx1) * scale_c))
        dy2 = dy1 + int(round((sy2 - sy1) * scale_c))

        if sx2 > sx1 and sy2 > sy1 and dx2 > dx1 and dy2 > dy1:
            part = cv2.resize(frame[sy1:sy2, sx1:sx2], (dx2 - dx1, dy2 - dy1))
            crop_canvas[dy1:dy2, dx1:dx2] = part

        crop_inp = crop_canvas.transpose(2, 0, 1)[None].astype(np.float16) / 255.0

        t0 = time.perf_counter()
        out_r = sess_rtm.run(None, {"images": crop_inp})
        t_pose = (time.perf_counter() - t0) * 1000.0

        sx_l = out_r[0][0] # [13, 512]
        sy_l = out_r[1][0] # [13, 512]
        vis_arr = out_r[3][0, :, 0] # [13]
        hand_p = float(out_r[5][0])

        # Decode SimCC
        kpt_confs = []
        for k in range(13):
            # Softmax
            px = np.exp(sx_l[k] - sx_l[k].max())
            px /= px.sum()
            idx_x = px.argmax()
            mask_x = np.abs(np.arange(512) - idx_x) <= 12
            mass_x = px[mask_x].sum()

            py = np.exp(sy_l[k] - sy_l[k].max())
            py /= py.sum()
            idx_y = py.argmax()
            mask_y = np.abs(np.arange(512) - idx_y) <= 12
            mass_y = py[mask_y].sum()

            conf = np.sqrt(mass_x * mass_y) * float(vis_arr[k])
            kpt_confs.append(conf)

        kpt_confs = np.array(kpt_confs)
        mean_conf = kpt_confs.mean()
        valid_kpts = (kpt_confs >= 0.20).sum()
        side = "Right Foot" if hand_p >= 0.50 else "Left Foot"

        print(f"  [+] Foot #{rank+1} ({side}, hand_prob={hand_p:.3f}):")
        print(f"      Score={sc:.3f}, Pose Infer={t_pose:.2f} ms, Mean Kpt Conf={mean_conf:.3f}, Valid Kpts={valid_kpts}/13")
        assert mean_conf >= 0.20 and valid_kpts >= 3, "Failed anatomical clutter gate!"

    print("\n" + "=" * 60)
    print("ALL TWO-STAGE PIPELINE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    verify_models()

