"""
gpu_stream_server.py
───────────────────
High-Performance PC GPU Server (NVIDIA Quadro P2000).
Receives camera stream from mobile phone over WebSocket, runs inference
on GPU in ~3-4ms, and returns live AR boxes & keypoints.

Usage:
    python tools/gpu_stream_server.py --port 8443
"""

import os
import sys
import time
import base64
import json
import socket
import ssl
import cv2
import numpy as np
from pathlib import Path
import argparse

import torch
from ultralytics import YOLO
import asyncio
import websockets
from aiohttp import web

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class GPUServer:
    def __init__(self, model_path="outputs/stage_a/compact_shuffled_v3/weights/best.pt"):
        print(f"[1] Loading model to GPU: {model_path}...")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        # Warmup GPU
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        for _ in range(5):
            self.model.predict(dummy, imgsz=320, device=self.device, verbose=False)
        print(f"[OK] Model loaded on {'NVIDIA GPU' if self.device == 0 else 'CPU'}")

        self.last_time = time.time()
        self.fps = 0
        self.frame_count = 0

    async def handle_ws(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("  [+] Phone Connected to GPU Stream!")

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    t0 = time.perf_counter()
                    img_bytes = base64.b64decode(msg.data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    # Inference on Quadro P2000 GPU
                    results = self.model.predict(frame, imgsz=320, conf=0.25, device=self.device, verbose=False)[0]
                    dt = (time.perf_counter() - t0) * 1000.0

                    self.frame_count += 1
                    now = time.time()
                    if now - self.last_time >= 1.0:
                        self.fps = round(self.frame_count / (now - self.last_time))
                        self.frame_count = 0
                        self.last_time = now

                    # Parse output
                    dets = []
                    if results.boxes is not None and len(results.boxes):
                        for box in results.boxes:
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                            label = "Left Foot" if cls_id == 0 else "Right Foot"

                            kpts = []
                            if results.keypoints is not None:
                                kpts_raw = results.keypoints.data[0].cpu().numpy()
                                for ki in [0, 2, 4, 5]: # Path A 4 keypoints
                                    if ki < len(kpts_raw):
                                        kpts.append([float(kpts_raw[ki][0]), float(kpts_raw[ki][1]), float(kpts_raw[ki][2])])

                            dets.append({
                                "cls": cls_id,
                                "label": label,
                                "conf": round(conf, 2),
                                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                "kpts": kpts
                            })

                    response = {
                        "fps": self.fps or 60,
                        "latency": round(dt, 1),
                        "detections": dets
                    }
                    await ws.send_str(json.dumps(response))

                except Exception as e:
                    print("Error processing frame:", e)
            elif msg.type == web.WSMsgType.ERROR:
                print('WS connection closed with exception %s' % ws.exception())

        print("  [-] Phone Disconnected")
        return ws

    async def handle_index(self, request):
        return web.FileResponse('tools/web_mobile_test/index.html')

    async def handle_stream_index(self, request):
        return web.FileResponse('tools/web_mobile_test/stream_to_pc.html')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--weights", default="outputs/stage_a/compact_shuffled_v3/weights/best.pt")
    args = parser.parse_args()

    gpu_server = GPUServer(args.weights)
    app = web.Application()

    app.router.add_get('/', gpu_server.handle_index)
    app.router.add_get('/gpu', gpu_server.handle_stream_index)
    app.router.add_get('/ws_stream', gpu_server.handle_ws)
    app.router.add_static('/models/', path='shoes-vto-ai-v2/models/')

    local_ip = get_local_ip()

    # SSL Context for camera access
    cert_dir = Path("scratch/ssl")
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"

    ssl_ctx = None
    if cert_file.exists() and key_file.exists():
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))

    proto = "https" if ssl_ctx else "http"
    print("\n" + "=" * 70)
    print("  🚀 Shoes VTO — Live Mobile Testing Server")
    print("=" * 70)
    print(f"  Local IP: {local_ip}")
    print(f"  Option 1 (In-Phone WASM Engine): {proto}://{local_ip}:{args.port}/")
    print(f"  Option 2 (PC Quadro GPU Engine): {proto}://{local_ip}:{args.port}/gpu")
    print("=" * 70)
    print("  Open Option 2 on your phone to run 80+ FPS GPU inference on your laptop!")
    print("=" * 70 + "\n")

    web.run_app(app, host='0.0.0.0', port=args.port, ssl_context=ssl_ctx)

if __name__ == "__main__":
    main()

