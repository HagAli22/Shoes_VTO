"""
scratch/test_async_pipeline.py
Tests the 3-thread Asynchronous pipeline (Capture -> Infer -> Render)
"""
import time, queue, threading
import cv2, numpy as np

def run_async_test():
    cap = cv2.VideoCapture("data/test_video.mp4")
    total_frames = 30
    frame_queue = queue.Queue(maxsize=2)
    render_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    
    # Thread 1: Capture
    def capture_worker():
        idx = 0
        while not stop_event.is_set() and idx < total_frames:
            ret, frame = cap.read()
            if not ret: break
            idx += 1
            # Put latest frame, discard old if full
            try:
                frame_queue.put((idx, frame), timeout=0.1)
            except queue.Full:
                try: frame_queue.get_nowait()
                except queue.Empty: pass
                frame_queue.put((idx, frame), timeout=0.1)
        frame_queue.put(None) # Sentinel

    # Thread 2: Infer
    def infer_worker():
        while not stop_event.is_set():
            item = frame_queue.get()
            if item is None:
                render_queue.put(None)
                break
            idx, frame = item
            # Simulate inference
            time.sleep(0.015) # 15 ms
            dets = {"left": [100, 100, 200, 200]}
            render_queue.put((idx, frame, dets))

    # Thread 3: Render
    rendered = []
    def render_worker():
        while not stop_event.is_set():
            item = render_queue.get()
            if item is None: break
            idx, frame, dets = item
            rendered.append(idx)

    t1 = threading.Thread(target=capture_worker)
    t2 = threading.Thread(target=infer_worker)
    t3 = threading.Thread(target=render_worker)
    t0 = time.perf_counter()
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()
    dt = time.perf_counter() - t0
    cap.release()
    print(f"Async test finished in {dt:.3f}s, rendered {len(rendered)} frames ({len(rendered)/dt:.1f} FPS)")

if __name__ == "__main__":
    run_async_test()

