import os
import sys
import time
import subprocess
import urllib.request

def test_server():
    print("=" * 60)
    print("TESTING MOBILE WEB BENCHMARK HTTP SERVER & ASSETS")
    print("=" * 60)
    server_process = subprocess.Popen(
        [sys.executable, "tools/mobile_web_demo/server.py", "8089"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(1.5)

    try:
        endpoints = [
            ("", "text/html"),
            ("index.html", "text/html"),
            ("style.css", "text/css"),
            ("demo.js", "application/javascript"),
            ("models/nanodet-improved4-int8.onnx", "application/octet-stream"),
            ("models/nanodet-improved4-fp32.onnx", "application/octet-stream"),
            ("models/nanodet-improved4-fp16.onnx", "application/octet-stream"),
            ("models/rtmpose-13kp-hand-fp16.onnx", "application/octet-stream"),
            ("models/stage-a-256-13kp-int8.onnx", "application/octet-stream"),
            ("models/stage-a-256-13kp-fp16.onnx", "application/octet-stream"),
            ("vendor/ort.min.js", "application/javascript"),
            ("vendor/float16.min.js", "application/javascript"),
            ("vendor/ort-wasm-simd-threaded.wasm", "application/wasm")
        ]

        base_url = "http://127.0.0.1:8089/"
        for endpoint, expected_mime in endpoints:
            url = base_url + endpoint
            req = urllib.request.Request(url, method="HEAD")
            resp = urllib.request.urlopen(req, timeout=5)
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            coop = resp.headers.get("Cross-Origin-Opener-Policy", "")
            coep = resp.headers.get("Cross-Origin-Embedder-Policy", "")
            print(f"  [+] {endpoint or 'root':<38}: Status {status} | MIME: {content_type:<26} | COOP: {coop}")
            assert status == 200, f"Failed status for {url}: {status}"
            assert expected_mime in content_type, f"MIME mismatch for {url}: expected {expected_mime}, got {content_type}"
            assert coop == "same-origin", f"COOP missing or incorrect on {url}"

        print("\n" + "=" * 60)
        print("ALL HTTP STATIC SERVER ENDPOINTS PASSED WITH 100% SUCCESS!")
        print("=" * 60)

    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_server()

