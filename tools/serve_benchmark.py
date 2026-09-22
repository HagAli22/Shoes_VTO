"""
serve_benchmark.py
──────────────────
HTTP Server with proper Cross-Origin Isolation headers to enable:
  1. WebGPU in onnxruntime-web (needs CORP + COEP headers)
  2. SharedArrayBuffer for multi-threaded WASM (needs COOP + COEP headers)

Run this instead of 'python -m http.server' to get accurate browser benchmarks.

Usage:
    python tools/serve_benchmark.py

Then open:
    http://localhost:8080/tools/benchmark_browser/index.html
"""

import http.server
import socketserver
import os

PORT = 8080
SERVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class IsolationHandler(http.server.SimpleHTTPRequestHandler):
    """Adds Cross-Origin Isolation headers required for WebGPU & SharedArrayBuffer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def end_headers(self):
        # Required for WebGPU backend in onnxruntime-web
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        # Cache control for large ONNX files
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format, *args):
        # Only log non-200 responses to keep terminal clean
        if args[1] not in ("200", "304"):
            super().log_message(format, *args)


print("=" * 60)
print("  Shoes VTO ONNX Benchmark Server")
print(f"  Serving: {SERVE_DIR}")
print(f"  URL: http://localhost:{PORT}/tools/benchmark_browser/index.html")
print()
print("  Headers enabled:")
print("    Cross-Origin-Opener-Policy:   same-origin")
print("    Cross-Origin-Embedder-Policy: require-corp")
print("    Cross-Origin-Resource-Policy: cross-origin")
print()
print("  These enable WebGPU and multi-threaded WASM in Chrome/Edge.")
print("  Press Ctrl+C to stop.")
print("=" * 60)

with socketserver.TCPServer(("", PORT), IsolationHandler) as httpd:
    httpd.serve_forever()

