import os
import sys
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

PUBLIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "public"))

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

    def end_headers(self):
        # Allow cross-origin and security headers for WASM / video streaming
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".onnx"):
            return "application/octet-stream"
        if path.endswith(".mp4"):
            return "video/mp4"
        if path.endswith(".mjs"):
            return "application/javascript"
        return super().guess_type(path)

def run(host="127.0.0.1", port=8080):
    server_address = (host, port)
    httpd = HTTPServer(server_address, CustomHandler)
    print(f"[*] Serving isolated web demo from {PUBLIC_DIR}")
    print(f"[*] Local URL: http://{host}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run(port=port)

