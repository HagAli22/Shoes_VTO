"""
HTTP Server with Cross-Origin-Opener-Policy (COOP) and Cross-Origin-Embedder-Policy (COEP)
headers enabled to unlock WebAssembly SharedArrayBuffer / multi-threading and WebGPU in Chrome.
"""
import sys
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

class IsolatedHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

def main(port=6006):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root_dir)
    server = HTTPServer(("0.0.0.0", port), IsolatedHandler)
    print(f"🚀 COOP/COEP Isolated Server running at http://0.0.0.0:{port}")
    print(f"📁 Serving root: {root_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 6006
    main(port)
