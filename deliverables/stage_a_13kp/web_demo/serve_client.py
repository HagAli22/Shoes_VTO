#!/usr/bin/env python3
"""
serve_client.py
───────────────
Simple HTTP server for client-side ONNX Runtime Web demo.
Serves static files with CORS headers and automatically routes root/templates to client_side.html.

Usage:
  python deliverables/stage_a_13kp/web_demo/serve_client.py
"""

import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support and smart URL routing."""
    
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.onnx': 'application/octet-stream',
        '.js': 'application/javascript',
        '.json': 'application/json',
    }

    def do_GET(self):
        # Normalize and route root and index requests directly to client_side.html
        req_path = self.path.split('?')[0].rstrip('/')
        
        if req_path in ('', '/templates', '/index.html', '/templates/index.html'):
            self.send_response(302)
            self.send_header('Location', '/templates/client_side.html')
            self.end_headers()
            return
            
        # Support relative model path resolution from /templates/
        if self.path.startswith('/templates/models/'):
            self.path = self.path.replace('/templates/models/', '/models/', 1)

        return super().do_GET()
    
    def end_headers(self):
        # Add CORS and cross-origin isolation headers for SharedArrayBuffer & multi-threading
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, HEAD')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Serve client-side ONNX demo")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()
    
    # Change to web_demo directory
    web_demo_dir = Path(__file__).parent
    os.chdir(web_demo_dir)
    
    # Check if model exists
    model_path = Path("models/stage-a-320-13kp-v7.onnx")
    if not model_path.exists():
        print("\n[!] Model not found in models/ directory")
        print("[*] Copying model...")
        model_path.parent.mkdir(exist_ok=True)
        
        source_model = Path("../models/stage-a-320-13kp-v7.onnx")
        if source_model.exists():
            import shutil
            shutil.copy(source_model, model_path)
            print(f"[✓] Model copied to {model_path}")
        else:
            print(f"[!] Source model not found: {source_model}")
            print("[!] Please ensure the model exists")
            return 1
    
    print("="*75)
    print("  CLIENT-SIDE ONNX RUNTIME WEB DEMO")
    print("="*75)
    print(f"  Server: http://{args.host}:{args.port}")
    print(f"  Open: http://localhost:{args.port}/templates/client_side.html")
    print(f"  (Root / and /templates/ auto-redirect to client_side.html)")
    print("="*75)
    print("\n[*] Press Ctrl+C to stop\n")
    
    # Start server
    server = HTTPServer((args.host, args.port), CORSRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
