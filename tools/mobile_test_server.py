"""
mobile_test_server.py
─────────────────────
Lightweight local HTTPS/HTTP server to test Stage A ONNX models directly
on mobile phone browsers (iOS Safari / Android Chrome) using WebGPU/WASM.

Usage:
    python tools/mobile_test_server.py --port 8443
"""

import os
import sys
import socket
import ssl
import http.server
import socketserver
import argparse
from pathlib import Path

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

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from project root so /models/... and /tools/web_mobile_test/... resolve
        super().__init__(*args, directory=os.path.abspath("."), **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/tools/web_mobile_test/index.html"
        elif self.path.startswith("/models/"):
            # Route /models/* to shoes-vto-ai-v2/models/*
            model_name = self.path.replace("/models/", "")
            self.path = f"/shoes-vto-ai-v2/models/{model_name}"
        return super().do_GET()

    def end_headers(self):
        # Allow cross-origin requests & WebAssembly/WebGPU headers
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def generate_self_signed_cert(cert_file, key_file):
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"ShoesVTO Mobile Test")])
        now = datetime.datetime.utcnow()
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .sign(key, hashes.SHA256())
        )

        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        return True
    except Exception as e:
        print(f"[!] Warning: Could not generate SSL certificate automatically ({e}). Falling back to HTTP.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Serve mobile browser test page.")
    parser.add_argument("--port", type=int, default=8443, help="Port to listen on")
    parser.add_argument("--http", action="store_true", help="Force plain HTTP instead of HTTPS")
    args = parser.parse_args()

    local_ip = get_local_ip()
    port = args.port

    cert_dir = Path("scratch/ssl")
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"

    use_ssl = not args.http
    if use_ssl and (not cert_file.exists() or not key_file.exists()):
        use_ssl = generate_self_signed_cert(cert_file, key_file)

    protocol = "https" if use_ssl else "http"

    print("=" * 70)
    print("  👟 Shoes VTO — Mobile Phone Live Browser Test Server")
    print("=" * 70)
    print(f"  Local IP Address: {local_ip}")
    print(f"  Server URL      : {protocol}://{local_ip}:{port}")
    print(f"  Localhost URL   : {protocol}://localhost:{port}")
    print("=" * 70)
    print("  📱 Instructions to test on your phone:")
    print(f"  1. Ensure your phone is connected to the same Wi-Fi as this PC.")
    print(f"  2. Open Safari (iOS) or Chrome (Android) on your phone.")
    print(f"  3. Navigate to: {protocol}://{local_ip}:{port}")
    if use_ssl:
        print("  4. If prompted with 'Your connection is not private', tap 'Advanced' -> 'Proceed'.")
    print("  5. Tap 'Allow' for camera access -> Real-time AR detection starts!")
    print("=" * 70)
    print("  Press Ctrl+C to stop the server.\n")

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        if use_ssl:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=str(cert_file), keyfile=str(key_file))
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()

