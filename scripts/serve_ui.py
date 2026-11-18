"""Serve the evaluator UI (Laptop 2):  python scripts/serve_ui.py [--port 5173]"""
import argparse
import functools
import http.server
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5173)
ap.add_argument("--host", default="0.0.0.0")
a = ap.parse_args()
root = Path(__file__).resolve().parents[1] / "frontend"
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
print(f"PathSnap UI on http://localhost:{a.port}  (append ?api=http://<laptop1-ip>:8000)")
http.server.ThreadingHTTPServer((a.host, a.port), handler).serve_forever()
