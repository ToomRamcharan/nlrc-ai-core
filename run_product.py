import http.server
import socketserver
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PORT = 3000
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

if __name__ == "__main__":
    while PORT < 3020:
        try:
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print("=" * 65)
                print(f"[+] NLRC AI Production Frontend is LIVE!")
                print(f"[+] Landing Page : http://localhost:{PORT}")
                print(f"[+] Chat & Model : http://localhost:{PORT}/chat.html")
                print("=" * 65)
                sys.stdout.flush()
                httpd.serve_forever()
        except OSError:
            PORT += 1
