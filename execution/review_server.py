"""
review_server.py — Custom HTTP server for Dataset Review Tools.
Serves static files from project root and handles API requests to save
review decisions directly to .tmp/outliers_review.csv and .tmp/duplicates_review.csv.
"""

import http.server
import socketserver
import json
import os
import pandas as pd

PORT = 8765
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ReviewRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        if self.path == '/api/save_outliers':
            try:
                data = json.loads(body)  # list of dicts: [{filename, action}, ...]
                outliers_path = os.path.join(PROJECT_ROOT, ".tmp", "outliers_review.csv")

                if os.path.exists(outliers_path):
                    df = pd.read_csv(outliers_path)
                    decisions = {item['filename']: item['action'] for item in data}
                    df['action'] = df['filename'].map(decisions).fillna(df['action'])
                    df.to_csv(outliers_path, index=False, encoding='utf-8-sig')

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Saved to .tmp/outliers_review.csv"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif self.path == '/api/save_duplicates':
            try:
                data = json.loads(body)  # list of dicts: [{dup_cluster_id, action}, ...]
                dups_path = os.path.join(PROJECT_ROOT, ".tmp", "duplicates_review.csv")

                if os.path.exists(dups_path):
                    df = pd.read_csv(dups_path)
                    cluster_decisions = {int(item['dup_cluster_id']): item['action'] for item in data}
                    df['action'] = df['dup_cluster_id'].map(cluster_decisions).fillna(df['action'])
                    df.to_csv(dups_path, index=False, encoding='utf-8-sig')

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Saved to .tmp/duplicates_review.csv"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ReviewRequestHandler) as httpd:
        print(f"Review Server running on http://localhost:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
