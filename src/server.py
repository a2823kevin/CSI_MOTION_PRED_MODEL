from http.server import BaseHTTPRequestHandler
import socketserver
from urllib.parse import urlparse, parse_qs
import numpy
import cv2
import ffmpeg

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global frames
        self.send_response(200)

        url = urlparse(self.path)
        if (url.path=="/index"):
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            with open("assets/label generator/index.html", "r", encoding="utf8") as fin:
                self.wfile.write(fin.read().encode())
        
        elif (url.path=="/get_frame"):
            self.send_header("content-type", "image/png")
            self.end_headers()
            frame_idx = int(parse_qs(url.query)["idx"][0])
            cv2.imwrite("assets/label generator/tmp/tmpimg.png", frames[frame_idx])
            with open("assets/label generator/tmp/tmpimg.png", "rb") as fin:
                self.wfile.write(fin.read())
            print(len(frames))

    def do_POST(self):
        global frames
        self.send_response(200)
        self.send_header("content-type", "text/plain")
        self.end_headers()

        url = urlparse(self.path)
        if (url.path=="/upload_video"):
            video = self.rfile.read(int(self.headers["Content-Length"]))
            with open("assets/label generator/tmp/tmpvd.mp4", "wb") as fout:
                fout.write(video)

            vd_info = ffmpeg.probe("assets/label generator/tmp/tmpvd.mp4")
            output, _ = (ffmpeg.input("assets/label generator/tmp/tmpvd.mp4").output("pipe:", format="rawvideo", pix_fmt="bgr24").run(capture_stdout=True))
            frames = (numpy.frombuffer(output, numpy.uint8).reshape([-1, vd_info["streams"][0]["height"], vd_info["streams"][0]["width"], 3]))
            self.wfile.write(str(len(frames)-2).encode())

if __name__=="__main__":
    frames = None
    with socketserver.TCPServer(("", 8080), Handler) as httpd:
        httpd.serve_forever()