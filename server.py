#!/usr/bin/python3

# Mostly copied from https://picamera.readthedocs.io/en/release-1.13/recipes2.html
# Run this script, then point a web browser at http:<this-ip-address>:8000
# Note: needs simplejpeg to be installed (pip3 install simplejpeg).
import process

import io, logging, socketserver, cv2, asyncio, random, threading, time
from http import server
from threading import Condition
import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="810" height="1080" />
</body>
</html>
"""

frames = []
batch_size = 8

buffers = []

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            stream_thread = threading.Thread(target=self.stream)
            record_thread = threading.Thread(target=self.record)
            stream_thread.start()
            record_thread.start()

            asyncio.run(self.processing())
        else:
            self.send_error(404)
            self.end_headers()
    
    async def processing(self):
        while True:
            if len(frames) < batch_size:
                continue

            images = frames[0:batch_size]
            del frames[0:batch_size]

            result, owns = await process.processing(images)
            images = [self.buffer_to_img(img) for img in images]

            for i in range(batch_size):
                self.detection_visualize(images[i], result[0][i], owns)
                self.pose_visualize(images[i], result[1][i])


                byte_img = cv2.imencode('.jpg', images[i])[1].tobytes()
                buffers.append(byte_img)


    def record(self):
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame

            time.sleep(0.15)
            frames.append(frame)

    def buffer_to_img(self, image):
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
            
    def detection_visualize(self, frame, data, owns):
        for object in data:
            x1, y1, x2, y2 = map(int, list(object['box'].values()))
            box_color = (0, 0, 255)

            text_color = (0, 255, 0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 4)
            cv2.putText(frame, f"Class: {object['name']}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Conf: {int(object['confidence']*100)}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

            if 'track_id' in object.keys():
                for human_id in owns:
                    for k in owns[human_id]:
                        if k[0] == object['track_id']:
                            cv2.putText(frame, f"possessed by {human_id}" if owns[human_id][k] else "warning!", (x1,y1-70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        return frame

    def pose_visualize(self, frame, data):
        drawLines = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (11, 5),
            (12, 6),
            (11, 12),
            (13, 11),
            (13, 15),
            (14, 12),
            (14, 16),
        ]

        for human in data:
            keypoints = human['keypoints']

            x_pos = list(map(int, keypoints['x']))
            y_pos = list(map(int, keypoints['y']))

            for dot in drawLines:
                color = (0, 0, 0)
            
                a, b = dot
                if (x_pos[a] == 0 and y_pos[a] == 0) or (x_pos[b] == 0 and y_pos[b] == 0):
                    continue

                if a <= 2:
                    color = (255, 0, 0)
                elif a <= 4:
                    color = (0, 255, 0)
                elif a <= 8:
                    color = (0, 255, 255)
                elif a <= 12:
                    color = (0, 255, 0)
                elif a <= 16:
                    color = (0, 0, 255)
                    
                cv2.line(frame, (x_pos[a], y_pos[a]), (x_pos[b], y_pos[b]), color, thickness=3)
            for i in range(len(x_pos)):
                cv2.circle(frame, (x_pos[i], y_pos[i]), radius=3, color=(0, 0, 0), thickness=5)
        return frame


            
    def stream(self):
        try:
            while True:
                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                
                if len(buffers) < 1:
                    continue
                time.sleep(0.1)

                frame = buffers[0]
                del buffers[0]

                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(frame))
                self.end_headers()

                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
        except Exception as e:
            logging.warning(
                'Removed streaming client %s: %s',
                self.client_address, str(e))



class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (810, 1080)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    picam2.stop_recording()