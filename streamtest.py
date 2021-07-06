import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from Tracking import *
capture = None


class CamHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary'
            )
            self.end_headers()

            while True:
                
                try:
                    imageFrame, nostream = getFrames(capture)

                    if nostream == True:
                        print('No Frame')
                        break
                    
                    img = Publish(client,imageFrame)
                    #s img = cv2.cvtColor(img, cv2.CV_GRAY2RGB)

                    img_str = cv2.imencode('.jpg', img)[1].tostring()

                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(img_str))
                    self.end_headers()

                    self.wfile.write(img_str)
                    self.wfile.write(b"\r\n--jpgboundary\r\n")

                except KeyboardInterrupt:
                    self.wfile.write(b"\r\n--jpgboundary--\r\n")
                    break
                except BrokenPipeError:
                    continue
            return

        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><head></head><body>')
            self.wfile.write(b'<img src="http://10.90.103.150:8081/cam.mjpg"/>')
            self.wfile.write(b'</body></html>')
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def main():
    global client
    client = connect_mqtt()
    client.loop_start()

    global capture
    capture = cv.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    global img
    try:
        server = ThreadedHTTPServer(('10.90.103.150', 8081), CamHandler)
        print("server started at http://10.90.103.150:8081/cam.html")
        server.serve_forever()
    except KeyboardInterrupt:
        capture.release()
        server.socket.close()
    client.disconnect()
    client.loop_stop()

if __name__ == '__main__':
    main()
    
