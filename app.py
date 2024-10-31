from flask import Flask, request, render_template, send_file, Response, jsonify
from werkzeug.utils import secure_filename
import io
from flask_cors import CORS 
from ultralytics import YOLO
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import cv2
import os
import base64
import io
from io import BytesIO

load_dotenv()

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
model = YOLO(r"object_detection/yolov10b.pt")
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]
    frame = np.frombuffer(base64.b64decode(frame_data), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    results = model(frame)
    annotaded_frame = results[0].plot()

    _, buffer = cv2.imencode('.jpg', annotaded_frame)
    image = Image.open(BytesIO(buffer))

    img_io = BytesIO()
    image.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

# def process_frame(image):
#     results = model.predict(image, conf=0.5)
#     for result in results:
#         for box in result.boxes:
#             cv2.rectangle(image,
#                           (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
#                           (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
#                           (0,255,0),2)
#             cv2.putText(image, f"{result.names[int(box.cls[0])]}",
#                         int(box.xyxy[0][0], int(box.xyxy[0][1]) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
#     return image
@app.route('/object_detection/', methods=['POST'])
def object_detection():
    file = request.files['image'].read()
    np_img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    processed_img = process_frame(img)

    _, buffer = cv2.imencode('.jpg', processed_img)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/jpeg')

class Detection:
    def __init__(self):
        #download weights from here:https://github.com/ultralytics/ultralytics and change the path
        self.model = YOLO(r"object_detection/yolov10b.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)

        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img


detection = Detection()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (512, 512))
        img = detection.detect_from_image(img)
        output = Image.fromarray(img)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        os.remove(file_path)
        return send_file(buf, mimetype='image/png')



# @app.route('/object-detection/', methods=['POST'])
# def apply_detection():
#     data = request.get_json()
#     if 'image' not in data:
#         return 'No image data', 400
    
#     image_data = data['image'].split(',')[1]
#     img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
#     img = np.array(img)
#     img = cv2.resize(img, (512,512))

#     img = detection.detect_from_image(img)
#     output = Image.fromarray(img)

#     buf = io.BytesIO()
#     output.save(buf, format="PNG")
#     buf.seek(0)
#     base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

#     return jsonify({'image:' f'data:image/png:base62,{base64_img}'})
    # file = request.files['image']
    # if file.filename == '':
    #     return 'No selected file'
    
    # if file:
    #     filename = secure_filename(file.filename)
    #     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     file.save(file_path)

    #     img = Image.open(file_path).convert("RGB")
    #     img = np.array(img)
    #     img = cv2.resize(img, (512, 512))
    #     img = detection.detect_from_image(img)
    #     output = Image.fromarray(img)

    #     buf = io.BytesIO()
    #     output.save(buf, format="PNG")
    #     buf.seek(0)

    #     os.remove(file_path)
    #     return send_file(buf, mimetype='image/png')

    
@app.route('/video')
def index_video():
    return render_template('video.html')

# def gen_frames():
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         frame = cv2.resize(frame,(512,512))
#         if frame is None:
#             break
#         frame = detection.detect_from_image(frame)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield ( b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def gen_frames():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise RuntimeError("Webcam não encontrada")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame = cv2.resize(frame,(512,512))
#         frame = detection.detect_from_image(frame)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield ( b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     cap.release()
        
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #port = int(os.environ.get('PORT',8080))
    app.run(host="0.0.0.0", port=8080)