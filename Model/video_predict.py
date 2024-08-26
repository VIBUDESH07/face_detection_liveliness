
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np
import argparse

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 
    bbox = face_detector([img])[0]
    
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)   
    
    return bbox, label, score

if __name__ == "__main__":
    # parsing arguments
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue
    
    p = argparse.ArgumentParser(
        description="Spoofing attack detection on videostream")
    p.add_argument("--input", "-i", type=str, default=None, 
                   help="Path to video for predictions")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Path to save processed video")
    p.add_argument("--model_path", "-m", type=str, 
                   default="saved_models/AntiSpoofing_bin_1.5_128.onnx", 
                   help="Path to ONNX model")
    p.add_argument("--threshold", "-t", type=check_zero_to_one, default=0.5, 
                   help="real face probability threshold above which the prediction is considered true")
    args = p.parse_args()
    
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof(args.model_path)

    # Create a video capture object
    if args.input:  # file
        vid_capture = cv2.VideoCapture(args.input)
    else:           # webcam
        vid_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    print('Frame size  :', frame_size)

    if not vid_capture.isOpened():
        print("Error opening a video stream")
    # Reading fps and frame rate
    else:
        fps = vid_capture.get(5)    # Get information about frame rate
        print('Frame rate  : ', fps, 'FPS')
        if fps == 0:
            fps = 24
        # frame_count = vid_capture.get(7)    # Get the number of frames
        # print('Frames count: ', frame_count) 
    
    # videowriter
    if args.output: 
        output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    print("Video is processed. Press 'Q' or 'Esc' to quit")
    
    # process frames
    rec_width = max(1, int(frame_width/240))
    txt_offset = int(frame_height/50)
    txt_width = max(1, int(frame_width/480))
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret:
            # predict score of Live face
            pred = make_prediction(frame, face_detector, anti_spoof)
            # if face is detected
            if pred is not None:
                (x1, y1, x2, y2), label, score = pred
                if label == 0:
                    if score > args.threshold:
                        res_text = "REAL      {:.2f}".format(score)
                        color = COLOR_REAL
                    else: 
                        res_text = "unknown"
                        color = COLOR_UNKNOWN
                else:
                    res_text = "FAKE      {:.2f}".format(score)
                    color = COLOR_FAKE
                    
                # draw bbox with label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, rec_width)
                cv2.putText(frame, res_text, (x1, y1-txt_offset), 
                            cv2.FONT_HERSHEY_COMPLEX, (x2-x1)/250, color, txt_width)
            
            if args.output:
                output.write(frame)
            
            # if video captured from webcam
            if not args.input:
                cv2.imshow('Face AntiSpoofing', frame)
                key = cv2.waitKey(20)
                if (key == ord('q')) or key == 27:
                    break
        else:
            print("Streaming is Off")
            break

    # Release the video capture and writer objects
    vid_capture.release()
    output.release() 
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

app = Flask(__name__)
CORS(app)

# Load models
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# Function to decode base64 image
def decode_image(img_str):
    img_bytes = base64.b64decode(img_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# Function for cropping and prediction
def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img, y1-y, int(l*bbox_inc-y2+y), x1-x, int(l*bbox_inc)-x2+x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img])[0]
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None
    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)
    return bbox, label, score

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_str = data['image']
    img = decode_image(img_str)
    result = make_prediction(img, face_detector, anti_spoof)
    if result is not None:
        bbox, label, score = result
        response = {
            'label': 'REAL' if label == 0 else 'FAKE',
            'score': float(score),
            'bbox': bbox.tolist(),
        }
    else:
        response = {
            'label': 'No face detected',
            'score': 0.0,
            'bbox': [],
        }
    return jsonify(response)

# Face matching
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
orb = cv2.ORB_create()
reference_img = cv2.imread('VIBU.jpg')
gray_reference = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
reference_faces = face_cascade.detectMultiScale(gray_reference, scaleFactor=1.1, minNeighbors=5)

if len(reference_faces) == 0:
    raise ValueError("No face detected in the reference image.")

(x, y, w, h) = reference_faces[0]
reference_face = gray_reference[y:y+h, x:x+w]
kp1, des1 = orb.detectAndCompute(reference_face, None)

if des1 is None:
    raise ValueError("No descriptors found in the reference image.")

@app.route('/match', methods=['POST'])
def match():
    data = request.json
    img_data = data['image']
    img = base64.b64decode(img_data)
    np_img = np.frombuffer(img, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    match_found = False
    for (x, y, w, h) in faces:
        detected_face = frame[y:y+h, x:x+w]
        kp2, des2 = orb.detectAndCompute(detected_face, None)
        if des2 is None:
            continue
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 50]
        if len(good_matches) > 10:
            match_found = True
            break
    return jsonify({'match': match_found})

if __name__ == '__main__':
    app.run(debug=True)
"""