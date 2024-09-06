from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import cv2
import numpy as np
from deepface import DeepFace

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

# Anti-spoofing prediction
def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
    y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img, y1 - y, int(l * bbox_inc - y2 + y), x1 - x, int(l * bbox_inc) - x2 + x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
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

if __name__ == "__main__":
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

    # Load pre-registered image for matching
    image_path_1 = "VIBU.jpg"
    known_face_img = cv2.imread(image_path_1)

    vid_capture = cv2.VideoCapture(0)

    print("Video is processed. Press 'Q' or 'Esc' to quit")
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret:
            pred = make_prediction(frame, face_detector, anti_spoof)
            if pred is not None:
                (x1, y1, x2, y2), label, score = pred
                if label == 0 and score > 0.5:  # Real face
                    res_text = "REAL {:.2f}".format(score)
                    color = COLOR_REAL

                    # Crop the face from the bounding box
                    face_crop = frame[y1:y2, x1:x2]

                    # Use DeepFace to verify against the known face
                    try:
                        result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")  # You can change the model
                        if result["verified"]:
                            res_text += " | MATCHED"
                        else:
                            res_text += " | NOT MATCHED"
                    except Exception as e:
                        res_text += f" | ERROR: {e}"
                else:  # Fake face
                    res_text = "FAKE {:.2f}".format(score)
                    color = COLOR_FAKE

                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, res_text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            # Show the frame
            cv2.imshow('Face Anti-Spoofing & Matching', frame)

            # Press 'q' or 'Esc' to exit
            if cv2.waitKey(20) & 0xFF in [27, ord('q')]:
                break
        else:
            break

    vid_capture.release()
    cv2.destroyAllWindows()
