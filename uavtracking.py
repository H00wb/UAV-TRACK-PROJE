from ultralytics import YOLO
import cv2
import torch
device ='cuda'
class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FPS, 120)  # Set FPS

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

class ObjectTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def track(self, img):
        results = self.model.track(source=img, show=True,conf=0.74)
        return results

def draw_boxes(frame, results):
    box_list = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        box_list.append(boxes.xyxy)
        print(box_list)

def main():
    cam = Camera()
    tracker = ObjectTracker("best.pt")

    while True:
        frame = cam.get_frame()
        results = tracker.track(frame)
        draw_boxes(frame, results)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()