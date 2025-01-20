import cv2
import numpy as np


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def non_max_suppression(boxes, scores, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, threshold)
    return [boxes[i[0]] for i in indices]


def draw_bounding_boxes(frame, detections):
    for det in detections:
        x, y, w, h = det
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


def filter_detections(detections, confidence_threshold):
    return [det for det in detections if det['confidence'] >= confidence_threshold]
