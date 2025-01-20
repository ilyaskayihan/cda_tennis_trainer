import cv2
import numpy as np


class CourtLineDetector:
    def __init__(self):
        self.kernel = np.ones((5, 5), np.uint8)

    def detect_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
        return lines

    def draw_lines(self, frame, lines):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dilated = cv2.dilate(gray, self.kernel, iterations=1)
        return dilated
