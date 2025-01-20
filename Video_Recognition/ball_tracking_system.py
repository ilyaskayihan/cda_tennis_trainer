import cv2
from collections import deque

class BallTracker:
    def __init__(self, max_history=50):
        self.track_history = deque(maxlen=max_history)

    def update(self, detections):
        updated_tracks = {}
        for idx, detection in enumerate(detections):
            if isinstance(detection, (list, tuple)) and len(detection) == 2:
                updated_tracks[idx] = (detection[0], detection[1])  # x, y
            else:
                updated_tracks[idx] = (0, 0)  # Standardwerte für ungültige Detektionen
            self.track_history.append(updated_tracks[idx])
        return updated_tracks

    def predict_next_position(self):
        if len(self.track_history) < 2:
            return None
        dx = self.track_history[-1][0] - self.track_history[-2][0]
        dy = self.track_history[-1][1] - self.track_history[-2][1]
        predicted_position = (self.track_history[-1][0] + dx, self.track_history[-1][1] + dy)
        return predicted_position

    def draw_trajectory(self, frame):
        for i in range(1, len(self.track_history)):
            cv2.line(frame, self.track_history[i - 1], self.track_history[i], (0, 0, 255), 2)
        return frame
