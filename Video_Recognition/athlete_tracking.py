import cv2
from collections import deque

class AthleteTracker:
    def __init__(self, max_history=30):
        self.tracks = []
        self.max_history = max_history

    def update(self, detections):
        updated_tracks = {}
        for idx, detection in enumerate(detections):
            track = self._match_or_create_track(detection)
            if isinstance(detection, (list, tuple)) and len(detection) == 4:
                updated_tracks[idx] = (detection[0], detection[1], detection[2], detection[3])  # x, y, w, h
            else:
                updated_tracks[idx] = (0, 0, 0, 0)  # Standardwerte für ungültige Detektionen
        self.tracks = list(updated_tracks.values())
        return updated_tracks

    def _match_or_create_track(self, detection):
        for track in self.tracks:
            if self._is_match(track, detection):
                track['history'].append(detection)
                if len(track['history']) > self.max_history:
                    track['history'].popleft()
                return track
        new_track = {'id': len(self.tracks), 'history': deque([detection], maxlen=self.max_history)}
        return new_track

    def _is_match(self, track, detection, threshold=50):
        last_position = track['history'][-1]
        distance = ((last_position[0] - detection[0]) ** 2 + (last_position[1] - detection[1]) ** 2) ** 0.5
        return distance < threshold
