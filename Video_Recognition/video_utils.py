import cv2

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Fehler beim Öffnen der Videodatei: {video_path}")
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.output_writer = None

    def has_frames(self):
        return self.cap.isOpened()

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return None
        return frame

    def initialize_writer(self, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

    def draw_detections(self, frame, players, ball):
        for player in players:
            cv2.rectangle(frame, player[0], player[1], (0, 255, 0), 2)
        if ball:
            cv2.circle(frame, ball, 5, (0, 0, 255), -1)

    def write_frame(self, frame):
        if self.output_writer is not None:
            self.output_writer.write(frame)

    def save_output(self, output_path):
        if self.output_writer is not None:
            self.output_writer.release()
        self.cap.release()
        print(f"Video gespeichert unter: {output_path}")
