FPS = 30
DOUBLE_LINE_WIDTH = 8.23  # Meter

# Mini-Court Maße in Metern
MINI_COURT_WIDTH = 23.77  # Official tennis court width
MINI_COURT_HEIGHT = 10.97  # Official tennis court height

# Farben für Visualisierung
COLOR_PLAYER_1 = (0, 255, 0)
COLOR_PLAYER_2 = (255, 0, 0)
COLOR_BALL = (0, 255, 255)
COLOR_LINES = (255, 255, 255)

# Modellpfade
YOLO_MODEL_PATH = "yolo/yolov8x"
BALL_MODEL_PATH = "yolo/models/yolo5_last.pt"
COURT_MODEL_PATH = "yolo/models/keypoints_model.pth"

# Stub-Pfade
PLAYER_DETECTIONS_STUB = "yolo/tracker_stubs/player_detections.pkl"
BALL_DETECTIONS_STUB = "yolo/tracker_stubs/ball_detections.pkl"
