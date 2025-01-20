import cv2
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Fehler beim Öffnen der Videodatei: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps=30):
    if len(frames) == 0:
        raise ValueError("Keine Frames zum Speichern vorhanden.")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def measure_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def convert_pixel_distance_to_meters(pixel_distance, reference_length_pixels, reference_length_meters):
    if reference_length_pixels == 0:
        raise ValueError("Referenzlänge in Pixeln darf nicht null sein.")
    return (pixel_distance / reference_length_pixels) * reference_length_meters
