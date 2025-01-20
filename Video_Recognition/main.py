import cv2
import logging
import os
from athlete_tracking import AthleteTracker
from ball_tracking_system import BallTracker
from court_line_analysis import CourtLineDetector
from mini_court import MiniCourt
from utils import save_video
import constants


def convert_svo_to_mp4(input_svo_path, output_mp4_path):
    cap = cv2.VideoCapture(input_svo_path)
    if not cap.isOpened():
        raise ValueError(f"Fehler beim Öffnen der SVO-Datei: {input_svo_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (frame_width // 2, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[:, :frame_width // 2]
        out.write(cropped_frame)

    cap.release()
    out.release()
    print(f"Konvertierung abgeschlossen: {output_mp4_path}")


# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(video_path, output_path):
    if video_path.endswith(".svo"):
        converted_video_path = video_path.replace(".svo", ".mp4")
        logging.info("Konvertiere SVO zu MP4...")
        convert_svo_to_mp4(video_path, converted_video_path)
        video_path = converted_video_path

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    logging.info("Initialisierung der Komponenten")

    video_frames = []
    cap = cv2.VideoCapture(video_path)
    max_empty_frames = 50
    empty_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            empty_frame_count += 1
            if empty_frame_count > max_empty_frames:
                logging.error("Zu viele leere oder beschädigte Frames, Verarbeitung wird abgebrochen.")
                break
            logging.warning("Leeres oder beschädigtes Frame erkannt, wird übersprungen.")
            continue
        video_frames.append(frame)
    cap.release()

    athlete_tracker = AthleteTracker()
    ball_tracker = BallTracker()
    court_detector = CourtLineDetector()
    mini_court = MiniCourt(video_frames[0])

    if not video_frames:
        logging.error("Fehler: Das Video konnte nicht geöffnet werden oder enthält keine Frames.")
        return

    logging.info("Starte die Videoverarbeitung")

    player_detections = athlete_tracker.update(video_frames)
    formatted_player_detections = {i: {"bbox": detection} if isinstance(detection, tuple) else {"bbox": (0, 0, 0, 0)}
                                   for i, detection in player_detections.items()}

    ball_detections = ball_tracker.update(video_frames)
    formatted_ball_detections = {i: (detection[0], detection[1]) for i, detection in ball_detections.items()}

    court_keypoints = court_detector.detect_lines(video_frames[0])

    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        formatted_player_detections, formatted_ball_detections, court_keypoints)

    output_video_frames = mini_court.draw_mini_court(video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections,
                                                               color=constants.COLOR_BALL)

    save_video(output_video_frames, output_path, constants.FPS)
    logging.info("Speicherung abgeschlossen: %s", output_path)


if __name__ == "__main__":
    video_input_path = r'/Videos/Schlieren_20230913/schlieren1.svo'
    video_output_path = r'/output/processed_video.mp4'
    main(video_input_path, video_output_path)
