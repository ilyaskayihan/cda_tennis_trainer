import cv2
import numpy as np
import logging
from ultralytics import YOLO

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_objects(video_path, output_path, model_path='yolov8m.pt'):
    logging.info("Lade das YOLO-Modell...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO-Vorhersagen ausführen
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame)
        logging.info(f'Verfügbare Klassen im Modell: {model.names}')
        if 'sports ball' not in model.names.values():
            logging.warning('Ball-Klasse nicht im Modell vorhanden. Eventuell wird der Ball nicht erkannt.')
        for result in results:
            for box in result.boxes.data.tolist():
                if len(box) < 6:
                    continue
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = float(box[4])
                class_id = int(box[5])
                logging.info(f"Erkannt: Klasse={model.names[class_id]}, Konfidenz={confidence:.2f}")
                if confidence > 0.02:
                    if model.names[class_id] == 'person':
                        color = (0, 255, 0)
                    elif model.names[class_id] == 'sports ball':
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{model.names[class_id]} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imwrite(f'output/frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg', frame)
        out.write(frame)

    cap.release()
    out.release()
    logging.info("Objekterkennung abgeschlossen. Ausgabe gespeichert unter: %s", output_path)


if __name__ == "__main__":
    input_video = r'C:\\Users\\ilyas\\OneDrive\\Desktop\\cda2_versuch3000\\output\\processed_video.mp4'
    output_video = r'C:\\Users\\ilyas\\OneDrive\\Desktop\\cda2_versuch3000\\output\\detected_video.mp4'
    detect_objects(input_video, output_video)
