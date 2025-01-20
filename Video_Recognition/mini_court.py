import cv2
import numpy as np
import constants

class MiniCourt:
    def __init__(self, reference_frame):
        self.court_width = constants.MINI_COURT_WIDTH
        self.court_height = constants.MINI_COURT_HEIGHT
        self.frame_height, self.frame_width, _ = reference_frame.shape
        self.court_image = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_detections, ball_detections, court_keypoints):
        player_mini_court = {}
        ball_mini_court = {}

        for frame_num, players in player_detections.items():
            player_mini_court[frame_num] = {}
            for player_id, player_data in players.items():
                if isinstance(player_data, tuple):
                    bbox = player_data  # Falls es ein Tupel ist
                else:
                    bbox = player_data['bbox']
                x_ratio = bbox[0] / self.frame_width
                y_ratio = bbox[1] / self.frame_height
                player_mini_court[frame_num][player_id] = (x_ratio * self.court_width, y_ratio * self.court_height)

        for frame_num, ball_pos in ball_detections.items():
            x_ratio = ball_pos[0] / self.frame_width
            y_ratio = ball_pos[1] / self.frame_height
            ball_mini_court[frame_num] = (x_ratio * self.court_width, y_ratio * self.court_height)

        return player_mini_court, ball_mini_court

    def draw_mini_court(self, frames):
        for frame in frames:
            cv2.rectangle(frame, (50, 50), (300, 150), constants.COLOR_LINES, 2)
        return frames

    def draw_points_on_mini_court(self, frames, detections, color=constants.COLOR_PLAYER_1):
        for frame_num, positions in detections.items():
            if isinstance(positions, tuple):
                pos_list = [positions]
            else:
                pos_list = positions.values()
            for pos in pos_list:
                cv2.circle(frames[frame_num], (int(pos[0]), int(pos[1])), 5, color, -1)
        return frames
