import math
import cv2
import mediapipe as mp

def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba * mag_bc == 0:
        return None

    angle_rad = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle_rad)


def draw_pose(frame, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    annotated = frame.copy()
    mp_drawing.draw_landmarks(
        annotated,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    return annotated
