"""All OpenCV drawing helpers: eye contours, HUD panel, alert overlays."""

import cv2
import numpy as np

from . import config as cfg


def draw_eye_contour(frame, landmarks, eye_indices, img_w, img_h, closed: bool):
    """Draw a filled + outlined polygon around an eye."""
    pts = np.array(
        [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices],
        dtype=np.int32
    )
    color = cfg.RED if closed else cfg.GREEN
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (*color, 40))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, tuple(pt), 2, color, -1, lineType=cv2.LINE_AA)


def draw_iris_dots(frame, landmarks, iris_indices, img_w, img_h):
    """Draw small dots on iris landmarks."""
    for i in iris_indices:
        x = int(landmarks[i].x * img_w)
        y = int(landmarks[i].y * img_h)
        cv2.circle(frame, (x, y), 2, cfg.BLUE, -1, lineType=cv2.LINE_AA)


def draw_ear_bar(frame, ear_val: float, threshold: float, x: int, y: int, w: int = 160, h: int = 12):
    """Draw a horizontal EAR progress bar."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 60), -1)
    fill_w = int(np.clip(ear_val / 0.45, 0, 1) * w)
    bar_color = cfg.RED if ear_val < threshold else (cfg.ORANGE if ear_val < threshold + 0.05 else cfg.GREEN)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), bar_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 90), 1)
    thresh_x = x + int(np.clip(threshold / 0.45, 0, 1) * w)
    cv2.line(frame, (thresh_x, y - 2), (thresh_x, y + h + 2), cfg.WHITE, 1)


def draw_hud(frame, state: dict):
    """Draw the full HUD panel on the right side of the frame."""
    h, w = frame.shape[:2]
    panel_w = 230
    panel_x = w - panel_w - 10
    top_y   = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 8, top_y), (w - 5, h - 10), cfg.DARK_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (panel_x - 8, top_y), (w - 5, h - 10), (60, 60, 80), 1)

    y = top_y + 20
    line_h = 26

    cv2.putText(frame, "DROWSINESS MONITOR", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 200), 1, cv2.LINE_AA)
    y += line_h - 4
    cv2.line(frame, (panel_x - 4, y), (w - 8, y), (60, 60, 90), 1)
    y += 14

    ear_color = cfg.RED if state["ear"] < state["ear_thresh"] else cfg.GREEN
    cv2.putText(frame, "Eye Aspect Ratio", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 170), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(frame, f"{state['ear']:.3f}", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, ear_color, 2, cv2.LINE_AA)
    y += 8
    draw_ear_bar(frame, state["ear"], state["ear_thresh"], panel_x, y, w=panel_w - 20)
    y += 22

    cv2.putText(frame, "Drowsy Frames", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 170), 1, cv2.LINE_AA)
    y += 18
    count_color = cfg.RED if state["counter"] > state["frame_thresh"] // 2 else cfg.WHITE
    cv2.putText(frame, f"{state['counter']}  /  {state['frame_thresh']}", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, count_color, 2, cv2.LINE_AA)
    y += line_h

    cv2.putText(frame, f"FPS: {state['fps']:.0f}", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 150), 1, cv2.LINE_AA)
    y += line_h - 4

    cv2.line(frame, (panel_x - 4, y), (w - 8, y), (60, 60, 90), 1)
    y += 14
    cv2.putText(frame, f"EAR thresh : {state['ear_thresh']:.2f}  (+/-)", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 150), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(frame, f"Frame limit: {state['frame_thresh']}  ([/])", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 150), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(frame, "R: reset   Q: quit (terminal)", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 130), 1, cv2.LINE_AA)

    y = h - 30
    cv2.putText(frame, f"Total alerts: {state['alert_count']}", (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 170), 1, cv2.LINE_AA)


def draw_alert_overlay(frame, level: str):
    """
    Draw a colored border + large alert text over the frame.
    level: 'warn' | 'danger'
    """
    h, w = frame.shape[:2]
    if level == "danger":
        border_color = cfg.RED
        text1, text2 = "!! WAKE UP !!", "PULL OVER SAFELY"
        text_color = cfg.RED
    else:
        border_color = cfg.ORANGE
        text1, text2 = "DROWSINESS DETECTED", "Stay alert!"
        text_color = cfg.ORANGE

    thickness = 8
    cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness)

    cx = w // 2
    for text, scale, dy in [(text1, 1.1, -24), (text2, 0.65, 30)]:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        tx = cx - tw // 2
        ty = h // 2 + dy
        cv2.putText(frame, text, (tx + 2, ty + 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, cfg.BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, scale, text_color, 2, cv2.LINE_AA)


def draw_no_face(frame):
    h, w = frame.shape[:2]
    cv2.putText(frame, "No face detected", (w // 2 - 100, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 110), 1, cv2.LINE_AA)
