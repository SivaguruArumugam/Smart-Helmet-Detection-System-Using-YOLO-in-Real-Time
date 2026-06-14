import cv2
import numpy as np


# ─── Color palette (BGR) ─────────────────────────────────────────────────────
GREEN  = (34, 197, 94)    # Safe  – with helmet
RED    = (239, 68, 68)    # Violation – no helmet
YELLOW = (250, 204, 21)   # Unknown / person only
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DARK   = (15, 23, 42)


def _draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=2):
    """Draw a rounded-corner rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Draw straight lines
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    # Draw corners
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def _label_box(img, bbox, label, color):
    """Draw a filled label tag above a bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness  = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 5

    tag_x1 = x1
    tag_y1 = max(0, y1 - th - 2 * pad)
    tag_x2 = x1 + tw + 2 * pad
    tag_y2 = y1

    # Filled tag background
    cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    cv2.putText(img, label,
                (tag_x1 + pad, tag_y2 - pad),
                font, font_scale, WHITE, thickness, cv2.LINE_AA)


def draw_boxes(frame, compliant, non_compliant, helmets=None):
    """
    Draw professional bounding boxes on the frame.
    - compliant     : list of bbox arrays for workers WITH helmets  → green
    - non_compliant : list of bbox arrays for workers WITHOUT helmets → red
    - helmets       : (optional) list of helmet bbox arrays (not drawn separately)
    """
    output = frame.copy()

    for bbox in compliant:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        _draw_rounded_rect(output, (x1, y1), (x2, y2), GREEN, radius=6, thickness=2)
        _label_box(output, bbox, "✓ HELMET", GREEN)

    for bbox in non_compliant:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        _draw_rounded_rect(output, (x1, y1), (x2, y2), RED, radius=6, thickness=2)
        _label_box(output, bbox, "✗ NO HELMET", RED)

    return output


def draw_hud(frame, total, safe, violations):
    """
    Draws a professional HUD (Heads-Up Display) overlay on the frame
    showing real-time statistics.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark banner at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font  = cv2.FONT_HERSHEY_DUPLEX
    small = 0.5
    large = 0.6

    # Title
    cv2.putText(frame, "SMART HELMET DETECTION SYSTEM",
                (10, 30), font, 0.55, WHITE, 1, cv2.LINE_AA)

    # Stats on the right
    stats = [
        (f"Total: {total}", WHITE),
        (f"Safe: {safe}", GREEN),
        (f"Violations: {violations}", RED if violations > 0 else WHITE),
    ]
    x_offset = w - 50
    for text, color in reversed(stats):
        (tw, _), _ = cv2.getTextSize(text, font, small, 1)
        x_offset -= tw + 20
        cv2.putText(frame, text, (x_offset, 30), font, small, color, 1, cv2.LINE_AA)

    # Violation warning banner at bottom
    if violations > 0:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 45), (w, h), RED, -1)
        cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
        warn = f"⚠  SAFETY VIOLATION DETECTED — {violations} Worker(s) Without Helmet!"
        (tw, _), _ = cv2.getTextSize(warn, font, large, 1)
        cv2.putText(frame, warn, ((w - tw) // 2, h - 14),
                    font, large, WHITE, 1, cv2.LINE_AA)

    return frame
