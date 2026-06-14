"""
detect.py – Real-time helmet compliance detection via webcam / video file.
Run:  python src/detect.py
Press 'q' to quit.
"""

import os
import sys
import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# PyTorch ≥ 2.6 requires explicit safe-globals allowance
try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    pass

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils import draw_boxes, draw_hud
from compliance import check_helmet_compliance, ViolationLogger


# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH      = "outputs/helmet_detection_custom/weights/best.pt"
FALLBACK_MODEL  = "yolov8n.pt"
CONF_THRESHOLD  = 0.15          # Detection confidence (lower = more sensitive)
LOG_COOLDOWN    = 5             # Seconds between violation log entries
WINDOW_TITLE    = "Smart Helmet Detection  |  Press Q to quit"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def classify(cls_name: str):
    """Return one of: 'person', 'violation', 'compliant', or None."""
    n = cls_name.lower()
    if "person" in n or "human" in n:
        return "person"
    if ("no" in n and ("helmet" in n or "hardhat" in n or "hard-hat" in n)) or "without" in n:
        return "violation"
    if "helmet" in n or "hardhat" in n or "hard-hat" in n or "with helmet" in n:
        return "compliant"
    return None


def open_camera(source=0):
    """Try multiple backends to open a camera reliably on Windows."""
    for idx in ([source] if source != 0 else [0, 1]):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"  Camera opened on index={idx}, backend={backend}")
                    return cap
                cap.release()
    return None


# ─── Main ────────────────────────────────────────────────────────────────────

def run(source=0):
    # Load model (custom > fallback)
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Classes: {model.names}")

    print("\n[INFO] Opening camera …")
    cap = open_camera(source)

    if cap is None:
        print("\n" + "=" * 60)
        print("❌  Cannot access webcam!  Possible fixes:")
        print("  1. Close any app using the camera (Zoom, Teams, Chrome).")
        print("  2. Windows Settings → Privacy → Camera")
        print('     → Enable "Allow desktop apps to access your camera"')
        print("  3. Try connecting an external USB webcam.")
        print("=" * 60 + "\n")
        input("Press Enter to exit …")
        return

    logger        = ViolationLogger()
    last_log_time = 0

    print("\n[INFO] Detection running. Press Q in the video window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — stream ended.")
            break

        results = model(frame, verbose=False, conf=CONF_THRESHOLD)

        persons     = []
        compliant   = []
        non_compliant = []
        helmets     = []

        for r in results:
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                bbox     = box.xyxy[0].cpu().numpy()
                kind     = classify(cls_name)

                if kind == "person":
                    persons.append(bbox)
                elif kind == "violation":
                    non_compliant.append(bbox)
                elif kind == "compliant":
                    compliant.append(bbox)
                    helmets.append(bbox)

        # Spatial fallback: when model only outputs person boxes (no head labels)
        if persons and not compliant and not non_compliant:
            compliant, non_compliant = check_helmet_compliance(persons, helmets)

        total = len(persons) if persons else (len(compliant) + len(non_compliant))

        # Draw detections and HUD
        frame = draw_boxes(frame, compliant, non_compliant, helmets)
        frame = draw_hud(frame, total, len(compliant), len(non_compliant))

        # Log violation with screenshot (throttled by cooldown)
        if non_compliant:
            now = time.time()
            if now - last_log_time > LOG_COOLDOWN:
                logger.log_violation(frame, len(non_compliant))
                last_log_time = now

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Detection stopped. Logs saved in: logs/")


if __name__ == "__main__":
    run(source=0)
