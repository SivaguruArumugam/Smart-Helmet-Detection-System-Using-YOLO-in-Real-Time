import os
import cv2
import csv
from datetime import datetime


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    box1_area  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def check_helmet_compliance(persons, helmets, iou_threshold=0.10):
    """
    Spatial matching: matches helmets to the upper 30% (head region) of each person.
    Returns (compliant_list, non_compliant_list).
    """
    compliant     = []
    non_compliant = []

    for pb in persons:
        # Head region = upper 30 % of the person bounding box
        head = [pb[0], pb[1], pb[2], pb[1] + (pb[3] - pb[1]) * 0.30]
        has_helmet = any(calculate_iou(head, hb) > iou_threshold for hb in helmets)
        (compliant if has_helmet else non_compliant).append(pb)

    return compliant, non_compliant


class ViolationLogger:
    """
    Logs safety violations to a CSV file and saves screenshot evidence.
    """

    def __init__(self, log_dir="logs", screenshot_dir="screenshots"):
        self.log_dir        = log_dir
        self.screenshot_dir = screenshot_dir
        os.makedirs(log_dir,        exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, "violations.csv")

        # Create CSV with headers if it does not exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as f:
                csv.writer(f).writerow(
                    ["Date", "Time", "Violations_Count", "Screenshot_Path"]
                )

    def log_violation(self, frame, violations_count):
        """Save evidence screenshot and write a row to the CSV log."""
        if violations_count == 0:
            return

        now       = datetime.now()
        date_str  = now.strftime("%Y-%m-%d")
        time_str  = now.strftime("%H:%M:%S")
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        filename = f"violation_{timestamp}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)
        cv2.imwrite(filepath, frame)

        with open(self.log_file, mode="a", newline="") as f:
            csv.writer(f).writerow([date_str, time_str, violations_count, filepath])

        print(f"[ALERT] {violations_count} violation(s) at {time_str}. Evidence → {filepath}")
