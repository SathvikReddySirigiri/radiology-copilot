"""Bounding box extraction and anatomical region labeling from Grad-CAM maps.

Converts heatmap arrays into clinical annotations with anatomical landmarks.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image

ANATOMY_GRID = {
    (0, 0): "Right Upper Lobe",
    (0, 1): "Superior Mediastinum",
    (0, 2): "Left Upper Lobe",
    (1, 0): "Right Mid Zone",
    (1, 1): "Cardiac Silhouette",
    (1, 2): "Left Mid Zone",
    (2, 0): "Right Lower Lobe",
    (2, 1): "Right Hemidiaphragm",
    (2, 2): "Left Lower Lobe",
}

RISK_COLORS = {
    "high": (50, 50, 220),
    "medium": (50, 165, 220),
    "low": (50, 200, 100),
}


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float
    region: str
    risk_level: str
    area_pct: float


def get_risk_level(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    if confidence >= 0.4:
        return "medium"
    return "low"


def get_anatomical_region(cx: int, cy: int, img_w: int, img_h: int) -> str:
    """Return anatomical region name for a bounding box center point."""
    col = min(int(cx / img_w * 3), 2)
    row = min(int(cy / img_h * 3), 2)
    return ANATOMY_GRID.get((row, col), "Unspecified Region")


def extract_bounding_boxes(
    cam_array: np.ndarray,
    label: str,
    confidence: float,
    threshold: float = 0.45,
    min_area_pct: float = 0.01,
    max_boxes: int = 3,
) -> List[BoundingBox]:
    """Extract bounding boxes from a Grad-CAM activation map."""
    height, width = cam_array.shape
    img_area = height * width

    cam_uint8 = (cam_array * 255).astype(np.uint8)
    _, binary = cv2.threshold(
        cam_uint8,
        int(threshold * 255),
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return []

    boxes = []
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area_pct = (box_w * box_h) / img_area

        if area_pct < min_area_pct:
            continue

        center_x = x + box_w // 2
        center_y = y + box_h // 2
        region = get_anatomical_region(center_x, center_y, width, height)
        risk = get_risk_level(confidence)

        boxes.append(
            BoundingBox(
                x1=x,
                y1=y,
                x2=x + box_w,
                y2=y + box_h,
                label=label,
                confidence=confidence,
                region=region,
                risk_level=risk,
                area_pct=area_pct,
            )
        )

    boxes.sort(key=lambda box: box.area_pct, reverse=True)
    return boxes[:max_boxes]


def draw_boxes_on_image(
    image: Image.Image,
    boxes: List[BoundingBox],
    show_region: bool = True,
    show_confidence: bool = True,
    line_width: int = 3,
) -> Image.Image:
    """Draw bounding boxes with labels on a PIL image."""
    img_rgb = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    _, img_w = img_cv.shape[:2]
    font_scale = max(0.5, img_w / 800)
    thickness = max(2, line_width)

    for box in boxes:
        color = RISK_COLORS[box.risk_level]

        cv2.rectangle(
            img_cv,
            (box.x1, box.y1),
            (box.x2, box.y2),
            color,
            thickness,
        )

        label_text = (
            f"{box.label} ({box.confidence:.0%})"
            if show_confidence
            else box.label
        )

        (text_w, text_h), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        label_y = max(box.y1 - 8, text_h + 8)

        cv2.rectangle(
            img_cv,
            (box.x1, label_y - text_h - 6),
            (box.x1 + text_w + 6, label_y + 2),
            color,
            -1,
        )

        cv2.putText(
            img_cv,
            label_text,
            (box.x1 + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

        if show_region:
            cv2.putText(
                img_cv,
                box.region,
                (box.x1 + 3, box.y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.8,
                color,
                max(1, thickness - 1),
            )

    img_result = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_result)


def annotate_xray(
    image: Image.Image,
    all_heatmaps: dict,
    pathology_labels: dict,
    selected_label: str = None,
    threshold: float = 0.45,
) -> dict:
    """Draw bounding boxes for selected or high-confidence pathology labels."""
    all_boxes = []

    if selected_label and selected_label in all_heatmaps:
        labels_to_draw = [selected_label]
    else:
        labels_to_draw = [
            label
            for label, score in pathology_labels.items()
            if score > 0.4 and label in all_heatmaps
        ]

    for label in labels_to_draw:
        heatmap_data = all_heatmaps.get(label)
        if not heatmap_data:
            continue

        cam_array = heatmap_data.get("cam_array")
        if cam_array is None:
            continue

        orig_w, orig_h = image.size
        cam_resized = cv2.resize(
            cam_array.astype(np.float32),
            (orig_w, orig_h),
        )

        score = pathology_labels.get(label, heatmap_data.get("target_score", 0))
        boxes = extract_bounding_boxes(cam_resized, label, score, threshold)
        all_boxes.extend(boxes)

    if not all_boxes:
        return {
            "annotated_image": image,
            "boxes": [],
            "summary": "No significant regions detected above threshold.",
        }

    annotated = draw_boxes_on_image(image, all_boxes)

    summary_parts = [
        f"• {box.label} ({box.confidence:.0%}) detected in {box.region}"
        for box in all_boxes
    ]

    return {
        "annotated_image": annotated,
        "boxes": all_boxes,
        "summary": "\n".join(summary_parts),
    }
