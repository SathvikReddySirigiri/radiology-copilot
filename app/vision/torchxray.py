"""TorchXRayVision model wrapper for chest X-ray classification.

Loads pretrained TorchXRayVision models and runs inference on input images
to produce pathology probability scores.
"""

import gc
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_MODEL = xrv.models.DenseNet(weights="densenet121-res224-all")
_MODEL.eval()

_TRANSFORM = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])

_CONFIDENCE_THRESHOLD = 0.3
_TOP_K = 5
_HEATMAP_ALPHA = 0.4

PATHOLOGY_COLORS = {
    "high": (255, 50, 50),
    "medium": (255, 165, 0),
    "low": (50, 205, 50),
}

SWITCHABLE_LABELS = [
    "Mass",
    "Nodule",
    "Consolidation",
    "Effusion",
    "Pneumothorax",
    "Atelectasis",
    "Cardiomegaly",
    "Infiltration",
]


def _format_label(label: str) -> str:
    return label.replace("_", " ").strip().title()


def _preprocess(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L")
    arr = torchvision.transforms.functional.pil_to_tensor(gray).squeeze(0).float().numpy()
    arr = xrv.datasets.normalize(arr, 255)
    if arr.ndim == 2:
        arr = arr[None, ...]
    arr = _TRANSFORM(arr)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return torch.from_numpy(arr).float()


def _resolve_target_index(target_label: str) -> tuple[str, int]:
    pathologies = xrv.datasets.default_pathologies
    normalized = target_label.strip().casefold()

    for index, pathology in enumerate(pathologies):
        if not pathology:
            continue
        if pathology.casefold() == normalized or _format_label(pathology).casefold() == normalized:
            return pathology, index

    raise ValueError(
        f"Label '{target_label}' not found in pathologies list."
    )


def _resolve_auto_target(image: Image.Image) -> tuple[str, int, float]:
    labels = get_pathology_labels(image)
    if not labels:
        raise ValueError("No pathology labels found for image.")

    top_label = next(iter(labels))
    raw_label, target_index = _resolve_target_index(top_label)
    return raw_label, target_index, labels[top_label]


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cam_min) / (cam_max - cam_min)


def _create_overlay(image: Image.Image, cam_array: np.ndarray) -> Image.Image:
    original_rgb = np.array(image.convert("RGB"))
    cam_uint8 = np.uint8(255 * _normalize_cam(cam_array))
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(
        heatmap,
        (original_rgb.shape[1], original_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay = cv2.addWeighted(original_rgb, 1.0 - _HEATMAP_ALPHA, heatmap, _HEATMAP_ALPHA, 0)
    return Image.fromarray(overlay)


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_pathology_labels(image: Image.Image) -> dict:
    """Run chest X-ray pathology inference and return the top 5 labels."""
    tensor = _preprocess(image)

    with torch.no_grad():
        scores = _MODEL(tensor).squeeze(0)

    top_values, top_indices = torch.topk(scores, k=min(_TOP_K, scores.numel()))

    labels = {}
    for value, index in zip(top_values, top_indices):
        raw_label = _MODEL.pathologies[index.item()]
        if not raw_label:
            continue
        labels[_format_label(raw_label)] = round(value.item(), 2)

    if scores.numel() > 0 and torch.all(scores < _CONFIDENCE_THRESHOLD):
        print(
            "Warning: All pathology confidences are below 0.3 — "
            "this may indicate a normal chest X-ray."
        )

    return labels


def get_heatmap(image: Image.Image, target_label: str = None) -> dict:
    """Generate a Grad-CAM heatmap overlay for a target pathology."""
    if target_label is None:
        raw_label, target_index, target_score = _resolve_auto_target(image)
    else:
        raw_label, target_index = _resolve_target_index(target_label)
        tensor = _preprocess(image)
        with torch.no_grad():
            scores = _MODEL(tensor).squeeze(0)
        target_score = round(scores[target_index].item(), 2)

    tensor = _preprocess(image)
    cam_extractor = GradCAM(model=_MODEL, target_layers=[_MODEL.features.denseblock4])
    targets = [ClassifierOutputTarget(target_index)]

    try:
        grayscale_cam = cam_extractor(input_tensor=tensor, targets=targets)
        cam_array = grayscale_cam[0].astype(np.float32)
        heatmap_overlay = _create_overlay(image, cam_array)
    finally:
        cam_extractor = None
        _cleanup_memory()

    return {
        "heatmap_overlay": heatmap_overlay,
        "target_label": _format_label(raw_label),
        "target_score": target_score,
        "cam_array": cam_array,
    }


def _get_label_score(scores: dict, label: str) -> float:
    if label in scores:
        return float(scores[label])

    label_norm = _format_label(label).lower()
    for key, value in scores.items():
        if _format_label(key).lower() == label_norm:
            return float(value)
    return 0.0


def get_all_heatmaps(
    image: Image.Image,
    labels: list,
    min_score: float = 0.3,
) -> dict:
    """Generate Grad-CAM heatmaps for multiple pathology labels."""
    scores = get_pathology_labels(image)

    results = {}
    for label in labels:
        score = _get_label_score(scores, label)
        if score >= min_score:
            try:
                result = get_heatmap(image, target_label=label)
                results[label] = result
                print(f"Generated heatmap for {label}: {score:.2f}")
            except Exception as exc:
                print(f"Skipped {label}: {exc}")

    return results


if __name__ == "__main__":
    img = Image.fromarray(
        np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    ).convert("L")

    labels = get_pathology_labels(img)
    print("Top labels:", dict(list(labels.items())[:3]))

    result = get_heatmap(img)
    print("Heatmap target:", result["target_label"])
    print("Heatmap score:", result["target_score"])
    print("Overlay size:", result["heatmap_overlay"].size)

    result["heatmap_overlay"].save("test_heatmap.png")
    print("Saved test_heatmap.png — open it to verify visually")
