"""Evaluation utilities for benchmarking against CheXpert labels.

Maps TorchXRayVision pathology scores to CheXpert ground truth labels
and computes classification metrics.
"""

LABEL_MAP = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Effusion",
    "Lung Opacity": "Infiltration",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Fibrosis": "Fibrosis",
    "Hernia": "Hernia",
    "Pleural Thickening": "Pleural_Thickening",
    "Emphysema": "Emphysema",
}

_TORCHXRAY_TO_CHEXPERT = {torchxray: chexpert for chexpert, torchxray in LABEL_MAP.items()}


def _normalize_label(label: str) -> str:
    return label.lower().replace("_", " ").strip()


def _lookup_torchxray_score(pathology_labels: dict, torchxray_label: str) -> float:
    target = _normalize_label(torchxray_label)
    for label, score in pathology_labels.items():
        normalized = _normalize_label(label)
        if normalized == target:
            return float(score)
    return 0.0


def map_pathology_to_chexpert(pathology_labels: dict) -> dict[str, float]:
    """Map TorchXRayVision scores to CheXpert label names."""
    chexpert_scores = {label: 0.0 for label in LABEL_MAP}
    for chexpert_label, torchxray_label in LABEL_MAP.items():
        chexpert_scores[chexpert_label] = max(
            chexpert_scores[chexpert_label],
            _lookup_torchxray_score(pathology_labels, torchxray_label),
            _lookup_torchxray_score(pathology_labels, chexpert_label),
        )
    return chexpert_scores


def threshold_predictions(
    pathology_labels: dict,
    threshold: float = 0.5,
) -> dict:
    """Convert continuous pathology scores to binary predictions."""
    return {
        label: 1 if float(score) > threshold else 0
        for label, score in pathology_labels.items()
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_label_metrics(prediction: int, ground_truth: int) -> dict:
    tp = int(prediction == 1 and ground_truth == 1)
    fp = int(prediction == 1 and ground_truth == 0)
    fn = int(prediction == 0 and ground_truth == 1)
    tn = int(prediction == 0 and ground_truth == 0)

    sensitivity = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    ppv = _safe_ratio(tp, tp + fp)
    f1 = _safe_ratio(2 * ppv * sensitivity, ppv + sensitivity)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "f1": f1,
    }


def compute_metrics(predictions: dict, ground_truth: dict) -> dict:
    """Compute per-label and overall metrics against CheXpert labels."""
    labels = sorted(set(LABEL_MAP) | set(predictions) | set(ground_truth))
    per_label = {}

    for label in labels:
        prediction = int(predictions.get(label, 0))
        truth = int(ground_truth.get(label, 0))
        per_label[label] = _compute_label_metrics(prediction, truth)

    f1_scores = [metrics["f1"] for metrics in per_label.values()]
    sensitivities = [metrics["sensitivity"] for metrics in per_label.values()]
    specificities = [metrics["specificity"] for metrics in per_label.values()]

    return {
        "per_label": per_label,
        "overall": {
            "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "macro_sensitivity": (
                sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
            ),
            "macro_specificity": (
                sum(specificities) / len(specificities) if specificities else 0.0
            ),
            "auc_note": "AUC requires multiple samples",
        },
    }


def format_metrics_report(metrics: dict) -> str:
    """Return a formatted metrics table for display."""
    lines = [
        "CheXpert Evaluation Metrics",
        "=" * 72,
        f"{'Label':<22} {'F1':>8} {'Sens':>8} {'Spec':>8} {'PPV':>8} "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}",
        "-" * 72,
    ]

    for label, values in sorted(metrics["per_label"].items()):
        lines.append(
            f"{label:<22} "
            f"{values['f1']:8.3f} "
            f"{values['sensitivity']:8.3f} "
            f"{values['specificity']:8.3f} "
            f"{values['ppv']:8.3f} "
            f"{values['tp']:4d} "
            f"{values['fp']:4d} "
            f"{values['fn']:4d} "
            f"{values['tn']:4d}"
        )

    overall = metrics["overall"]
    lines.extend(
        [
            "-" * 72,
            f"Macro F1          : {overall['macro_f1']:.3f}",
            f"Macro Sensitivity : {overall['macro_sensitivity']:.3f}",
            f"Macro Specificity : {overall['macro_specificity']:.3f}",
            f"Note              : {overall['auc_note']}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    preds = threshold_predictions(
        {"Pneumonia": 0.78, "Effusion": 0.45, "Mass": 0.20}
    )
    print(preds)

    sample_metrics = compute_metrics(
        {"Pneumonia": 1, "Effusion": 0, "Mass": 0},
        {"Pneumonia": 1, "Effusion": 1, "Mass": 0},
    )
    print(format_metrics_report(sample_metrics))
