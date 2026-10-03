"""LangGraph workflow definition for the radiology report pipeline.

Defines the state graph connecting vision analysis, report drafting, and
QA validation into a single orchestrated pipeline.
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image

from app.pipeline.drafter import detect_conflict, draft_report
from app.pipeline.lung_rads import LungRADSResult
from app.pipeline.qa_agent import run_qa
from app.utils.ner import extract_entities
from app.vision.llava_client import describe_xray
from app.vision.torchxray import get_all_scores


class ReportState(TypedDict):
    image: Image.Image
    pathology_labels: dict
    all_scores: dict
    urgent_findings: list
    conflict: dict
    heatmap_result: dict
    all_heatmaps: dict
    lung_rads: Optional[LungRADSResult]
    llava_description: str
    ner_entities: dict
    draft: str
    qa_result: dict
    final_report: str
    error: str


def _append_error(existing: str, message: str) -> str:
    if existing:
        return f"{existing}; {message}"
    return message


def _full_scores(state: ReportState) -> dict:
    all_scores = state.get("all_scores")
    if all_scores:
        return all_scores
    return state.get("pathology_labels", {})


def run_vision(state: ReportState) -> dict:
    try:
        all_scores = get_all_scores(state["image"])
        return {
            "all_scores": all_scores,
            "pathology_labels": dict(list(all_scores.items())[:5]),
        }
    except Exception as exc:
        return {"error": _append_error(state.get("error", ""), str(exc))}


def run_heatmap(state: ReportState) -> dict:
    try:
        from app.vision.torchxray import (
            SWITCHABLE_LABELS,
            get_all_heatmaps,
            get_heatmap,
        )

        top_result = get_heatmap(state["image"], target_label=None)
        all_results = get_all_heatmaps(
            state["image"],
            SWITCHABLE_LABELS,
            min_score=0.3,
        )
        print(f"Generated {len(all_results)} label heatmaps")
        return {
            "heatmap_result": top_result,
            "all_heatmaps": all_results,
        }
    except Exception as exc:
        return {
            "error": _append_error(state.get("error", ""), f"Heatmap error: {exc}"),
            "all_heatmaps": {},
        }


def run_lung_rads(state: ReportState) -> dict:
    try:
        from app.pipeline.lung_rads import check_urgent_findings, score_lung_rads

        scores = _full_scores(state)
        return {
            "lung_rads": score_lung_rads(scores),
            "urgent_findings": check_urgent_findings(scores),
        }
    except Exception as exc:
        return {"error": _append_error(state.get("error", ""), str(exc))}


def run_llava(state: ReportState) -> dict:
    try:
        return {"llava_description": describe_xray(state["image"])}
    except Exception as exc:
        return {"error": _append_error(state.get("error", ""), str(exc))}


def run_ner(state: ReportState) -> dict:
    try:
        return {"ner_entities": extract_entities(state["llava_description"])}
    except Exception as exc:
        return {"error": _append_error(state.get("error", ""), str(exc))}


def run_drafter(state: ReportState) -> dict:
    try:
        scores = _full_scores(state)
        conflict = detect_conflict(state.get("llava_description", ""), scores)
        return {
            "conflict": conflict,
            "draft": draft_report(
                scores,
                state["llava_description"],
                state["ner_entities"],
                lung_rads=state.get("lung_rads"),
                urgent_findings=state.get("urgent_findings"),
                conflict=conflict,
            ),
        }
    except Exception as exc:
        return {"error": _append_error(state.get("error", ""), str(exc))}


def run_qa_node(state: ReportState) -> dict:
    try:
        return {
            "qa_result": run_qa(
                state["draft"],
                _full_scores(state),
                state.get("urgent_findings") or [],
            )
        }
    except Exception as exc:
        return {"error": _append_error(state.get("error", ""), str(exc))}


def finalize(state: ReportState) -> dict:
    try:
        qa_result = state.get("qa_result") or {}
        report_text = state.get("draft", "")
        if qa_result.get("passed") and qa_result.get("approved_report"):
            report_text = qa_result["approved_report"]
        return {"final_report": report_text}
    except Exception as exc:
        return {"error": _append_error(state.get("error", ""), str(exc))}


graph = StateGraph(ReportState)
graph.add_node("run_vision", run_vision)
graph.add_node("run_heatmap", run_heatmap)
graph.add_node("run_lung_rads", run_lung_rads)
graph.add_node("run_llava", run_llava)
graph.add_node("run_ner", run_ner)
graph.add_node("run_drafter", run_drafter)
graph.add_node("run_qa", run_qa_node)
graph.add_node("finalize", finalize)

graph.add_edge(START, "run_vision")
graph.add_edge("run_vision", "run_heatmap")
graph.add_edge("run_heatmap", "run_lung_rads")
graph.add_edge("run_lung_rads", "run_llava")
graph.add_edge("run_llava", "run_ner")
graph.add_edge("run_ner", "run_drafter")
graph.add_edge("run_drafter", "run_qa")
graph.add_edge("run_qa", "finalize")
graph.add_edge("finalize", END)

pipeline = graph.compile()


def run_pipeline(image: Image.Image) -> dict:
    """Run the full radiology report pipeline on a chest X-ray image."""
    initial_state: ReportState = {
        "image": image,
        "pathology_labels": {},
        "all_scores": {},
        "urgent_findings": [],
        "conflict": {},
        "heatmap_result": {},
        "all_heatmaps": {},
        "lung_rads": None,
        "llava_description": "",
        "ner_entities": {},
        "draft": "",
        "qa_result": {},
        "final_report": "",
        "error": "",
    }
    return pipeline.invoke(initial_state)


if __name__ == "__main__":
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    result = run_pipeline(img)
    print("Final report:", result["final_report"])
    print("QA result:", result["qa_result"])
    print("Errors:", result["error"])
