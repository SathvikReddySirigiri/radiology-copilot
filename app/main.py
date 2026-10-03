"""Streamlit entry point for the Radiology Report Copilot application.

Wires together the vision pipeline, report drafter, and QA agent into an
interactive UI for radiologists.
"""

from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from app.utils.evaluator import threshold_predictions
from app.vision.bbox import annotate_xray


@st.cache_resource(show_spinner="Loading TorchXRayVision model...")
def load_torchxray_model():
    import app.vision.torchxray as torchxray

    return torchxray._MODEL


@st.cache_resource(show_spinner="Loading scispaCy model...")
def load_scispacy_model():
    import app.utils.ner as ner

    return ner._NLP


def _merge_state(state: dict, updates: dict) -> dict:
    return {**state, **updates}


def run_pipeline_with_progress(image: Image.Image) -> dict:
    """Run the pipeline step-by-step with progress and toast updates."""
    load_torchxray_model()
    load_scispacy_model()

    from app.pipeline.graph import (
        finalize,
        run_drafter,
        run_heatmap,
        run_llava,
        run_lung_rads,
        run_ner,
        run_qa_node,
        run_vision,
    )

    state = {
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

    progress_bar = st.progress(0)
    status_text = st.empty()

    steps = [
        (
            run_vision,
            "Step 1/7: Running vision model...",
            "Vision analysis complete",
        ),
        (
            run_heatmap,
            "Step 2/7: Generating Grad-CAM heatmap...",
            "Grad-CAM heatmap complete",
        ),
        (
            run_lung_rads,
            "Step 3/7: Scoring Lung-RADS category...",
            "Lung-RADS scoring complete",
        ),
        (
            run_llava,
            "Step 4/7: LLaVA describing image...",
            "Visual description complete",
        ),
        (
            run_ner,
            "Step 5/7: Extracting clinical entities...",
            "Entity extraction complete",
        ),
        (
            run_drafter,
            "Step 6/7: Drafting report...",
            "Report draft complete",
        ),
        (
            run_qa_node,
            "Step 7/7: QA check...",
            "QA check complete",
        ),
    ]

    for index, (step_fn, step_message, toast_message) in enumerate(steps):
        status_text.text(step_message)
        progress_bar.progress(index / len(steps))
        state = _merge_state(state, step_fn(state))
        st.toast(toast_message)

    status_text.text("Finalizing report...")
    progress_bar.progress(0.95)
    state = _merge_state(state, finalize(state))
    progress_bar.progress(1.0)
    status_text.text("Analysis complete.")
    st.toast("Pipeline complete")

    return state


def _render_confidence_breakdown(pathology_labels: dict) -> None:
    predictions = threshold_predictions(pathology_labels)

    header = st.columns([2, 1, 1])
    header[0].markdown("**Pathology**")
    header[1].markdown("**Score**")
    header[2].markdown("**Prediction**")

    for label, score in pathology_labels.items():
        prediction = predictions[label]
        row = st.columns([2, 1, 1])

        if score >= 0.70:
            row[0].error(label)
            row[1].error(f"{score:.0%}")
            row[2].error(str(prediction))
        elif score > 0.4:
            row[0].warning(label)
            row[1].warning(f"{score:.0%}")
            row[2].warning(str(prediction))
        else:
            row[0].success(label)
            row[1].success(f"{score:.0%}")
            row[2].success(str(prediction))

    st.caption(
        "Threshold: 0.5 | Scores from TorchXRayVision DenseNet121"
    )


def _render_risk_level(heatmap_result: dict) -> None:
    score = heatmap_result.get("target_score", 0.0)
    if score >= 0.70:
        st.error("🔴 HIGH RISK")
        st.metric("Confidence", f"{score:.0%}")
        st.write("Immediate radiologist review recommended")
    elif score > 0.4:
        st.warning("🟡 MODERATE RISK")
        st.metric("Confidence", f"{score:.0%}")
        st.write("Follow-up imaging in 3–6 months")
    else:
        st.success("🟢 LOW RISK")
        st.metric("Confidence", f"{score:.0%}")
        st.write("Routine annual screening")


st.title("🫁 Radiology Report Copilot")
st.caption(
    "Frontal (PA/AP) chest X-rays only. Upload an image to get a structured clinical report."
)

st.sidebar.header("Model Info")
st.sidebar.info("Vision: TorchXRayVision + LLaVA")
st.sidebar.info("LLM: Llama 3.1 8B (local)")
st.sidebar.info("NER: scispaCy en_core_sci_sm")
st.sidebar.info("Orchestration: LangGraph")
st.sidebar.warning("First run takes ~2 min while models load into memory")

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

uploaded = st.file_uploader(
    "Upload a frontal (PA/AP) chest X-ray",
    type=["png", "jpg", "jpeg"],
)

image = None
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.session_state.uploaded_image = image
        if image.width < 1000 or image.height < 1000:
            st.warning("Low-resolution image — results may be unreliable")
        if st.session_state.pipeline_result is None:
            st.image(image, caption="Uploaded X-ray")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")

    if image is not None and st.button("Run Analysis"):
        st.session_state.pipeline_result = None
        st.session_state.pipeline_error = None
        st.session_state.uploaded_image = image
        try:
            st.session_state.pipeline_result = run_pipeline_with_progress(image)
        except Exception as exc:
            st.session_state.pipeline_error = str(exc)

if st.session_state.pipeline_error:
    st.error(st.session_state.pipeline_error)
    if st.button("Clear and retry"):
        st.session_state.pipeline_result = None
        st.session_state.pipeline_error = None
        st.rerun()

result = st.session_state.pipeline_result

if result is not None:
    pathology_labels = result.get("pathology_labels", {})
    all_scores = result.get("all_scores") or {}
    lung_rads = result.get("lung_rads")
    urgent_findings = result.get("urgent_findings") or []
    conflict = result.get("conflict") or {}

    if urgent_findings:
        st.subheader("Urgent findings")
        for finding in urgent_findings:
            st.error(finding)

    if conflict.get("conflict"):
        st.warning(
            "Conflict notice: The visual description disagreed with the "
            "quantitative scores. Radiologist review is essential."
        )
    for reason in conflict.get("reasons") or []:
        st.warning(reason)

    if (
        len(all_scores) >= 18
        and all(0.40 <= float(score) <= 0.65 for score in all_scores.values())
    ):
        st.warning("Model is uncertain on this image")

    score_source = all_scores or pathology_labels
    if score_source:
        st.sidebar.subheader("This Scan")
        positives = sum(1 for score in score_source.values() if float(score) >= 0.50)
        st.sidebar.metric("Findings Detected", positives)
        st.sidebar.metric(
            "Highest Confidence",
            f"{max(score_source.values()):.0%}",
        )
        if lung_rads is not None:
            st.sidebar.metric("Lung-RADS Category", lung_rads.category_str)

    all_heatmaps = result.get("all_heatmaps", {})
    heatmap_result = result.get("heatmap_result", {})
    selected_label = None
    selected_heatmap = heatmap_result or {}

    st.subheader("🔬 Explainability Explorer")

    available_labels = list(all_heatmaps.keys())

    if available_labels:
        available_labels.sort(
            key=lambda label: pathology_labels.get(label, 0),
            reverse=True,
        )

        selected_label = st.radio(
            "Select pathology to visualize:",
            options=available_labels,
            format_func=lambda label: (
                f"{label} — {pathology_labels.get(label, 0):.0%}"
            ),
            horizontal=True,
            key="label_selector",
        )

        selected_heatmap = all_heatmaps.get(selected_label, heatmap_result)

        col_orig, col_heat, col_info = st.columns([1, 1, 1])

        with col_orig:
            st.markdown("**Original X-ray**")
            uploaded_image = st.session_state.get("uploaded_image")
            if uploaded_image is not None:
                st.image(uploaded_image, use_container_width=True)
            else:
                st.info("Original image unavailable")

        with col_heat:
            st.markdown(f"**Grad-CAM → {selected_label}**")
            if selected_heatmap:
                st.image(
                    selected_heatmap["heatmap_overlay"],
                    use_container_width=True,
                    caption=(
                        f"Model attention for {selected_label} "
                        f"({selected_heatmap['target_score']:.0%} confidence)"
                    ),
                )
                buf = BytesIO()
                selected_heatmap["heatmap_overlay"].save(buf, format="PNG")
                st.download_button(
                    f"Download {selected_label} Heatmap",
                    data=buf.getvalue(),
                    file_name=f"gradcam_{selected_label.lower().replace(' ', '_')}.png",
                    mime="image/png",
                    key=f"dl_{selected_label}",
                )
            else:
                st.info("Heatmap unavailable")

        with col_info:
            score = pathology_labels.get(
                selected_label,
                selected_heatmap.get("target_score", 0) if selected_heatmap else 0,
            )

            if score >= 0.70:
                st.error("🔴 HIGH RISK")
            elif score > 0.4:
                st.warning("🟡 MODERATE RISK")
            else:
                st.success("🟢 LOW RISK")

            st.metric("Confidence", f"{score:.0%}")

            st.markdown("**All detected findings:**")
            for label in available_labels:
                label_score = pathology_labels.get(label, 0)
                indicator = "🔴" if label_score >= 0.70 else "🟡" if label_score > 0.4 else "🟢"
                if label == selected_label:
                    st.markdown(f"{indicator} **{label}: {label_score:.0%}**")
                else:
                    st.markdown(f"{indicator} {label}: {label_score:.0%}")

            st.caption("Click any label above to switch the heatmap view")
    elif heatmap_result:
        col_orig, col_heat, col_info = st.columns([1, 1, 1])

        with col_orig:
            st.markdown("**Original X-ray**")
            uploaded_image = st.session_state.get("uploaded_image")
            if uploaded_image is not None:
                st.image(uploaded_image, use_container_width=True)

        with col_heat:
            st.markdown("**Grad-CAM Heatmap**")
            st.image(
                heatmap_result["heatmap_overlay"],
                use_container_width=True,
                caption=(
                    f"Targeting: {heatmap_result['target_label']} "
                    f"({heatmap_result['target_score']:.0%})"
                ),
            )

        with col_info:
            _render_risk_level(heatmap_result)
    else:
        st.info("Heatmap unavailable — no explainability data for this scan.")

    uploaded_image = st.session_state.get("uploaded_image")
    if uploaded_image is not None and (all_heatmaps or heatmap_result):
        st.divider()
        st.subheader("📍 Lesion Localization")

        if "bbox_threshold" not in st.session_state:
            st.session_state.bbox_threshold = 0.45

        with st.expander("⚙️ Adjust Detection Sensitivity"):
            st.session_state.bbox_threshold = st.slider(
                "Activation threshold",
                min_value=0.2,
                max_value=0.8,
                value=float(st.session_state.bbox_threshold),
                step=0.05,
                help=(
                    "Lower = more regions detected. "
                    "Higher = only strongest activations."
                ),
                key="bbox_threshold_slider",
            )

        view_mode = st.radio(
            "Visualization mode:",
            ["Grad-CAM Heatmap", "Bounding Box Annotation", "Side by Side"],
            horizontal=True,
            key="view_mode",
        )

        active_label = selected_label or heatmap_result.get("target_label")
        annotation = annotate_xray(
            uploaded_image,
            all_heatmaps,
            pathology_labels,
            selected_label=active_label,
            threshold=st.session_state.bbox_threshold,
        )

        if view_mode == "Grad-CAM Heatmap" and selected_heatmap:
            st.image(
                selected_heatmap["heatmap_overlay"],
                use_container_width=True,
                caption=f"Grad-CAM: {active_label}",
            )

        elif view_mode == "Bounding Box Annotation":
            st.image(
                annotation["annotated_image"],
                use_container_width=True,
                caption="AI-detected regions with anatomical labels",
            )

            buf = BytesIO()
            annotation["annotated_image"].save(buf, format="PNG")
            st.download_button(
                "📥 Download Annotated X-ray",
                data=buf.getvalue(),
                file_name="annotated_xray.png",
                mime="image/png",
            )

        elif view_mode == "Side by Side" and selected_heatmap:
            col_hm, col_bb = st.columns(2)
            with col_hm:
                st.markdown("**Grad-CAM Heatmap**")
                st.image(
                    selected_heatmap["heatmap_overlay"],
                    use_container_width=True,
                )
            with col_bb:
                st.markdown("**Bounding Box**")
                st.image(
                    annotation["annotated_image"],
                    use_container_width=True,
                )

        if annotation["boxes"]:
            st.markdown("**Detected Regions:**")

            rows = [
                {
                    "Pathology": box.label,
                    "Confidence": f"{box.confidence:.0%}",
                    "Anatomical Region": box.region,
                    "Risk Level": box.risk_level.upper(),
                    "Area Coverage": f"{box.area_pct:.1%}",
                }
                for box in annotation["boxes"]
            ]

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Localization Summary:**")
            st.code(annotation["summary"])
        else:
            st.info(
                "No regions detected above threshold. "
                "Try lowering threshold or uploading a clearer image."
            )

    if lung_rads is not None:
        from app.pipeline.lung_rads import get_lung_rads_display

        display = get_lung_rads_display(lung_rads)

        st.divider()
        st.subheader("🫁 Lung-RADS Assessment")

        col_cat, col_risk, col_action = st.columns([1, 1, 2])

        with col_cat:
            st.markdown(f"### {display['emoji']} {lung_rads.category_str}")
            st.markdown(
                f"<div style='background:{display['color']}22;"
                f"border-left: 4px solid {display['color']};"
                f"padding:10px;border-radius:6px;'>"
                f"<b>{lung_rads.finding_description}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_risk:
            st.metric("Malignancy Risk", lung_rads.malignancy_risk)
            follow_up_label = (
                f"{lung_rads.follow_up_months} months"
                if lung_rads.follow_up_months > 0
                else "IMMEDIATE"
            )
            st.metric("Follow-up", follow_up_label)
            if lung_rads.urgent:
                st.error("⚠️ URGENT ACTION REQUIRED")

        with col_action:
            st.markdown("**Recommended Actions:**")
            for action in display["action_items"]:
                st.markdown(f"- {action}")
            st.caption(
                "Lung-RADS-inspired risk category (Lung-RADS is designed for CT; "
                "used here as a reference scale). This tool assists — does not "
                "replace — radiologist judgment."
            )

        st.divider()

    llava_description = result.get("llava_description", "")
    final_report = result.get("final_report", "")
    qa_result = result.get("qa_result", {})

    with st.expander("All 18 scores"):
        if all_scores:
            for label, score in all_scores.items():
                st.write(f"{label}: {score:.0%}")
                st.progress(min(max(float(score), 0.0), 1.0))
        else:
            st.write("No scores available.")

    with st.expander("📊 Pathology Scores", expanded=True):
        if pathology_labels:
            for label, score in pathology_labels.items():
                percentage = f"{score * 100:.0f}%"
                label_text = f"{label} — {percentage}"
                if score >= 0.70:
                    st.error(label_text)
                elif score > 0.4:
                    st.warning(label_text)
                else:
                    st.success(label_text)
                st.progress(min(max(float(score), 0.0), 1.0))
        else:
            st.write("No pathology scores available.")

    with st.expander("📈 Model Confidence Breakdown"):
        if pathology_labels:
            _render_confidence_breakdown(pathology_labels)
        else:
            st.write("No pathology scores available.")

    with st.expander("🔍 LLaVA Visual Description"):
        st.write(llava_description or "No visual description available.")

    if qa_result and not qa_result.get("passed", False):
        st.error("QA FAILED — not reviewed")
        for flag in qa_result.get("flags", []):
            st.warning(flag)

    with st.expander("🏥 Final Report", expanded=True):
        st.text_area(
            "Final Report",
            value=final_report,
            height=300,
            label_visibility="collapsed",
        )
        filename = (
            f"radiology_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        st.download_button(
            "📄 Download Full Clinical Report",
            data=final_report,
            file_name=filename,
            mime="text/plain",
        )

    with st.expander("✅ QA Results"):
        passed = qa_result.get("passed", False)
        confidence = qa_result.get("confidence", "unknown")
        if passed:
            st.success(f"QA Passed — confidence: {confidence}")
        else:
            st.error("QA FAILED — not reviewed")
        for flag in qa_result.get("flags", []):
            st.warning(flag)

    if result.get("error"):
        st.error(f"Pipeline error: {result['error']}")
        if st.button("Clear and retry", key="clear_after_pipeline_error"):
            st.session_state.pipeline_result = None
            st.session_state.pipeline_error = None
            st.rerun()

st.divider()
st.caption(
    "Built with TorchXRayVision · LLaVA · Llama 3.1 · scispaCy · "
    "LangGraph · Streamlit | All models run locally — no data leaves "
    "your machine"
)
