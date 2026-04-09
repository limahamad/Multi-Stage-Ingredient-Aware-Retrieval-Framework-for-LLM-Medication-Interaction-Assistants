import itertools
import json
import html
from typing import Any, Dict, List

import streamlit as st
from sentence_transformers import SentenceTransformer

from app_three_stage import (
    EMBED_MODEL,
    INDEX_ROOT,
    GEN_MODEL,
    HybridRetriever,
    run_three_stage_pipeline,
)


st.set_page_config(
    page_title="Medication Safety Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


APP_TITLE = "Medication Safety Assistant"
APP_SUBTITLE = "Check interactions between medicines and ask medication safety questions."
HOME_TAGLINE = "Your trusted companion for managing medication safely."
SAFETY_NOTE = (
    "This tool is for informational support and does not replace advice from a "
    "pharmacist or physician."
)
HOME_EXAMPLES = [
    "Can I take Panadol with warfarin?",
    "I take Advil for pain and baby aspirin for my heart. Is that safe?",
    "Can I use UnknownBrandX with paracetamol?",
    "Is it okay to drink grapefruit juice while taking Zocor?",
    "Can I take Augmentin if I'm allergic to amoxicillin?",
]
CHAT_EXAMPLES = [
    "Can I take Panadol with warfarin?",
    "Is it safe to use paracetamol while taking metoclopramide?",
    "Can I take ibuprofen together with aspirin?",
    "Can I use Advil while I'm on warfarin?",
    "Is Tylenol the same as paracetamol, and can I take them together?",
    "Can I take UnknownBrandX with paracetamol?",
    "Is paracetamol safe to take with a herbal supplement?",
    "Can I take metformin and lisinopril together?",
    "Is it okay to use cholestyramine with paracetamol?",
    "Can I take lamotrigine while using paracetamol?",
    "Is it safe to combine paracetamol and isoniazid?",
    "Can I take paracetamol with zidovudine?",
    "Is paracetamol safe to use with flucloxacillin?",
    "Can I take paracetamol while on oral contraceptives?",
    "Can I use 0.9% sodium chloride irrigation with metronidazole injection?",
    "I take Tylenol and Panadol together for headaches. Is that safe?",
    "Can I take Augmentin if I'm allergic to amoxicillin?",
    "I'm taking Lipitor and clarithromycin for an infection. Is that okay?",
    "I take Plavix for my heart. Can I also take omeprazole for reflux?",
    "I take Advil for pain and a daily baby aspirin for my heart. Is that combination safe?",
    "My doctor prescribed Cipro and I also take warfarin. Should I worry about interactions?",
    "Is it okay to drink grapefruit juice while taking Zocor?",
    "I take Cozaar for blood pressure and sometimes ibuprofen for back pain. Is that safe?",
    "I'm taking codeine cough syrup and also a medicine that contains acetaminophen. Is that okay?",
]


def init_session_state() -> None:
    if "medicine_list" not in st.session_state:
        st.session_state.medicine_list = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input_seed" not in st.session_state:
        st.session_state.chat_input_seed = ""
    if "ask_inline_input" not in st.session_state:
        st.session_state.ask_inline_input = ""
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .main {
                background: linear-gradient(180deg, #f8fbff 0%, #ffffff 42%, #f8fcff 100%);
            }
            .block-container {
                padding-top: 1.1rem;
                padding-bottom: 3.5rem;
                max-width: 1120px;
            }
            .feature-card, .panel-card, .soft-card, .safety-info-card, .dashboard-card {
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                border: 1px solid #d8e8e6;
                background: rgba(255,255,255,0.92);
                box-shadow: 0 12px 30px rgba(11, 68, 83, 0.06);
            }
            .app-header {
                display: flex;
                align-items: center;
                gap: 12px;
                margin-top: 16px;
                margin-bottom: 8px;
            }
            .app-header-icon {
                width: 36px;
                height: 36px;
                border-radius: 10px;
                background: linear-gradient(135deg, #3b82f6 0%, #06c167 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #ffffff;
                font-size: 15px;
                font-weight: 700;
                box-shadow: 0 8px 16px rgba(37, 99, 235, 0.14);
                margin-top: 6px;
            }
            .app-header-title {
                margin: 0 !important;
                font-size: 15px !important;
                font-weight: 700 !important;
                color: #0f172a !important;
                line-height: 1.1 !important;
            }
            .app-header-subtitle {
                margin: -10px 0 0 0 !important;
                font-size: 9px !important;
                color: #475467 !important;
                line-height: 1 !important;
            }
            .nav-shell {
                border-bottom: 1px solid #d9e3ef;
                margin-bottom: 2.2rem;
                padding-bottom: 0.1rem;
            }
            .dashboard-hero {
                text-align: center;
                max-width: 760px;
                margin: 2.1rem auto 2.9rem auto;
            }
            .dashboard-icon {
                width: 64px;
                height: 64px;
                margin: 0 auto 1.4rem auto;
                border-radius: 18px;
                background: linear-gradient(135deg, #3b82f6 0%, #06c167 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.7rem;
                box-shadow: 0 16px 30px rgba(37, 99, 235, 0.18);
            }
            .dashboard-title {
                margin: 0;
                font-size: 2.25rem;
                font-weight: 700;
                color: #0f172a;
            }
            .dashboard-subtitle {
                margin: 1rem auto 0 auto;
                font-size: 1.18rem;
                line-height: 1.65;
                color: #475467;
                max-width: 700px;
            }
            .feature-card {
                min-height: 165px;
                background: linear-gradient(180deg, #ffffff 0%, #f7fcfb 100%);
            }
            .dashboard-card {
                min-height: 250px;
                padding: 1.5rem;
                border: 1px solid #d9d9de;
                background: rgba(255,255,255,0.97);
                box-shadow: none;
            }
            .dashboard-card-icon {
                width: 48px;
                height: 48px;
                border-radius: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.35rem;
                margin-bottom: 1.3rem;
            }
            .dashboard-card-title {
                margin: 0 0 0.6rem 0;
                font-size: 1.2rem;
                font-weight: 700;
                color: #0f172a;
            }
            .dashboard-card-text {
                margin: 0;
                font-size: 1rem;
                line-height: 1.6;
                color: #667085;
                min-height: 84px;
            }
            .card-icon-blue {
                background: #dbeafe;
                color: #2563eb;
            }
            .card-icon-green {
                background: #dcfce7;
                color: #16a34a;
            }
            .safety-info-card {
                margin-top: 1.75rem;
                padding: 1.35rem 1.5rem;
                border: 1px solid #bfd7ff;
                background: linear-gradient(180deg, #edf4ff 0%, #e8f0ff 100%);
                box-shadow: none;
            }
            .safety-info-wrap {
                display: flex;
                align-items: flex-start;
                gap: 1rem;
            }
            .safety-info-icon {
                width: 40px;
                height: 40px;
                min-width: 40px;
                border-radius: 12px;
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.1rem;
                font-weight: 800;
            }
            .safety-info-title {
                font-size: 1.05rem;
                font-weight: 700;
                color: #163b73;
                margin: 0 0 0.45rem 0;
            }
            .safety-info-text {
                font-size: 1rem;
                line-height: 1.55;
                color: #2e4a73;
                margin: 0;
            }
            .badge {
                display: inline-block;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.01em;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
            }
            .badge-green { background:#e6f7ea; color:#157347; }
            .badge-yellow { background:#fff4d6; color:#9a6700; }
            .badge-red { background:#fde7e9; color:#b42318; }
            .badge-gray { background:#eceff3; color:#475467; }
            .badge-blue { background:#e7f0ff; color:#175cd3; }
            .section-label {
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #386a6c;
                font-weight: 700;
            }
            .ask-shell {
                max-width: 1120px;
                margin: 0 auto;
                border-radius: 22px;
                overflow: hidden;
                border: 1px solid #dbe3ef;
                background: #ffffff;
                box-shadow: 0 16px 34px rgba(15, 23, 42, 0.08);
            }
            .ask-header {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1.6rem 1.4rem;
                background: linear-gradient(90deg, #3b82f6 0%, #06c167 100%);
                color: #ffffff;
            }
            .ask-header-icon {
                width: 50px;
                height: 50px;
                min-width: 50px;
                border-radius: 16px;
                background: rgba(255,255,255,0.18);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.35rem;
                font-weight: 700;
            }
            .ask-header-title {
                margin: 0;
                font-size: 1.08rem;
                font-weight: 700;
            }
            .ask-header-subtitle {
                margin: 0.2rem 0 0 0;
                font-size: 0.95rem;
                opacity: 0.95;
            }
            .ask-suggestions {
                padding: 1.25rem 1.35rem 1.1rem 1.35rem;
                background: #eaf2ff;
                border-bottom: 1px solid #dce6f5;
            }
            .st-key-ask-suggestions-block {
                padding: 1.05rem 1.35rem 1.75rem 1.35rem;
                background: linear-gradient(180deg, #eef5ff 0%, #e8f1ff 100%);
                border-bottom: 1px solid #d6e4fb;
            }
            .ask-suggestions-title {
                margin: 0 0 1rem 0;
                color: #304768;
                font-size: 0.95rem;
                font-weight: 700;
            }
            .st-key-ask-suggestions-block [data-testid="stButton"] {
                margin-top: 0;
            }
            .st-key-ask-suggestion-0 {
                background: transparent !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
            }
            .st-key-ask-suggestion-0 button {
                width: auto !important;
                min-width: 290px !important;
                border-radius: 999px !important;
                border: 1px solid #bfd4fb !important;
                background: #ffffff !important;
                color: #1d4ed8 !important;
                text-align: left !important;
                font-weight: 600 !important;
                box-shadow: 0 4px 14px rgba(37, 99, 235, 0.06) !important;
                padding: 0.38rem 1rem !important;
            }
            .st-key-ask-suggestion-0 button:hover {
                border-color: #8fb6ff !important;
                color: #1d4ed8 !important;
                background: #f8fbff !important;
                box-shadow: 0 6px 16px rgba(37, 99, 235, 0.1) !important;
            }
            .st-key-ask-suggestion-0 button p {
                color: #1d4ed8 !important;
            }
            .st-key-ask-thread-block {
                min-height: 360px;
                padding: 1.1rem 1.35rem 1.4rem 1.35rem;
                background: #ffffff;
            }
            .ask-message-row {
                display: flex;
                align-items: flex-start;
                gap: 0.9rem;
                margin-bottom: 1rem;
            }
            .ask-message-icon {
                width: 40px;
                height: 40px;
                min-width: 40px;
                border-radius: 999px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #ffffff;
                font-size: 1rem;
                font-weight: 700;
            }
            .ask-message-icon-assistant {
                background: linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
            }
            .ask-message-icon-user {
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            }
            .ask-bubble {
                max-width: 760px;
                border-radius: 18px;
                padding: 1rem 1.1rem;
                line-height: 1.65;
                font-size: 1.02rem;
            }
            .ask-bubble-assistant {
                background: #f2f4f7;
                color: #111827;
            }
            .ask-bubble-user {
                background: #eaf2ff;
                color: #163b73;
            }
            .ask-input-panel {
                padding: 0.6rem 0 0.8rem 3.05rem;
                background: #ffffff;
            }
            [class*="st-key-ask-suggestion-"] button {
                border-radius: 14px;
                border: 1px solid #b8d2ff;
                background: #ffffff;
                color: #1d4ed8;
                text-align: left;
                font-weight: 500;
            }
            [class*="st-key-ask-suggestion-"] button:hover {
                border-color: #7fb1ff;
                color: #1d4ed8;
                background: #f8fbff;
            }
            .ask-input-panel [data-testid="stTextInput"] {
                background: transparent;
                border-radius: 22px;
            }
            .ask-input-panel [data-testid="stTextInput"] > div {
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                background: #f1f5f9;
                box-shadow: none;
                min-height: 34px;
            }
            .ask-input-panel [data-testid="stTextInput"] > div:focus-within {
                border-color: #d6dbe3;
                box-shadow: none;
            }
            .ask-input-panel input {
                color: #1f2937;
                font-size: 0.95rem;
            }
            .ask-input-panel .stCaption {
                color: #7a7a7a;
                margin-top: 0.3rem;
            }
            .ask-input-panel [data-testid="stForm"] {
                margin-bottom: 0;
            }
            .ask-input-panel [data-testid="stForm"] button {
                background: #2563eb !important;
                color: #ffffff !important;
                border: 1px solid #2563eb !important;
                min-height: 34px;
            }
            .ask-input-panel [data-testid="stForm"] button:hover {
                background: #1d4ed8 !important;
                border-color: #1d4ed8 !important;
                color: #ffffff !important;
            }
            @media (max-width: 768px) {
                .dashboard-title {
                    font-size: 1.85rem;
                }
                .dashboard-subtitle {
                    font-size: 1.02rem;
                }
                .dashboard-card {
                    min-height: auto;
                }
                .dashboard-card-text {
                    min-height: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pipeline_resources():
    model = SentenceTransformer(EMBED_MODEL)
    stage1_retriever = HybridRetriever(INDEX_ROOT / "stage1_normalization", model)
    stage2_retriever = HybridRetriever(INDEX_ROOT / "stage2_ingredients", model)
    stage3_retriever = HybridRetriever(INDEX_ROOT / "stage3_interactions", model)
    return stage1_retriever, stage2_retriever, stage3_retriever


def severity_badge_html(label: str) -> str:
    normalized = label.lower()
    if "major" in normalized or "not safe" in normalized or "high risk" in normalized:
        css_class = "badge-red"
    elif "moderate" in normalized or "caution" in normalized or "mild" in normalized:
        css_class = "badge-yellow"
    elif "uncertain" in normalized or "unknown" in normalized:
        css_class = "badge-gray"
    else:
        css_class = "badge-green"
    return f"<span class='badge {css_class}'>{label}</span>"


def feature_card(title: str, icon: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="section-label">{icon} {title}</div>
            <div style="font-size:1.02rem; margin-top:0.7rem; color:#1f2937;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_pair_question(name_a: str, name_b: str) -> str:
    return f"Can I take {name_a} with {name_b}?"


def format_pair_display(question: str) -> str:
    normalized = question.strip()
    if normalized.lower().startswith("can i take "):
        normalized = normalized[11:]
    if normalized.endswith("?"):
        normalized = normalized[:-1]
    return normalized


def run_pipeline_query(question: str) -> Dict[str, Any]:
    stage1_retriever, stage2_retriever, stage3_retriever = load_pipeline_resources()
    try:
        output = run_three_stage_pipeline(
            user_question=question,
            stage1_retriever=stage1_retriever,
            stage2_retriever=stage2_retriever,
            stage3_retriever=stage3_retriever,
            verbose=False,
        )
        return {
            "ok": True,
            "question": question,
            "output": output,
            "summary": output.get("final_output", {}).get("short_answer", output.get("answer", "")),
            "severity": output.get("final_output", {}).get("severity", "unknown"),
            "decision": output.get("final_output", {}).get("decision", "Uncertain"),
        }
    except Exception as exc:
        return {
            "ok": False,
            "question": question,
            "error": str(exc),
            "summary": (
                "The system could not reach a confident answer from the available information. "
                "Please verify the product name or ingredient list before relying on the result."
            ),
            "severity": "uncertain",
            "decision": "Uncertain",
        }


def pairwise_medicine_results(names: List[str]) -> List[Dict[str, Any]]:
    return [run_pipeline_query(build_pair_question(left, right)) for left, right in itertools.combinations(names, 2)]


def render_pipeline_reasoning(output: Dict[str, Any]) -> None:
    stage1_output = output.get("stage1_output", {})
    stage2_output = output.get("stage2_output", {})
    final_output = output.get("final_output", {})
    st.markdown("**Stage 1: Name normalization**")
    normalized = stage1_output.get("normalized_medications", [])
    uncertain = stage1_output.get("uncertain_mentions", [])
    mentions = stage1_output.get("input_mentions", [])
    if mentions:
        st.markdown(f"- Input mentions: {', '.join(mentions)}")
    if normalized:
        for item in normalized:
            input_mention = item.get("input_mention", "")
            generic_name = item.get("canonical_generic_name", "")
            matched_alias = item.get("matched_brand_or_alias", "")
            confidence = item.get("confidence", "")
            st.markdown(
                f"- `{input_mention}` -> **{generic_name}**"
                f" | matched as: `{matched_alias}` | confidence: `{confidence}`"
            )
    else:
        st.markdown("- No confident normalization found.")
    if uncertain:
        st.markdown(f"- Uncertain mentions: {', '.join(uncertain)}")

    st.markdown("**Stage 2: Ingredient retrieval**")
    medications = stage2_output.get("medications", [])
    if medications:
        for item in medications:
            med_name = item.get("canonical_generic_name", "")
            ingredients = item.get("active_ingredients", [])
            therapeutic_class = item.get("therapeutic_class", "")
            confidence = item.get("confidence", "")
            st.markdown(
                f"- **{med_name}**: {', '.join(ingredients) if ingredients else 'No ingredients found'}"
            )
            if therapeutic_class:
                st.markdown(f"  Therapeutic class: `{therapeutic_class}`")
            if confidence != "":
                st.markdown(f"  Confidence: `{confidence}`")
    else:
        st.markdown("- No ingredient mapping found.")

    st.markdown("**Stage 3: Interaction reasoning**")
    short_answer = final_output.get("short_answer", "")
    mechanism = final_output.get("mechanism_summary", "")
    advice = final_output.get("safety_advice", [])
    evidence = final_output.get("evidence_summary", [])
    disclaimer = final_output.get("disclaimer", "")
    if short_answer:
        st.markdown(f"- Short answer: {short_answer}")
    st.markdown(f"- Interaction found: `{final_output.get('interaction_found', False)}`")
    if mechanism:
        st.markdown(f"- Mechanism summary: {mechanism}")
    if advice:
        st.markdown("**Recommended actions**")
        for item in advice:
            st.markdown(f"- {item}")
    if evidence:
        st.markdown("**Evidence used**")
        for item in evidence:
            source_title = item.get("source_title", "Unknown source")
            reason_used = item.get("reason_used", "")
            st.markdown(f"- **{source_title}**: {reason_used}")
    if disclaimer:
        st.caption(disclaimer)


def render_pair_result(result: Dict[str, Any], expanded: bool = False) -> None:
    st.write(format_pair_display(result["question"]))
    st.markdown(
        f"{severity_badge_html(result['decision'])} {severity_badge_html(str(result['severity']))}",
        unsafe_allow_html=True,
    )
    st.write(result["summary"])
    with st.expander("Show reasoning", expanded=expanded):
        if result["ok"]:
            render_pipeline_reasoning(result["output"])
        else:
            st.info(result["error"])
            st.caption(SAFETY_NOTE)


def render_home_tab() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.35rem;">
                <div style="width:42px; height:42px; border-radius:12px; background:linear-gradient(135deg, #0f766e 0%, #38b2ac 100%); display:flex; align-items:center; justify-content:center; color:white; font-size:1.25rem; font-weight:800; box-shadow:0 8px 18px rgba(15, 118, 110, 0.22);">✚</div>
                <div>
                    <div class="section-label" style="margin:0; font-size:1.25rem;">MediSafe</div>
                </div>
            </div>
            <h1 style="margin:0.35rem 0 0.4rem 0;">{APP_TITLE}</h1>
            <p style="font-size:1.05rem; color:#475467; margin:0;">{APP_SUBTITLE}</p>
            <p style="font-size:1rem; color:#5b6b73; margin:0.45rem 0 0 0;">Your trusted companion for managing medication safely.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        feature_card(
            "My Medicines",
            "🧾",
            "Add medicines you currently take to avoid any possible interactions or overdose.",
            title_color="#175cd3",
            card_class="feature-card-blue",
        )
        if st.button("Open My Medicines", key="open-my-medicines", use_container_width=True):
            st.session_state.active_page = "My Medicines"
            st.rerun()
    with col2:
        feature_card("Ask a Question", "💬", "Ask a natural-language medication safety question and inspect the normalization, ingredient, and interaction reasoning.")
        if st.button("Open Ask a Question", key="open-ask-question", use_container_width=True):
            st.session_state.active_page = "Ask a Question"
            st.rerun()
    st.markdown(
        """
        <div class="safety-info-card">
            <div class="safety-info-wrap">
                <div class="safety-info-icon">🛡</div>
                <div>
                    <div class="safety-info-title">Important Safety Information</div>
                    <p class="safety-info-text">
                        This tool is designed to assist with medication safety awareness. Always consult with your
                        healthcare provider or pharmacist before making any changes to your medication regimen.
                        This tool does not replace professional medical advice.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_medicine_chip(name: str, index: int) -> None:
    col_a, col_b = st.columns([6.2, 1.2], gap="small")
    with col_a:
        st.markdown(
            f"<div class='panel-card' style='margin-right:0; border-top-right-radius:10px; border-bottom-right-radius:10px;'><strong>{name}</strong></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        if st.button("Remove", key=f"remove-med-{index}", use_container_width=True, type="primary"):
            st.session_state.medicine_list.pop(index)
            st.rerun()


def add_medicine(name: str) -> None:
    cleaned = name.strip()
    if cleaned and cleaned not in st.session_state.medicine_list:
        st.session_state.medicine_list.append(cleaned)


def render_my_medicines_tab() -> None:
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### My Medicines")
    st.caption("Add medicines one by one. The app compares each pair to ensure the medications are safe to take together.")
    st.markdown(
        """
        <style>
            div[data-testid="stForm"] button {
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            div[data-testid="stForm"] button:hover {
                background: #1d4ed8;
                color: #ffffff;
                border-color: #1d4ed8;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.form("add_medicine_form", clear_on_submit=True):
        input_col, action_col = st.columns([5, 1])
        with input_col:
            medicine_name = st.text_input(
                "Add a medicine or product name",
                placeholder="e.g., Panadol, warfarin, Lipitor, UnknownBrandX",
                label_visibility="collapsed",
                key="medicine_input",
            )
        with action_col:
            submitted = st.form_submit_button("Add Medicine", use_container_width=True, type="primary")
        if submitted:
            add_medicine(medicine_name)
            st.rerun()
    st.write("")
    current = st.session_state.medicine_list
    if current:
        st.markdown("#### Current medicines")
        for idx, name in enumerate(current):
            render_medicine_chip(name, idx)
    else:
        st.info("No medicines added yet. Start by adding one medicine or product name.")
    st.write("")
    st.markdown("#### Interaction panel")
    if len(current) < 2:
        st.markdown("<div class='soft-card'>Add at least two medicines to compare them.</div>", unsafe_allow_html=True)
        st.caption(SAFETY_NOTE)
        return
    with st.spinner("Running the three-stage pipeline across your medicine pairs..."):
        results = pairwise_medicine_results(current)
    for result in results:
        render_pair_result(result)
        st.write("")
    if all(result["severity"] in {"none", "unknown"} for result in results):
        st.success(
            "No major direct interaction was found in the current pairwise checks. "
            "Patient-specific context, dose, timing, and medical history still matter."
        )
    st.caption(SAFETY_NOTE)


def queue_example_question(question: str) -> None:
    st.session_state.chat_input_seed = question


def process_question(question: str) -> Dict[str, Any] | None:
    cleaned = question.strip()
    if not cleaned:
        return None
    with st.spinner("Running the three-stage pipeline..."):
        result = run_pipeline_query(cleaned)
    return result


def render_ask_question_tab() -> None:
    st.markdown(
        """
        <div class="ask-shell">
            <div class="ask-header">
                <div class="ask-header-icon">🤖</div>
                <div>
                    <div class="ask-header-title">Medication Safety Assistant</div>
                    <div class="ask-header-subtitle">Ask me about medication safety</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ask-shell">', unsafe_allow_html=True)
    with st.container(key="ask-suggestions-block"):
        st.markdown('<div class="ask-suggestions-title">Try asking:</div>', unsafe_allow_html=True)
        if st.button("Can I take Panadol with Ibuprofen?", key="ask-suggestion-0"):
            queue_example_question("Can I take Panadol with Ibuprofen?")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="ask-shell"><div class="ask-thread">', unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="ask-message-row">
                <div class="ask-message-icon ask-message-icon-assistant">🤖</div>
                <div class="ask-bubble ask-bubble-assistant">
                    Hello! I'm your Medication Safety Assistant. I can help answer questions about medication
                    interactions, side effects, and general medication safety. How can I assist you today?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            escaped_content = html.escape(message["content"])
            st.markdown(
                f"""
                <div class="ask-message-row">
                    <div class="ask-message-icon ask-message-icon-user">You</div>
                    <div class="ask-bubble ask-bubble-user">{escaped_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue
        escaped_content = html.escape(message["content"])
        st.markdown(
            f"""
            <div class="ask-message-row">
                <div class="ask-message-icon ask-message-icon-assistant">🤖</div>
                <div class="ask-bubble ask-bubble-assistant">{escaped_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        result = message.get("result")
        if result is not None:
            st.markdown(
                f"{severity_badge_html(str(result['decision']))} "
                f"{severity_badge_html(str(result['severity']))}",
                unsafe_allow_html=True,
            )
            with st.expander("Show reasoning", expanded=False):
                if result["ok"]:
                    render_pipeline_reasoning(result["output"])
                else:
                    st.info(result["error"])
    st.markdown('<div class="ask-input-panel">', unsafe_allow_html=True)
    seeded = st.session_state.chat_input_seed
    if seeded and not st.session_state.ask_inline_input:
        st.session_state.ask_inline_input = seeded
        st.session_state.chat_input_seed = ""
    user_question = ""
    with st.form("ask_inline_form", clear_on_submit=True):
        input_col, action_col = st.columns([12, 1], gap="small")
        with input_col:
            st.text_input(
                "Ask about medication interactions...",
                placeholder="Ask about medication interactions...",
                label_visibility="collapsed",
                key="ask_inline_input",
            )
        with action_col:
            submitted = st.form_submit_button("Send", use_container_width=True)
        if submitted:
            user_question = st.session_state.ask_inline_input.strip()
    st.caption("This is a demonstration tool. Always consult healthcare professionals for medical advice.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        result = process_question(user_question)
        if result is not None:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["summary"], "result": result}
            )
        st.rerun()


def render_navigation() -> None:
    items = [
        ("Dashboard", "Home", "⌂", "nav-home"),
        ("My Medicines", "My Medicines", "⌕", "nav-medicines"),
        ("Ask a Question", "Ask a Question", "◻", "nav-ask"),
    ]
    current_page = st.session_state.active_page
    current_nav_class = {
        "Home": "nav-home",
        "My Medicines": "nav-medicines",
        "Ask a Question": "nav-ask",
    }.get(current_page, "nav-home")
    st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <style>
            .st-key-nav-home button,
            .st-key-nav-medicines button,
            .st-key-nav-ask button {{
                background: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                border-radius: 0;
                box-shadow: none;
                color: #344054;
                font-weight: 500;
                justify-content: flex-start;
                padding: 0.7rem 0.85rem 0.95rem 0.85rem;
            }}
            .st-key-nav-home button:hover,
            .st-key-nav-medicines button:hover,
            .st-key-nav-ask button:hover {{
                background: transparent;
                color: #2f6df6;
                border-color: transparent;
            }}
            .st-key-{current_nav_class} button {{
                color: #2f6df6;
                border-bottom-color: #2f6df6;
                font-weight: 700;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(len(items), gap="small")
    for col, (label, target_page, icon, key_name) in zip(cols, items):
        with col:
            if st.button(f"{icon} {label}", key=key_name, use_container_width=True):
                if target_page != current_page:
                    st.session_state.active_page = target_page
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_ask_question_tab() -> None:
    st.markdown(
        """
        <div class="ask-shell">
            <div class="ask-header">
                <div class="ask-header-icon">🤖</div>
                <div>
                    <div class="ask-header-title">Medication Safety Assistant</div>
                    <div class="ask-header-subtitle">Ask me about medication safety</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ask-shell">', unsafe_allow_html=True)
    with st.container(key="ask-suggestions-block"):
        st.markdown('<div class="ask-suggestions-title">Try asking:</div>', unsafe_allow_html=True)
        if st.button("Can I take Panadol with Ibuprofen?", key="ask-suggestion-0"):
            queue_example_question("Can I take Panadol with Ibuprofen?")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ask-shell">', unsafe_allow_html=True)
    with st.container(key="ask-thread-block"):
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div class="ask-message-row">
                    <div class="ask-message-icon ask-message-icon-assistant">🤖</div>
                    <div class="ask-bubble ask-bubble-assistant">
                        Hello! I'm your Medication Safety Assistant. I can help answer questions about medication
                        interactions, side effects, and general medication safety. How can I assist you today?
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for message in st.session_state.chat_history:
            escaped_content = html.escape(message["content"])
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div class="ask-message-row">
                        <div class="ask-message-icon ask-message-icon-user">You</div>
                        <div class="ask-bubble ask-bubble-user">{escaped_content}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue
            if message.get("pending"):
                st.markdown(
                    f"""
                    <div class="ask-message-row">
                        <div class="ask-message-icon ask-message-icon-assistant">ðŸ¤–</div>
                        <div class="ask-bubble ask-bubble-assistant"><em>{escaped_content}</em></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue
            st.markdown(
                f"""
                <div class="ask-message-row">
                    <div class="ask-message-icon ask-message-icon-assistant">🤖</div>
                    <div class="ask-bubble ask-bubble-assistant">{escaped_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            result = message.get("result")
            if result is not None:
                st.markdown(
                    f"{severity_badge_html(str(result['decision']))} "
                    f"{severity_badge_html(str(result['severity']))}",
                    unsafe_allow_html=True,
                )
                with st.expander("Show reasoning", expanded=False):
                    if result["ok"]:
                        render_pipeline_reasoning(result["output"])
                    else:
                        st.info(result["error"])

        st.markdown('<div class="ask-input-panel">', unsafe_allow_html=True)
        seeded = st.session_state.chat_input_seed
        if seeded and not st.session_state.ask_inline_input:
            st.session_state.ask_inline_input = seeded
            st.session_state.chat_input_seed = ""
        user_question = ""
        with st.form("ask_inline_form", clear_on_submit=True):
            input_col, action_col = st.columns([12, 1], gap="small")
            with input_col:
                st.text_input(
                    "Ask about medication interactions...",
                    placeholder="Ask about medication interactions...",
                    label_visibility="collapsed",
                    key="ask_inline_input",
                )
            with action_col:
                submitted = st.form_submit_button("Send", use_container_width=True)
            if submitted:
                user_question = st.session_state.ask_inline_input.strip()
        st.caption("This is a demonstration tool. Always consult healthcare professionals for medical advice.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "Running the three-stage pipeline...", "pending": True}
        )
        st.session_state.pending_question = user_question
        st.rerun()

    if st.session_state.pending_question:
        pending_question = st.session_state.pending_question
        result = process_question(pending_question)
        if st.session_state.chat_history and st.session_state.chat_history[-1].get("pending"):
            if result is not None:
                st.session_state.chat_history[-1] = {
                    "role": "assistant",
                    "content": result["summary"],
                    "result": result,
                }
            else:
                st.session_state.chat_history[-1] = {
                    "role": "assistant",
                    "content": "I couldn't process that question.",
                }
        st.session_state.pending_question = ""
        st.rerun()


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-icon">⌕</div>
            <div>
                <h1 class="app-header-title">Medication Safety Assistant</h1>
                <p class="app-header-subtitle">Professional medication interaction checker</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_tab() -> None:
    st.markdown(
        f"""
        <div class="dashboard-hero">
            <div class="dashboard-icon">🛡</div>
            <h1 class="dashboard-title">{APP_TITLE}</h1>
            <p class="dashboard-subtitle">{APP_SUBTITLE} {HOME_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
            .st-key-open-my-medicines-dashboard button {
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            .st-key-open-my-medicines-dashboard button:hover {
                background: #1d4ed8;
                border-color: #1d4ed8;
                color: #ffffff;
            }
            .st-key-open-ask-question-dashboard button {
                background: #16a34a;
                color: #ffffff;
                border: 1px solid #16a34a;
            }
            .st-key-open-ask-question-dashboard button:hover {
                background: #15803d;
                border-color: #15803d;
                color: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="dashboard-card-icon card-icon-blue">💊</div>
                <h3 class="dashboard-card-title">My Medicines</h3>
                <p class="dashboard-card-text">
                    Manage your current medications and check for potential interactions between them automatically.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Medicine List", key="open-my-medicines-dashboard", use_container_width=True, type="primary"):
            st.session_state.active_page = "My Medicines"
            st.rerun()
    with col2:
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="dashboard-card-icon card-icon-green">💬</div>
                <h3 class="dashboard-card-title">Ask a Question</h3>
                <p class="dashboard-card-text">
                    Get answers to your medication safety questions through our AI-powered chat assistant.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Conversation", key="open-ask-question-dashboard", use_container_width=True, type="primary"):
            st.session_state.active_page = "Ask a Question"
            st.rerun()
    st.markdown(
        """
        <div class="safety-info-card">
            <div class="safety-info-wrap">
                <div class="safety-info-icon">🛡</div>
                <div>
                    <div class="safety-info-title">Important Safety Information</div>
                    <p class="safety-info-text">
                        This tool is designed to assist with medication safety awareness. Always consult with your
                        healthcare provider or pharmacist before making any changes to your medication regimen.
                        This tool does not replace professional medical advice.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_session_state()
    inject_styles()
    render_app_header()
    render_navigation()
    active_page = st.session_state.active_page
    if active_page == "My Medicines":
        render_my_medicines_tab()
    elif active_page == "Ask a Question":
        render_ask_question_tab()
    else:
        render_home_tab()


if __name__ == "__main__":
    main()
