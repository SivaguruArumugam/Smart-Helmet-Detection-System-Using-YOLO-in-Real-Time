"""
dashboard.py – Smart Helmet Detection System
Teal / Blue / Purple Theme — Clean single-page layout.
Run: streamlit run src/dashboard.py
"""

import os, sys, cv2, numpy as np, pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

try:
    torch.serialization.add_safe_globals([DetectionModel])
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from utils import draw_boxes
from compliance import check_helmet_compliance, ViolationLogger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Helmet Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ─ Reset ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
.stApp { background: #0b0c2a; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }

/* ─ Page padding ─────────────────────────────────────── */
.block-container {
    padding: 2.5rem 4rem 4rem !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

/* ─ Hero ─────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1a3a8f 0%, #3d1080 100%);
    border-radius: 20px;
    border-top: 4px solid #6db5b0;
    padding: 40px 52px;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(109,181,176,0.08);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(109,181,176,0.15);
    color: #6db5b0;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(109,181,176,0.3);
    margin-bottom: 16px;
}
.hero h1 {
    color: #e8f4f8;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 10px;
    line-height: 1.2;
}
.hero p {
    color: #6db5b0;
    font-size: 0.88rem;
    margin: 0;
    letter-spacing: 0.02em;
    line-height: 1.6;
}

/* ─ Section label ────────────────────────────────────── */
.sec-label {
    color: #6db5b0;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 0 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1a3a8f, transparent);
}

/* ─ Drag & Drop Zone — Reference Style ──────────────── */

/* Reset all inner Streamlit chrome */
[data-testid="stFileUploader"] *         { box-shadow: none !important; }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Outer card — clean, no dashed border */
[data-testid="stFileUploader"] {
    background: #0f1535 !important;
    border: 1px solid rgba(109,181,176,0.18) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 4px 28px rgba(0,0,0,0.35) !important;
    transition: box-shadow 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    box-shadow: 0 6px 36px rgba(10,138,138,0.18) !important;
}

/* Inner dashed box — position:relative so button can be pinned */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(13,22,64,0.55) !important;
    border: 2px dashed rgba(10,138,138,0.55) !important;
    border-radius: 12px !important;
    padding: 56px 32px 64px !important;
    margin: 0 !important;
    width: 100% !important;
    min-height: 210px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    transition: all 0.25s ease !important;
    position: relative !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(10,138,138,0.06) !important;
    border-color: #6db5b0 !important;
}

/* Instructions wrapper — force column + center */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    width: 100% !important;
    gap: 0 !important;
}

/* Cloud / image icon */
[data-testid="stFileUploaderDropzone"] svg {
    width: 58px !important;
    height: 58px !important;
    fill: #0a8a8a !important;
    color: #0a8a8a !important;
    display: block !important;
    margin: 0 auto 16px !important;
}

/* "Drag and Drop" main text */
[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #e8f4f8 !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    display: block !important;
    text-align: center !important;
    width: 100% !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}

/* "Limit 200MB" sub-text */
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #4a6a9a !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
    margin-top: 8px !important;
}

/* "Browse files" — pushed to the right side */
[data-testid="stFileUploaderDropzone"] button {
    position: static !important;
    align-self: flex-end !important;
    margin-left: auto !important;
    margin-right: 8px !important;
    margin-top: 20px !important;
    background: transparent !important;
    color: #0a8a8a !important;
    border: 1px solid rgba(10,138,138,0.5) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 6px 20px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(10,138,138,0.12) !important;
    border-color: #6db5b0 !important;
    color: #6db5b0 !important;
    transform: none !important;
}

/* ─ Image captions ───────────────────────────────────── */
.img-cap {
    color: #6db5b0;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: center;
    margin-bottom: 10px;
}

/* ─ Divider ──────────────────────────────────────────── */
.hr {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1a3a8f 30%, #3d1080 70%, transparent);
    margin: 32px 0;
    border: none;
}

/* ─ Metric cards ─────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #0f1535;
    border: 1px solid #1a3a8f;
    border-radius: 16px;
    padding: 28px 20px !important;
    text-align: center;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #0a8a8a; }
[data-testid="stMetricLabel"] {
    color: #6db5b0 !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    justify-content: center;
    margin-bottom: 6px !important;
}
[data-testid="stMetricValue"] {
    color: #e8f4f8 !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    justify-content: center;
    line-height: 1;
}
[data-testid="stMetricDelta"] { justify-content: center; font-size: 0.78rem !important; }

/* ─ Alert banners ────────────────────────────────────── */
.banner {
    border-radius: 14px;
    padding: 20px 32px;
    font-weight: 700;
    font-size: 1rem;
    text-align: center;
    margin-top: 24px;
    letter-spacing: 0.01em;
}
.banner-danger {
    background: linear-gradient(90deg, #5c0f0f, #a31515);
    border: 1px solid #d63031;
    color: #ffe8e8;
    animation: pulse 1.6s ease-in-out infinite;
}
.banner-safe {
    background: linear-gradient(90deg, #0a3a2a, #0a8a4a);
    border: 1px solid #00b894;
    color: #e0fff4;
}
.banner-info {
    background: #0f1535;
    border: 1px dashed #1a3a8f;
    color: #6db5b0;
    font-weight: 500;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.8} }

/* ─ Expander ─────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0f1535;
    border: 1px solid #1a3a8f !important;
    border-radius: 12px;
    margin-top: 24px;
}
[data-testid="stExpander"] summary { color: #6db5b0 !important; font-weight: 600; }

/* ─ DataFrame ────────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #1a3a8f; }

/* ─ Spinner ──────────────────────────────────────────── */
.stSpinner > div { border-top-color: #6db5b0 !important; }

/* ─ Text ─────────────────────────────────────────────── */
h2,h3,h4 { color: #e8f4f8 !important; }
p, li     { color: #9ab8d4; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">🦺 AI Safety System</div>
  <h1>Smart Helmet Detection System</h1>
  <p>Powered by YOLOv8 · Real-Time Computer Vision · Instant Compliance Reporting</p>
</div>
""", unsafe_allow_html=True)

# ── Model (auto, cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading YOLOv8 model …")
def load_model():
    path = "outputs/helmet_detection_custom/weights/best.pt"
    return YOLO(path if os.path.exists(path) else "yolov8n.pt")

model  = load_model()
logger = ViolationLogger()

def classify(name: str):
    n = name.lower()
    if "person" in n or "human" in n: return "person"
    if ("no" in n and ("helmet" in n or "hardhat" in n)) or "without" in n: return "violation"
    if "helmet" in n or "hardhat" in n or "with helmet" in n: return "compliant"
    return None

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">Upload Image for Inspection</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("_", type=["jpg","jpeg","png"], label_visibility="collapsed")

# JS: move Browse Files button to bottom-right of the dashed box
components.html("""
<script>
(function() {
    function moveBtn() {
        var doc = window.parent.document;
        var zone = doc.querySelector('[data-testid="stFileUploaderDropzone"]');
        if (!zone) { setTimeout(moveBtn, 150); return; }
        var btn = zone.querySelector('button');
        if (!btn) { setTimeout(moveBtn, 150); return; }
        zone.style.position = 'relative';
        btn.style.cssText = [
            'position:absolute !important',
            'bottom:14px !important',
            'right:16px !important',
            'top:auto !important',
            'left:auto !important',
            'margin:0 !important',
            'align-self:unset !important',
            'background:transparent !important',
            'color:#0a8a8a !important',
            'border:1px solid rgba(10,138,138,0.55) !important',
            'border-radius:8px !important',
            'font-size:0.82rem !important',
            'font-weight:700 !important',
            'padding:6px 20px !important',
            'cursor:pointer !important',
            'transition:all 0.2s ease !important',
        ].join(';');
        btn.onmouseenter = function(){ this.style.background='rgba(10,138,138,0.12)'; this.style.borderColor='#6db5b0'; this.style.color='#6db5b0'; };
        btn.onmouseleave = function(){ this.style.background='transparent'; this.style.borderColor='rgba(10,138,138,0.55)'; this.style.color='#0a8a8a'; };
    }
    moveBtn();
    var obs = new MutationObserver(moveBtn);
    obs.observe(window.parent.document.body, { childList:true, subtree:true });
})();
</script>
""", height=0)

# Footer hint
if uploaded is None:
    st.markdown("""
<p style="text-align:center;color:#3a5a9a;font-size:0.78rem;margin-top:10px;">
  Supports JPG &amp; PNG · Fully automatic detection · No settings required
</p>""", unsafe_allow_html=True)

# ── Detection ─────────────────────────────────────────────────────────────────
if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("🔍 AI is scanning the image …"):
        results = model(frame, verbose=False, conf=0.10, imgsz=640)

    persons, compliant, non_compliant, helmets = [], [], [], []
    for r in results:
        for box in r.boxes:
            kind = classify(model.names[int(box.cls[0])])
            bbox = box.xyxy[0].cpu().numpy()
            if   kind == "person":    persons.append(bbox)
            elif kind == "violation": non_compliant.append(bbox)
            elif kind == "compliant": compliant.append(bbox); helmets.append(bbox)

    if persons and not compliant and not non_compliant:
        compliant, non_compliant = check_helmet_compliance(persons, helmets)

    total     = len(persons) if persons else len(compliant) + len(non_compliant)
    out_frame = draw_boxes(frame, compliant, non_compliant, helmets)
    out_rgb   = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)

    # ── Divider ───────────────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Images ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<p class="img-cap">📁 Original Image</p>', unsafe_allow_html=True)
        st.image(pil_img, use_column_width=True)
    with c2:
        st.markdown('<p class="img-cap">🤖 AI Detection Output</p>', unsafe_allow_html=True)
        st.image(out_rgb, use_column_width=True)

    # ── Divider ───────────────────────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2 = st.columns(2, gap="medium")
    m1.metric("✅ Safe Workers",    len(compliant),
               delta="With Helmet" if compliant else None)
    m2.metric("⚠️ Violations",      len(non_compliant),
               delta="No Helmet"   if non_compliant else None, delta_color="inverse")

    # ── Banner ────────────────────────────────────────────────────────────────
    if non_compliant:
        st.markdown(
            f'<div class="banner banner-danger">🚨 SAFETY VIOLATION — '
            f'{len(non_compliant)} Worker(s) Detected Without a Helmet!</div>',
            unsafe_allow_html=True)
    elif total > 0:
        st.markdown(
            '<div class="banner banner-safe">✅ All Workers Are Wearing Helmets — Workplace Is SAFE</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="banner banner-info">ℹ️ No workers detected. Try a clearer or closer image.</div>',
            unsafe_allow_html=True)

    # ── Debug ─────────────────────────────────────────────────────────────────
    with st.expander("🔎 View Raw AI Detections"):
        rows = [{"Class": model.names[int(b.cls[0])],
                 "Confidence": f"{float(b.conf[0]):.2%}",
                 "X1":int(b.xyxy[0][0]),"Y1":int(b.xyxy[0][1]),
                 "X2":int(b.xyxy[0][2]),"Y2":int(b.xyxy[0][3])}
                for r in results for b in r.boxes]
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame(), use_container_width=True)
