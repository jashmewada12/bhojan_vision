import json

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & SHADCN-INSPIRED DESIGN SYSTEM (NO PURPLE / CLEAN ZINC THEME)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BhojanVisionv1 — Precision Nutrition AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS implementing shadcn/ui minimal aesthetics
st.markdown(
    """
<style>
    /* Vibrant Dark Teal-Emerald Layered Ambient Gradient */
    .stApp {
        background:
            radial-gradient(circle at 50% -15%, rgba(16, 185, 129, 0.28) 0%, transparent 50%),
            radial-gradient(circle at 90% 10%, rgba(20, 184, 166, 0.18) 0%, transparent 40%),
            radial-gradient(circle at 10% 85%, rgba(13, 148, 136, 0.22) 0%, transparent 45%),
            linear-gradient(165deg, #091a1a 0%, #061317 40%, #03080b 100%);
        background-attachment: fixed;
        color: #f4f4f5;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Responsive Top Header Bar */
    .header-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.75rem;
    }
    .header-title {
        font-size: clamp(1.25rem, 5vw, 1.75rem);
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #fafafa;
        margin: 0;
        white-space: nowrap;
        word-break: keep-all;
    }
    .status-badge {
        font-size: 0.725rem;
        font-weight: 500;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background-color: rgba(6, 78, 59, 0.7);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        backdrop-filter: blur(8px);
        white-space: nowrap;
        align-self: center;
    }

    /* Translucent Card with Frosted Depth */
    .card {
        background: rgba(13, 23, 28, 0.68);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
    }

    /* Metric Indicators */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
    }
    .metric-box {
        background: rgba(15, 28, 34, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 0.5rem;
        padding: 0.85rem;
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #fafafa;
        margin-top: 0.15rem;
    }
    .metric-val.emerald {
        color: #10b981;
    }

    /* File uploader & Camera Overrides */
    div[data-testid="stFileUploader"] section {
        background-color: rgba(15, 28, 34, 0.5);
        border: 1px dashed rgba(255, 255, 255, 0.18);
        border-radius: 0.75rem;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #10b981;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #f1f5f9;
        color: #091316;
        border-radius: 0.5rem;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton>button:hover {
        background-color: #e2e8f0;
        color: #000;
    }

    /* Mobile Layout Refinements */
    @media (max-width: 640px) {
        .header-container {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
        .header-title {
            font-size: 1.35rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. MODEL ASSETS & CLASS MAPPING
# -----------------------------------------------------------------------------
CLASS_NAMES = [
    "burger",
    "butter_naan",
    "chai",
    "chapati",
    "chole_bhature",
    "dal_makhani",
    "dhokla",
    "fried_rice",
    "idli",
    "jalebi",
    "kaathi_rolls",
    "kadai_paneer",
    "kulfi",
    "masala_dosa",
    "momos",
    "paani_puri",
    "pakode",
    "pav_bhaji",
    "pizza",
    "samosa",
]


@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession(
        "models/bhojan_vision.onnx", providers=["CPUExecutionProvider"]
    )
    with open("database.json", "r") as f:
        nutrition_db = json.load(f)
    return session, nutrition_db


try:
    session, NUTRITION_DB = load_onnx_model()
    model_ready = True
except Exception as e:
    model_ready = False
    model_error = str(e)


# -----------------------------------------------------------------------------
# 3. PREPROCESSING FUNCTION
# -----------------------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((288, 288))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0)


# -----------------------------------------------------------------------------
# 4. USER INTERFACE LAYOUT
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="header-container">
    <div>
        <h1 class="header-title">BhojanVision</h1>
        <p style="color: #71717a; margin: 0.15rem 0 0 0; font-size: 0.875rem;">
            Real-time inference & nutritional analytics for Indian cuisine
        </p>
    </div>
    <div class="status-badge">● ONNX Runtime Active</div>
</div>
""",
    unsafe_allow_html=True,
)

if not model_ready:
    st.error(f"Failed to load ONNX model. Check file path: {model_error}")
    st.stop()

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown(
        '<div class="card-title">Image Acquisition</div>', unsafe_allow_html=True
    )

    # Clean input selector
    input_method = st.radio(
        label="Input Source",
        options=["Upload File", "Take Photo"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_image = None
    if input_method == "Upload File":
        uploaded = st.file_uploader(
            "Select an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw_image = Image.open(uploaded)
    else:
        captured = st.camera_input("Capture item", label_visibility="collapsed")
        if captured:
            raw_image = Image.open(captured)

    if raw_image:
        st.markdown(
            '<div class="card" style="padding: 0.5rem; margin-top: 1rem;">',
            unsafe_allow_html=True,
        )
        st.image(raw_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown(
        '<div class="card-title">Prediction & Nutrition Engine</div>',
        unsafe_allow_html=True,
    )

    if raw_image is not None:
        # Run Inference
        with st.spinner("Classifying image tensor..."):
            input_tensor = preprocess(raw_image)
            input_name = session.get_inputs()[0].name
            logits = session.run(None, {input_name: input_tensor})[0][0]

            # Softmax calculation for confidence
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx]) * 100
            pred_class = CLASS_NAMES[pred_idx]
            food_info = NUTRITION_DB.get(pred_class, None)

        if food_info:
            # Prediction Overview Card
            st.markdown(
                f"""
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="font-size: 1.4rem; font-weight: 700; color: #fafafa;">
                        {pred_class.replace("_", " ").title()}
                    </span>
                    <span style="color: #10b981; font-weight: 600; font-size: 0.875rem;">
                        {confidence:.1f}% confidence
                    </span>
                </div>
                <div style="margin-top: 0.5rem; height: 4px; background: #27272a; border-radius: 2px;">
                    <div style="height: 100%; width: {min(confidence, 100):.1f}%; background: #10b981; border-radius: 2px;"></div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Initialize macro variables
            total_cals = 0.0
            total_protein = 0.0
            total_carbs = 0.0
            total_fat = 0.0
            display_title = ""

            food_type = food_info.get("type", "standard")

            if food_type == "compound":
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="card-title">Adjust Plate Components</div>',
                    unsafe_allow_html=True,
                )

                components = food_info["components"]
                cols = st.columns(len(components))

                for idx, (comp_name, comp_data) in enumerate(components.items()):
                    with cols[idx]:
                        unit = comp_data["unit"]
                        base_qty = comp_data["base_qty"]

                        qty = st.number_input(
                            label=f"{comp_name.replace('_', ' ').title()} ({unit})",
                            min_value=0.0,
                            value=float(base_qty),
                            step=1.0 if unit in ["pieces", "plate", "slice"] else 25.0,
                            key=f"input_{pred_class}_{comp_name}",
                        )

                        scale = qty / base_qty if base_qty > 0 else 0
                        total_cals += comp_data["calories"] * scale
                        total_protein += comp_data["protein"] * scale
                        total_carbs += comp_data["carbs"] * scale
                        total_fat += comp_data["fat"] * scale

                display_title = "Custom Plate Breakdown"
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                unit = food_info["unit"]
                base_qty = food_info["base_qty"]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="card-title">Serving Size ({unit})</div>',
                    unsafe_allow_html=True,
                )

                qty = st.number_input(
                    label=f"Enter quantity in {unit}",
                    min_value=0.1,
                    value=float(base_qty),
                    step=1.0 if unit in ["pieces", "plate", "slice"] else 25.0,
                    label_visibility="collapsed",
                    key=f"input_{pred_class}_standard",
                )
                st.markdown("</div>", unsafe_allow_html=True)

                scale = qty / base_qty if base_qty > 0 else 0
                total_cals = food_info["calories"] * scale
                total_protein = food_info["protein"] * scale
                total_carbs = food_info["carbs"] * scale
                total_fat = food_info["fat"] * scale
                display_title = f"Nutritional Breakdown for {qty:g} {unit}"

            # Nutritional Breakdown Metric Grid
            st.markdown(
                f"""
            <div class="card">
                <div class="card-title">{display_title}</div>
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-label">Energy</div>
                        <div class="metric-val emerald">{round(total_cals, 1)} <span style="font-size: 0.8rem; color: #71717a;">kcal</span></div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Protein</div>
                        <div class="metric-val">{round(total_protein, 1)} <span style="font-size: 0.8rem; color: #71717a;">g</span></div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Carbohydrates</div>
                        <div class="metric-val">{round(total_carbs, 1)} <span style="font-size: 0.8rem; color: #71717a;">g</span></div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Total Fats</div>
                        <div class="metric-val">{round(total_fat, 1)} <span style="font-size: 0.8rem; color: #71717a;">g</span></div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """
        <div class="card" style="text-align: center; padding: 3rem 1rem;">
            <p style="color: #71717a; margin: 0; font-size: 0.9rem;">
                Awaiting input image.<br>Use the left panel to upload a file or take a photo.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# 5. FOOTER & PORTFOLIO LINKS
# -----------------------------------------------------------------------------
st.markdown(
    """
<div style="margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.1); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
    <div style="color: #94a3b8; font-size: 0.85rem;">
        Built by <span style="font-weight: 600; color: #f4f4f5;">Jash Mewada</span>
    </div>
    <div style="display: flex; gap: 1.5rem; align-items: center;">
        <a href="https://github.com/jashmewada12" target="_blank" class="footer-link">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
            </svg>
            GitHub
        </a>
        <a href="https://www.linkedin.com/in/jash-mewada-86aa252b6/" target="_blank" class="footer-link">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                <rect x="2" y="9" width="4" height="12"></rect>
                <circle cx="4" cy="4" r="2"></circle>
            </svg>
            LinkedIn
        </a>
    </div>
</div>

<style>
    /* Footer link hover animations */
    .footer-link {
        color: #94a3b8;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .footer-link:hover {
        color: #10b981;
        transform: translateY(-1px);
    }
</style>
""",
    unsafe_allow_html=True,
)
