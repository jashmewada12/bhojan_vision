import json

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Indian Food Macro Tracker", page_icon="🍛", layout="centered"
)

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
def load_assets():
    session = ort.InferenceSession(
        "models/bhojan_vision.onnx", providers=["CPUExecutionProvider"]
    )
    with open("nutrition_data.json", "r") as f:
        nutrition_db = json.load(f)
    return session, nutrition_db


session, nutrition_db = load_assets()


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((288, 288))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0)


st.title("Indian Food Macro Tracker")
st.write("Upload an image of your food and specify the serving size.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 1. Preprocess & Run ONNX Inference
    input_tensor = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    predicted_idx = int(np.argmax(outputs[0]))
    predicted_food = CLASS_NAMES[predicted_idx]
    food_data = nutrition_db[predicted_food]

    st.success(f"**Detected Item:** {predicted_food.replace('_', ' ').title()}")

    # 2. Dynamic Input Locked to Required Unit
    st.subheader("Portion Size")
    unit = food_data["unit"]
    base_qty = food_data["base_qty"]

    amount = st.number_input(
        f"Enter amount in {unit}:",
        min_value=0.1,
        value=float(base_qty),
        step=1.0 if unit == "pieces" else 10.0,
    )

    # 3. Calculate and Render Macros
    multiplier = amount / base_qty
    cals = round(food_data["calories"] * multiplier, 1)
    protein = round(food_data["protein"] * multiplier, 1)
    carbs = round(food_data["carbs"] * multiplier, 1)
    fat = round(food_data["fat"] * multiplier, 1)

    st.markdown("---")
    st.subheader("Nutritional Breakdown")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Calories", f"{cals} kcal")
    col2.metric("Protein", f"{protein} g")
    col3.metric("Carbs", f"{carbs} g")
    col4.metric("Fat", f"{fat} g")
