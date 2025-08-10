import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Load models
lung_model = tf.keras.models.load_model("./models/lung.hdf5")
kidney_model = tf.keras.models.load_model("./models/Kidney_tumor.hdf5")
brain_model = tf.keras.models.load_model("./models/Brain_Tumor.hdf5")

# Define your Cancer classes (same as your original)
# lung_cancer, kidney_cancer, brain_cancer definitions here...

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((350, 350))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image, cancer_type):
    if cancer_type.name == "Lung Cancer":
        model = lung_model
    elif cancer_type.name == "Kidney Cancer":
        model = kidney_model
    else:
        model = brain_model

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    class_index = np.argmax(prediction)
    probability = prediction[class_index]
    predicted_class = cancer_type.labels[class_index]

    return predicted_class, probability, prediction

def main():
    st.set_page_config(
        page_title="NMIMR MEDICT",
        page_icon="🩺",
        layout="wide"
    )

    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

    # CSS Styles
    background = "#121212" if dark_mode else "#f5f7fa"
    text_color = "#ffffff" if dark_mode else "#333333"
    card_bg = "#1e1e1e" if dark_mode else "white"
    card_shadow = "0 4px 12px rgba(0,0,0,0.5)" if dark_mode else "0 4px 12px rgba(0,0,0,0.1)"

    st.markdown(f"""
        <style>
        body {{
            background-color: {background};
            color: {text_color};
        }}
        .main-title {{
            text-align: center;
            font-size: 2.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            text-align: center;
            opacity: 0.8;
            margin-bottom: 2rem;
        }}
        .result-card {{
            background: {card_bg};
            padding: 1rem;
            border-radius: 12px;
            box-shadow: {card_shadow};
            text-align: center;
            margin-bottom: 1rem;
        }}
        .Precaution {{
            background: #fff4e6;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">NMIMR MEDICT</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered Cancer Detection & Insights</div>', unsafe_allow_html=True)

    # Sidebar selection
    cancer_types = [lung_cancer, kidney_cancer, brain_cancer]
    options = [c.name for c in cancer_types]
    selected_option = st.sidebar.selectbox("🔍 Select Cancer Type", options)
    selected_cancer_type = next(c for c in cancer_types if c.name == selected_option)
    st.sidebar.markdown(f"**About:** {selected_cancer_type.description}")

    uploaded_file = st.file_uploader("📤 Upload a medical scan image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("🔄 Analyzing image..."):
                time.sleep(1)  # Small delay for animation effect
                predicted_class, probability, full_probs = predict(image, selected_cancer_type)

            # Result card
            st.markdown(f"""
                <div class="result-card">
                    <h4>{predicted_class}</h4>
                    <p><b>Confidence:</b> {probability:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            # Animated progress bars with color-coded levels
            st.write("### Prediction Confidence")
            for i, label in enumerate(selected_cancer_type.labels):
                prob = float(full_probs[i])
                color = "green" if prob < 0.5 else ("orange" if prob < 0.8 else "red")
                st.progress(prob)  # Animated bar
                st.markdown(f"**{label}** — {prob:.2%}", unsafe_allow_html=True)

            # Messages
            if predicted_class not in ["Normal", "no_tumor"]:
                st.warning(selected_cancer_type.true_positive_descriptions[predicted_class])
                st.info("Please consult a doctor for further evaluation and treatment.")
            else:
                st.success(selected_cancer_type.true_negative_description)

            # Precautions in expander
            if predicted_class in selected_cancer_type.precautions:
                with st.expander("🛡 Precautionary Measures"):
                    st.markdown(f"<div class='Precaution'>{selected_cancer_type.precautions[predicted_class]}</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; opacity:0.6;">© 2025 Justice Ohene Amofa</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
