import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# -------------------
# Load models (only kidney and brain)
# -------------------
kidney_model = tf.keras.models.load_model("./models/Kidney_tumor.hdf5")
brain_model = tf.keras.models.load_model("./models/Brain_Tumor.hdf5")

# -------------------
# Cancer class + definitions
# -------------------
class Cancer:
    def __init__(
        self,
        name,
        description,
        labels,
        true_positive_descriptions,
        true_negative_description,
        precautions,
    ):
        self.name = name
        self.description = description
        self.labels = labels
        self.true_positive_descriptions = true_positive_descriptions
        self.true_negative_description = true_negative_description
        self.precautions = precautions

# --- kidney_cancer ---
kidney_cancer = Cancer(
    name="Kidney Cancer",
    description="Kidney cancer, medically termed renal cancer, originates within the kidneys. The predominant form is renal cell carcinoma (RCC), accounting for the majority of cases. It typically begins in the lining of the renal tubules and can grow and spread to other parts of the body if not detected early. Symptoms may include blood in the urine, lower back pain, or a mass in the abdomen. Treatment options vary based on the stage and location of the cancer, including surgery, targeted therapy, immunotherapy, or radiation therapy. Regular medical check-ups are crucial for early detection and management of kidney cancer.",
    labels={0: "Cyst", 1: "Normal", 2: "Stone", 3: "Tumor"},
    true_positive_descriptions={
        "Cyst": "The image indicates the presence of a cyst in the kidney. It is recommended to seek medical attention.",
        "Stone": "The image suggests the presence of a kidney stone. It is recommended to seek medical attention.",
        "Tumor": "The image indicates the presence of a kidney tumor. It is recommended to seek medical attention.",
    },
    true_negative_description="No evidence of kidney cancer is found in the image. Maintaining a healthy lifestyle is encouraged.",
    precautions={
        "Cyst": """
            <ol>
                <li><strong>Limit consumption of processed and smoked foods</strong>, maintain a healthy weight, stay hydrated, and follow your doctor's recommendations:
                    <ul>
                        <li><strong>Processed and smoked foods:</strong> These can contain additives and compounds that may not be beneficial for overall health, including kidney health.</li>
                        <li><strong>Healthy weight:</strong> Obesity can contribute to various health issues, including kidney health. Maintaining a healthy weight through a balanced diet and regular physical activity can help.</li>
                        <li><strong>Stay hydrated:</strong> Adequate hydration is important for overall kidney function. It helps the kidneys clear sodium and toxins from the body.</li>
                        <li><strong>Follow doctor's recommendations:</strong> Regular check-ups and medical advice are crucial for monitoring kidney health and addressing any concerns early.</li>
                    </ul>
                </li>
                <li><strong>Maintain a healthy weight</strong> through regular physical activity and a balanced diet:
                    <ul>
                        <li><strong>Physical activity:</strong> Regular exercise can help maintain a healthy weight and promote overall health.</li>
                        <li><strong>Balanced diet:</strong> A diet rich in fruits, vegetables, whole grains, and lean proteins can support kidney health.</li>
                    </ul>
                </li>
                <li><strong>Stay hydrated</strong> by drinking plenty of water throughout the day:
                    <ul>
                        <li>Water helps the kidneys remove waste from the blood in the form of urine. Staying hydrated reduces the risk of kidney stones and supports overall kidney function.</li>
                    </ul>
                </li>
                <li><strong>Follow your doctor's recommendations</strong> regarding regular check-ups and medical advice:
                    <ul>
                        <li>Regular check-ups can help detect any kidney issues early. Your doctor may recommend specific tests or medications based on your individual health needs.</li>
                    </ul>
                </li>
            </ol>
        """,
        "Stone": """
            <ol>
                <li><strong>Increase fluid intake</strong> to help prevent the formation of kidney stones:
                    <ul>
                        <li>Drinking plenty of water dilutes the substances in urine that lead to stones. Aim for at least 8 glasses (64 ounces) of water per day, or more depending on your activity level and climate.</li>
                    </ul>
                </li>
                <li><strong>Limit sodium and protein intake</strong> to reduce the risk of stone formation:
                    <ul>
                        <li>Too much sodium can cause calcium to build up in your urine, while excessive protein can lead to increased uric acid levels, both of which can contribute to stone formation.</li>
                    </ul>
                </li>
                <li><strong>Avoid foods high in oxalates</strong> such as spinach, beets, and nuts:
                    <ul>
                        <li>Oxalates can bind with calcium in the urine to form kidney stones. Reducing intake of these foods can help lower your risk.</li>
                    </ul>
                </li>
                <li><strong>Follow your doctor's recommendations</strong> for dietary adjustments and medication:
                    <ul>
                        <li>Your doctor may recommend specific dietary changes or medications to prevent stones based on the type of stone you have and your medical history.</li>
                    </ul>
                </li>
            </ol>
        """,
        "Tumor": """
            <ol>
                <li><strong>Limit consumption of processed and smoked foods</strong>, maintain a healthy weight, stay hydrated, and follow your doctor's recommendations:
                    <ul>
                        <li><strong>Processed and smoked foods:</strong> Similar to cysts, these can contain additives and compounds that may not be beneficial for overall health.</li>
                        <li><strong>Healthy weight:</strong> Maintaining a healthy weight through diet and exercise can support overall health and recovery from treatment.</li>
                        <li><strong>Stay hydrated:</strong> Adequate hydration is important for supporting overall health, especially during treatment.</li>
                        <li><strong>Follow doctor's recommendations:</strong> Regular check-ups and medical advice are crucial during treatment and recovery.</li>
                    </ul>
                </li>
                <li><strong>Maintain a healthy weight</strong> through regular physical activity and a balanced diet:
                    <ul>
                        <li><strong>Physical activity:</strong> Physical activity can help maintain strength and overall health during treatment and recovery.</li>
                        <li><strong>Balanced diet:</strong> A balanced diet provides essential nutrients needed for healing and recovery.</li>
                    </ul>
                </li>
                <li><strong>Stay hydrated</strong> by drinking plenty of water throughout the day:
                    <ul>
                        <li>Staying hydrated supports overall health and can help manage side effects of treatment.</li>
                    </ul>
                </li>
                <li><strong>Follow your doctor's recommendations</strong> regarding regular check-ups and medical advice:
                    <ul>
                        <li>Regular check-ups and monitoring are essential during treatment to assess the effectiveness of treatment and manage any side effects.</li>
                    </ul>
                </li>
            </ol>
        """
        ,
    },
)

# --- brain_cancer ---
brain_cancer = Cancer(
    name="Brain Tumor",
    description="Brain tumors are abnormal growths of cells that can develop in the brain or central spine. These tumors can either be cancerous (malignant) or non-cancerous (benign). Malignant brain tumors are more aggressive and can invade nearby tissues, making them potentially life-threatening. Benign tumors, while generally less aggressive, can still cause problems depending on their size and location. Symptoms of brain tumors vary depending on their size, location, and rate of growth, and may include headaches, seizures, behavioral changes, or problems with vision or speech. Treatment options typically include surgery, radiation therapy, and chemotherapy, tailored to the specific type and location of the tumor. Regular monitoring and follow-up are essential to manage symptoms and monitor for recurrence.",
    labels={
        0: "no_tumor",
        1: "pituitary_tumor",
        2: "meningioma_tumor",
        3: "glioma_tumor",
    },
    true_positive_descriptions={
        "pituitary_tumor": "The image suggests the presence of a pituitary tumor. Seeking prompt medical care is advised.",
        "meningioma_tumor": "The image indicates the presence of a meningioma tumor. Seeking prompt medical care is advised.",
        "glioma_tumor": "The image suggests the presence of a glioma tumor. Seeking prompt medical care is advised.",
    },
    true_negative_description="The image does not indicate the presence of a brain tumor. Nevertheless, regular monitoring is advisable.",
    precautions={
        "pituitary_tumor": """
            <ol>
                <li><strong>Reduce exposure to radiation and harmful chemicals</strong>, manage stress levels, maintain a balanced diet, and follow your doctor's recommendations:
                    <ul>
                        <li><strong>Reduce exposure to radiation and harmful chemicals:</strong> Minimize exposure to environmental toxins and radiation, which may contribute to tumor growth.</li>
                        <li><strong>Manage stress levels:</strong> Activities such as meditation, yoga, or therapy can help reduce stress, which may impact tumor growth.</li>
                        <li><strong>Maintain a balanced diet:</strong> Include plenty of fruits, vegetables, and whole grains in your diet to support overall health.</li>
                        <li><strong>Follow doctor's recommendations:</strong> Regular check-ups and medical advice are crucial for monitoring tumor growth and managing symptoms.</li>
                    </ul>
                </li>
                <li><strong>Manage stress levels</strong> through activities such as meditation, yoga, or therapy:
                    <ul>
                        <li>Stress management techniques can help improve overall well-being and may impact tumor growth.</li>
                    </ul>
                </li>
                <li><strong>Maintain a balanced diet</strong> with plenty of fruits, vegetables, and whole grains:
                    <ul>
                        <li>A balanced diet provides essential nutrients and supports overall health.</li>
                    </ul>
                </li>
                <li><strong>Follow your doctor's recommendations</strong> regarding regular check-ups and medical advice:
                    <ul>
                        <li>Regular check-ups are important for monitoring tumor growth and adjusting treatment as needed.</li>
                    </ul>
                </li>
            </ol>
        """,
        "meningioma_tumor": """
            <ol>
                <li><strong>Reduce exposure to radiation and harmful chemicals</strong>, manage stress levels, maintain a balanced diet, and follow your doctor's recommendations:
                    <ul>
                        <li><strong>Reduce exposure to radiation and harmful chemicals:</strong> Minimize exposure to environmental toxins and radiation, which may contribute to tumor growth.</li>
                        <li><strong>Manage stress levels:</strong> Activities such as meditation, yoga, or therapy can help reduce stress, which may impact tumor growth.</li>
                        <li><strong>Maintain a balanced diet:</strong> Include plenty of fruits, vegetables, and whole grains in your diet to support overall health.</li>
                        <li><strong>Follow doctor's recommendations:</strong> Regular check-ups and medical advice are crucial for monitoring tumor growth and managing symptoms.</li>
                    </ul>
                </li>
                <li><strong>Manage stress levels</strong> through activities such as meditation, yoga, or therapy:
                    <ul>
                        <li>Stress management techniques can help improve overall well-being and may impact tumor growth.</li>
                    </ul>
                </li>
                <li><strong>Maintain a balanced diet</strong> with plenty of fruits, vegetables, and whole grains:
                    <ul>
                        <li>A balanced diet provides essential nutrients and supports overall health.</li>
                    </ul>
                </li>
                <li><strong>Follow your doctor's recommendations</strong> regarding regular check-ups and medical advice:
                    <ul>
                        <li>Regular check-ups are important for monitoring tumor growth and adjusting treatment as needed.</li>
                    </ul>
                </li>
            </ol>
        """,
        "glioma_tumor": """
            <ol>
                <li><strong>Reduce exposure to radiation and harmful chemicals</strong>, manage stress levels, maintain a balanced diet, and follow your doctor's recommendations:
                    <ul>
                        <li><strong>Reduce exposure to radiation and harmful chemicals:</strong> Minimize exposure to environmental toxins and radiation, which may contribute to tumor growth.</li>
                        <li><strong>Manage stress levels:</strong> Activities such as meditation, yoga, or therapy can help reduce stress, which may impact tumor growth.</li>
                        <li><strong>Maintain a balanced diet:</strong> Include plenty of fruits, vegetables, and whole grains in your diet to support overall health.</li>
                        <li><strong>Follow doctor's recommendations:</strong> Regular check-ups and medical advice are crucial for monitoring tumor growth and managing symptoms.</li>
                    </ul>
                </li>
                <li><strong>Manage stress levels</strong> through activities such as meditation, yoga, or therapy:
                    <ul>
                        <li>Stress management techniques can help improve overall well-being and may impact tumor growth.</li>
                    </ul>
                </li>
                <li><strong>Maintain a balanced diet</strong> with plenty of fruits, vegetables, and whole grains:
                    <ul>
                        <li>A balanced diet provides essential nutrients and supports overall health.</li>
                    </ul>
                </li>
                <li><strong>Follow your doctor's recommendations</strong> regarding regular check-ups and medical advice:
                    <ul>
                        <li>Regular check-ups are important for monitoring tumor growth and adjusting treatment as needed.</li>
                    </ul>
                </li>
            </ol>
        """,
    },
)

# -------------------
# Helpers
# -------------------
def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((350, 350))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image, cancer_type):
    if cancer_type.name == "Kidney Cancer":
        model = kidney_model
    else:
        model = brain_model

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    class_index = int(np.argmax(prediction))
    probability = float(prediction[class_index])
    predicted_class = cancer_type.labels[class_index]

    return predicted_class, probability, prediction

def color_for_prob(p):
    if p < 0.5:
        return ("#a8e6a3", "#4caf50")  # light green -> green
    elif p < 0.8:
        return ("#ffd8a8", "#ff9800")  # light orange -> orange
    else:
        return ("#ffb3b3", "#f44336")  # light red -> red

def animate_gradient_bar(container, label, target_prob, theme_text_color, duration=1.0):
    frames = max(12, int(duration * 30))
    start_color, end_color = color_for_prob(target_prob)
    for f in range(frames + 1):
        progress_frac = (f / frames) * target_prob
        width_pct = progress_frac * 100
        bar_html = f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-weight:600; color:{theme_text_color}; margin-bottom:6px;">
                <div>{label}</div>
                <div>{(progress_frac*100):.1f}%</div>
            </div>
            <div style="background: rgba(255,255,255,0.06); border-radius:10px; height:18px; overflow:hidden;">
                <div style="
                    width:{width_pct:.2f}%;
                    height:100%;
                    border-radius:10px;
                    background: linear-gradient(90deg,{start_color}, {end_color});
                    transition: width 0.15s linear;
                "></div>
            </div>
        </div>
        """
        container.markdown(bar_html, unsafe_allow_html=True)
        time.sleep(duration / frames)

# -------------------
# App UI
# -------------------
def main():
    st.set_page_config(page_title="NMIMR MEDICT", page_icon="🩺", layout="wide")

    # Sidebar controls
    st.sidebar.title("Controls")
    dark_mode = st.sidebar.checkbox("🌙 Dark mode", value=False)
    show_about = st.sidebar.checkbox("About developer", value=False)

    # Theme variables
    bg = "#0f1720" if dark_mode else "#f5f7fa"
    card_bg = "#121212" if dark_mode else "#ffffff"
    text_color = "#e6eef8" if dark_mode else "#111827"
    sub_text_color = "#9ca3af" if not dark_mode else "#bfc8d8"
    accent = "#2b6ef6"

    # inject base CSS
    st.markdown(
        f"""
        <style>
        .app-header {{
            text-align:center;
            margin-bottom: 8px;
        }}
        .page-title {{
            font-size:28px;
            font-weight:700;
            color: {accent};
            margin: 6px 0;
        }}
        .page-subtitle {{
            color: {sub_text_color};
            margin-bottom: 18px;
        }}
        .card {{
            background: {card_bg};
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(2,6,23,0.08);
        }}
        .muted {{
            color: {sub_text_color};
            font-size: 0.95rem;
        }}
        .small {{
            font-size: 0.9rem;
        }}
        .footer {{
            text-align:center;
            color:{sub_text_color};
            margin-top:18px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">NMIMR MEDICT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">AI-powered medical image diagnosis — Kidney, Brain</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # About developer
    if show_about:
        st.markdown(
            """
            <div class="card">
                <h4 style="margin-bottom:6px;">About the developer</h4>
                <p class="muted small">
                    This application was developed by Justice PANGenS. It uses TensorFlow deep learning models trained to assist with kidney and brain cancer detection.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Choose cancer type
    cancer_options = {
        "Kidney Cancer": kidney_cancer,
        "Brain Tumor": brain_cancer,
    }
    cancer_choice = st.selectbox("Select cancer type to analyze", options=list(cancer_options.keys()))

    cancer_type = cancer_options[cancer_choice]

    # Show description
    st.markdown(f"### About {cancer_type.name}")
    st.markdown(f"<p class='muted small'>{cancer_type.description}</p>", unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader(f"Upload an image for {cancer_type.name} diagnosis", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", use_column_width=True)

            # Predict button
            if st.button("Predict"):
                with st.spinner("Running model prediction..."):
                    predicted_class, probability, probs = predict(image, cancer_type)

                prob_percent = probability * 100
                color_bg, color_text = color_for_prob(probability)

                st.markdown(
                    f"""
                    <div class="card" style="background: {color_bg}; color: {color_text};">
                        <h3>Prediction Result:</h3>
                        <p><strong>{predicted_class}</strong> with confidence {prob_percent:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show detailed info based on prediction
                if predicted_class in cancer_type.true_positive_descriptions:
                    st.markdown(f"<b>Interpretation:</b> {cancer_type.true_positive_descriptions[predicted_class]}", unsafe_allow_html=True)
                    st.markdown(f"<b>Precautions & recommendations:</b>", unsafe_allow_html=True)
                    st.markdown(cancer_type.precautions.get(predicted_class, "No precautions available."), unsafe_allow_html=True)
                else:
                    st.markdown(f"<b>Interpretation:</b> {cancer_type.true_negative_description}", unsafe_allow_html=True)

                # Show prediction probabilities with animation
                with st.container() as bar_container:
                    st.markdown("<h4>Prediction probabilities:</h4>")
                    for idx, label in cancer_type.labels.items():
                        animate_gradient_bar(bar_container, label, probs[idx], text_color)

        except Exception as e:
            st.error(f"Error processing image or model prediction: {str(e)}")

if __name__ == "__main__":
    main()
