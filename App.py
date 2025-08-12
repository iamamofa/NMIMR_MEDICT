import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load the latest models
lung_model = tf.keras.models.load_model("./models/Lung_Cancer_2025_08_11.h5")
kidney_model = tf.keras.models.load_model("./models/Kidney_tumor_2025_08_11.h5")
brain_model = tf.keras.models.load_model("./models/Brain_Tumor_2025_08_11.h5")

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


lung_cancer = Cancer(
    name="Lung Cancer",
    description="Lung cancer is a malignant disease that originates in the lungs. It is categorized into two main types: non-small cell lung cancer (NSCLC) and small cell lung cancer (SCLC). NSCLC is the more common type and typically grows and spreads more slowly than SCLC. SCLC, although less common, tends to grow more aggressively and is more likely to spread to other organs in the body. Lung cancer is often associated with smoking but can also occur in non-smokers due to other factors such as exposure to secondhand smoke, air pollution, or genetic predisposition. Early detection and treatment are crucial for improving outcomes.",
    labels={
        0: "Adenocarcinoma",
        1: "Large Cell Carcinoma",
        2: "Normal",
        3: "Squamous Cell Carcinoma",
    },
    true_positive_descriptions={
        "Adenocarcinoma": "The image shows signs of Adenocarcinoma lung cancer. Please consult a doctor for further evaluation and treatment.",
        "Large Cell Carcinoma": "The image shows signs of Large Cell Carcinoma lung cancer. Please consult a doctor for further evaluation and treatment.",
        "Squamous Cell Carcinoma": "The image shows signs of Squamous Cell Carcinoma lung cancer. Please consult a doctor for further evaluation and treatment.",
    },
    true_negative_description="The image does not show any signs of lung cancer. However, regular check-ups are recommended.",
    precautions={
        "Adenocarcinoma": """
            <ol>
                <li><strong>Quit Smoking:</strong> Enroll in a smoking cessation program or use nicotine replacement therapies (patches, gums) and medications like varenicline or bupropion under medical supervision.</li>
                <li><strong>Avoid Secondhand Smoke:</strong> Stay away from areas where smoking is permitted and advocate for smoke-free environments in public spaces.</li>
                <li><strong>Healthy Diet:</strong> Include a diet rich in fruits, vegetables, whole grains, and lean proteins. Reduce red meat and processed foods.</li>
                <li><strong>Regular Exercise:</strong> Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity weekly, along with muscle-strengthening activities.</li>
                <li><strong>Routine Health Screenings:</strong> Schedule regular check-ups, including lung cancer screenings (low-dose CT scans) if you have a history of heavy smoking.</li>
                <li><strong>Environmental Factors:</strong> Minimize exposure to known carcinogens like radon, asbestos, and air pollution by using protective measures and improving ventilation at home and work.</li>
                <li><strong>Vaccinations:</strong> Stay updated with vaccinations, such as the flu shot, to reduce lung infections that can complicate respiratory health.</li>
                <li><strong>Follow Medical Advice:</strong> Adhere to prescribed treatments and medications, attend all follow-up appointments, and report any new symptoms to your doctor immediately.</li>
            </ol>
        """,
        "Large Cell Carcinoma": """
            <ol>
                <li><strong>Quit Smoking:</strong> Use behavioral therapy, support groups, and medications as recommended by your healthcare provider to help quit smoking.</li>
                <li><strong>Avoid Secondhand Smoke:</strong> Implement a no-smoking policy at home and choose smoke-free accommodations when traveling.</li>
                <li><strong>Balanced Nutrition:</strong> Emphasize a diet with antioxidants and anti-inflammatory properties, including omega-3 fatty acids found in fish and flaxseeds.</li>
                <li><strong>Physical Activity:</strong> Engage in regular physical activity tailored to your fitness level, such as walking, cycling, or swimming.</li>
                <li><strong>Occupational Safety:</strong> Use protective gear if you work in environments with chemical fumes, dust, or other hazardous substances. Follow workplace safety regulations.</li>
                <li><strong>Home Safety:</strong> Test your home for radon levels and install mitigation systems if necessary. Reduce exposure to household chemicals by using natural cleaning products.</li>
                <li><strong>Stress Management:</strong> Practice stress-relief techniques such as mindfulness, yoga, or meditation to improve overall well-being.</li>
                <li><strong>Follow Medical Advice:</strong> Maintain regular communication with your healthcare team, follow treatment plans, and promptly address any concerns or side effects.</li>
            </ol>
        """,
        "Squamous Cell Carcinoma": """
            <ol>
                <li><strong>Quit Smoking:</strong> Seek professional help through cessation programs, counseling, and FDA-approved medications to quit smoking effectively.</li>
                <li><strong>Avoid Secondhand Smoke:</strong> Create a smoke-free home environment and avoid social settings where smoking is prevalent.</li>
                <li><strong>Dietary Adjustments:</strong> Consume a diet high in vitamins and minerals, particularly vitamin A, C, and E, which are found in colorful fruits and vegetables.</li>
                <li><strong>Exercise Routine:</strong> Incorporate regular physical activity that includes both cardiovascular and strength-training exercises to enhance lung function and overall health.</li>
                <li><strong>Protective Measures:</strong> Use personal protective equipment (PPE) if you are exposed to dust, asbestos, or other harmful substances at work.</li>
                <li><strong>Regular Screenings:</strong> Participate in regular health check-ups and lung cancer screenings if you are at high risk. Early detection is crucial.</li>
                <li><strong>Avoid Carcinogens:</strong> Limit exposure to environmental carcinogens by using air purifiers, avoiding polluted areas, and ensuring proper ventilation in living and working spaces.</li>
                <li><strong>Follow Medical Advice:</strong> Keep up with all prescribed treatments, attend regular follow-up appointments, and stay informed about the latest treatment options and clinical trials.</li>
            </ol>
        """,
    },
)

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


def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((350, 350))
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(image, cancer_type):
    model = None
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

    if predicted_class != "Normal" and predicted_class != "no_tumor":
        
        st.markdown(
        f"""
        <div class="prediction-container">
            <p class="prediction">Predicted Class: {predicted_class}</p>
            <p class="prediction">Probability: {probability:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
        
        st.warning(cancer_type.true_positive_descriptions[predicted_class])
        
        st.info("Please consult a doctor for further evaluation and treatment.")
    else:
        st.markdown(
            f'<p class="prediction">Predicted Class: {predicted_class}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="prediction">Probability: {probability:.2f}</p>',
            unsafe_allow_html=True,
        )
        st.success(cancer_type.true_negative_description)

    return predicted_class, probability


def main():
    
    image_path = "./image.png"
    image = Image.open(image_path)
    
    st.set_page_config(
        page_title="NMIMR MEDICT",
        page_icon=image,
        layout="centered",
        initial_sidebar_state="expanded",
    )

    with open("styles.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    show_about_me = st.checkbox("About Me")
    if show_about_me:
        st.markdown(
            "<h2>About Me</h2>"
            "<p>I am Justice Ohene Amofa, I am interested in machine learning and web development. "
            "Connect with me on social media:</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="social-icons">
                <a href="https://www.linkedin.com/in/joamofa/">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn">
                </a>
                <a href="https://github.com/iamamofa">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Github-desktop-logo-symbol.svg/640px-Github-desktop-logo-symbol.svg.png" alt="GitHub">
                </a>
                <a href="#">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook">
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        '<div class="title"><u>NMIMR MEDICT</u> </br> Medical Diagnosis using Computer Vision</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


    cancer_types = [lung_cancer, kidney_cancer, brain_cancer]
    selected_cancer_type = None

    with st.sidebar:
        options = [cancer_type.name for cancer_type in cancer_types]
        selected_option = st.selectbox("Select Cancer Type", options)
        selected_cancer_type = next(
            cancer_type
            for cancer_type in cancer_types
            if cancer_type.name == selected_option
        )
        st.markdown(f"**Description:** {selected_cancer_type.description}")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image")

        with st.spinner("Predicting..."):
            predicted_class, probability = predict(image, selected_cancer_type)

        if predicted_class != "Normal" and predicted_class != "No Tumor":
            with st.expander("Precautions", expanded=False):
                st.markdown(f"<div class='Precaution'>{selected_cancer_type.precautions[predicted_class]}</div>", unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; margin-top: 2rem;">'
                '<p style="font-size: 0.9rem; color: #777;">'
                '© 2025 Justice Ohene Amofa. All rights reserved.</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
