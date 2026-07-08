import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import plotly.express as px
from streamlit_image_comparison import image_comparison
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="VisionAI - OCT Disease Detection",
    page_icon="👁️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp{
background:#f8fafc;
}
.title{
background:linear-gradient(90deg,#0f172a,#1e3a8a);
padding:25px;
border-radius:12px;
color:white;
text-align:center;
margin-bottom:20px;
}
.result{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 3px 10px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
"""
<div class="title">
<h1>👁️ VisionAI OCT Disease Detection</h1>
<p>AI Powered Retinal Disease Detection using OCT Images</p>
</div>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------
MODEL_PATH = r"retina_model.h5"
HEALTHY_IMAGE = r"healthy.jpeg"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found:\n{MODEL_PATH}")
        st.stop()
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return model
    except Exception as e:
        st.error(e)
        st.stop()

model = load_model()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------
CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "GLAUCOMA", "MH"]

# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------
def preprocess(image):
    image = ImageOps.fit(image, (224,224), Image.LANCZOS)
    image = image.convert("RGB")
    image = np.array(image).astype(np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
def predict(image):
    img = preprocess(image)
    prediction = model.predict(img, verbose=0)[0]
    return prediction

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded = st.file_uploader("Upload OCT Image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    prediction = predict(image)
    disease = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    col1, col2 = st.columns([1.6, 1])

    # --------------------------------------------------
    # IMAGE COMPARISON (COLUMN 1)
    # --------------------------------------------------
    with col1:
        st.subheader("Patient Retina vs Healthy Retina")
        if os.path.exists(HEALTHY_IMAGE):
            healthy = Image.open(HEALTHY_IMAGE).convert("RGB")
            image_comparison(
                img1=image.resize((900,450)),
                img2=healthy.resize((900,450)),
                label1="Patient",
                label2="Healthy",
                width=900,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True
            )
        else:
            st.warning("Healthy reference image not found.")
            st.image(image, use_container_width=True)

    # --------------------------------------------------
    # RESULT PANEL (COLUMN 2 - Correctly aligned)
    # --------------------------------------------------
    with col2:
        st.markdown(f"""
        <div class="result">
        <h2>Diagnosis</h2>
        <h1 style="color:#2563eb;">{disease}</h1>
        <h3>Confidence</h3>
        <h2>{confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        if disease == "AMD":
            st.info("Age-related Macular Degeneration (AMD)\n\n• Damage to the macula\n• Causes blurred central vision\n• Common in older adults")
        elif disease == "CNV":
            st.info("Choroidal Neovascularization (CNV)\n\n• Abnormal blood vessels grow under retina\n• Can cause severe vision loss")
        elif disease == "CSR":
            st.info("Central Serous Retinopathy (CSR)\n\n• Fluid builds beneath retina\n• Usually causes blurred or distorted vision")
        elif disease == "DME":
            st.info("Diabetic Macular Edema (DME)\n\n• Swelling of the macula\n• Caused by diabetes")
        elif disease == "DR":
            st.info("Diabetic Retinopathy (DR)\n\n• Damage to retinal blood vessels\n• Common complication of diabetes")
        elif disease == "DRUSEN":
            st.info("Drusen Detected\n\n• Yellow deposits beneath retina\n• May indicate early AMD")
        elif disease == "GLAUCOMA":
            st.info("Glaucoma Detected\n\n• Optic nerve damage\n• Often associated with high eye pressure")
        elif disease == "MH":
            st.info("Macular Hole\n\n• Small break in the macula\n• Can reduce central vision")

    st.divider()

    # --------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------
    fig = px.bar(
        x=CLASS_NAMES,
        y=prediction * 100,
        color=CLASS_NAMES,
        text=np.round(prediction * 100, 2),
        title="Prediction Probability"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Disease",
        yaxis_title="Probability (%)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # PROBABILITY TABLE
    # --------------------------------------------------
    st.subheader("Prediction Scores")
    scores = {
        "Disease": CLASS_NAMES,
        "Probability (%)": np.round(prediction * 100, 2)
    }
    st.dataframe(scores, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # DIAGNOSIS SUMMARY
    # --------------------------------------------------
    st.divider()
    st.subheader("Diagnosis Summary")
    st.success(
        f"""
Predicted Disease : {disease}
Confidence : {confidence:.2f}%

This prediction is generated by the trained deep learning model.
For medical diagnosis, always consult a qualified ophthalmologist.
"""
    )

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------
    report = f"""
==============================
VisionAI OCT Disease Detection
==============================

Predicted Disease : {disease}
Confidence : {confidence:.2f}%

Prediction Probabilities
"""
    for cls, prob in zip(CLASS_NAMES, prediction * 100):
        report += f"{cls:<12}: {prob:.2f}%\n"

    report += """
----------------------------------------
Disclaimer:
This software is intended for educational
and research purposes only.
It is NOT a substitute for professional
medical diagnosis.
"""
    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="VisionAI_Report.txt",
        mime="text/plain"
    )

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("VisionAI")
st.sidebar.markdown("""
### Supported Diseases
- AMD
- CNV
- CSR
- DME
- DR
- DRUSEN
- GLAUCOMA
- MH
""")
st.sidebar.info("Upload an OCT retinal image.\n\nThe AI model predicts the most likely retinal disease and displays the confidence score.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("👁️ VisionAI | OCT Retinal Disease Detection | Research Purpose Only")