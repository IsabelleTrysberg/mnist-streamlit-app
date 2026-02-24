import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin


st.set_page_config(
    page_title="MNIST Sifferigenkänning",
    page_icon="🔢",
    layout="centered"
)

# =============================
# Skapar en transformer för att minska antal pixlar
# =============================

class ResizeImages(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_images = X.reshape(-1, 28, 28)
        X_cropped = X_images[:, 4:24, 4:24]
        return X_cropped.reshape(-1, 20 * 20)

# =============================
# Ladda modell (hela pipelinen)
# =============================

@st.cache_resource
def load_model():
    return joblib.load("mnist_svc_model.pkl")

model = load_model()

# =============================
# UI
# =============================

st.title("🔢 Lyckas vi tolka dina siffror?")

st.markdown(
    "Rita en siffra mellan **0 och 9** i rutan nedan och klicka på **Skicka**."
)

st.divider()

if "stage" not in st.session_state:
    st.session_state.stage = "draw"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_probabilities" not in st.session_state:
    st.session_state.last_probabilities = None

# =============================
# Prediktion (DRAW)
# =============================
if st.session_state.stage == "draw":

    st.subheader("✍️ Rita din siffra här")

    with st.container():
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=st.session_state.get("canvas_key", "canvas"),
        )

    st.write("")  # lite luft

    predict_button = st.button("📨 Skicka", type="primary")
    clear_button = st.button("🧹 Rensa rutan")

    if clear_button:
        st.session_state.canvas_key = "canvas_cleared"
        st.rerun()

    if predict_button:
        if canvas_result.image_data is not None:

            # Hämta RGBA-bilden
            img = Image.fromarray(
                canvas_result.image_data.astype("uint8"),
                mode="RGBA"
            )

            # Konvertera till gråskala
            img = img.convert("L")

            # Konvertera till numpy 
            img_array = np.array(img)

            # Bounding box runt siffran
            coords = np.column_stack(np.where(img_array > 0))
            if coords.size == 0:
                st.warning("Rita en siffra först.")
                st.stop()

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            digit = img_array[y_min:y_max+1, x_min:x_max+1]

            # Skala så att största sidan blir 20 pixlar
            h, w = digit.shape
            scale = 20.0 / max(h, w)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            digit_resized = np.array(
                Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS)
            )
            # Gör siffran lite smalare
            squeeze = 0.85  # testat olika storlekar här
            new_w2 = max(1, int(round(digit_resized.shape[1] * squeeze)))
            digit_resized = np.array(
                Image.fromarray(digit_resized).resize((new_w2, digit_resized.shape[0]), Image.Resampling.LANCZOS)
            )
            new_w = new_w2  # så att placeringen på canvas använder nya bredden

            # Lägg på en 28x28 canvas (centrerad)
            canvas = np.zeros((28, 28), dtype=np.uint8)
            top = (28 - new_h) // 2
            left = (28 - new_w) // 2
            canvas[top:top+new_h, left:left+new_w] = digit_resized

            img_array = canvas

            # Debug kod    
            st.session_state.last_image = img_array

            # 5️⃣ Platta ut
            img_flat = img_array.reshape(1, -1)

            # 6️⃣ Prediktera (pipelinen skalar automatiskt)
            prediction = model.predict(img_flat)[0]
            probabilities = model.predict_proba(img_flat)[0]

            st.session_state.last_prediction = prediction
            st.session_state.last_probabilities = probabilities
            st.session_state.stage = "predicted"
            st.rerun()

        else:
            st.warning("Rita en siffra först.")

# =============================
# Resultat (PREDICTED)
# =============================
if st.session_state.stage == "predicted":
    prediction = st.session_state.last_prediction
    probabilities = st.session_state.last_probabilities

    proba_df = pd.DataFrame({
        "Siffra": list(range(10)),
        "Sannolikhet (%)": probabilities * 100
    })

    st.write(model.classes_)

    st.divider()
    st.markdown("## 🎯 Min gissning")
    st.markdown(f"# 🎉 Jag tror att det är en **{prediction}!**")
    st.markdown("Gissade jag rätt? 😉")

    if st.session_state.get("last_image") is not None:
        st.image(st.session_state.last_image, caption="Vad modellen ser (28x28)", clamp=True, width=150)

    st.markdown("### 📊 Sannolikheter")

    chart_df = proba_df.sort_values("Siffra")

    colors = [
        "crimson" if s == prediction else "steelblue"
        for s in chart_df["Siffra"]
    ]

    fig = px.bar(
        chart_df,
        x="Siffra",
        y="Sannolikhet (%)",
        text="Sannolikhet (%)",
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(10)),
        ticktext=[str(i) for i in range(10)]
    )

    fig.update_traces(
        marker_color=colors,
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Sannolikhet (%)",
        xaxis_title="Siffra",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    col_yes, col_no = st.columns(2)

    with col_yes:
        if st.button("✅ Ja, du gissade rätt!"):
            st.session_state.stage = "done_yes"
            st.rerun()

    with col_no:
        if st.button("❌ Nej, det blev fel"):
            st.session_state.stage = "done_no"
            st.rerun()

# =============================
# Slutmeddelande (DONE)
# =============================
if st.session_state.stage == "done_yes":
    st.success("WOOHO! Jag är grym på siffror! 🎉🤖")
    if st.button("🔄 Tillbaka till start"):
        st.session_state.stage = "draw"
        st.session_state.canvas_key = "canvas_cleared"
        st.rerun()
if st.session_state.stage == "done_no":
    st.info("Åh nej… 😭 Jag ska träna mer! 💪")
    if st.button("🔄 Tillbaka till start"):
        st.session_state.stage = "draw"
        st.session_state.canvas_key = "canvas_cleared"
        st.rerun()