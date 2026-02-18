import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="MNIST Sifferigenkänning",
    page_icon="🔢",
    layout="centered"
)

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


# =============================
# Prediktion
# =============================

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

        # 1️⃣ Hämta RGBA-bilden
        img = Image.fromarray(
            canvas_result.image_data.astype("uint8"),
            mode="RGBA"
        )

        # 2️⃣ Konvertera till gråskala
        img = img.convert("L")

        # 3️⃣ Resize till exakt 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 4️⃣ Konvertera till numpy (0–255, samma som träningen)
        img_array = np.array(img)

        # 🔎 DEBUG (kan tas bort sen)
        # st.write("Min pixel:", img_array.min(), "Max pixel:", img_array.max())

        # 5️⃣ Platta ut
        img_flat = img_array.reshape(1, -1)

        # 6️⃣ Prediktera (pipelinen skalar automatiskt)
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]

        # Skapa tabell för sannolikheter (0–9)
        proba_df = pd.DataFrame({
            "Siffra": list(range(10)),
            "Sannolikhet (%)": probabilities * 100
        }).sort_values("Sannolikhet (%)", ascending=False)


        # =============================
        # Visa resultat
        # =============================

        st.divider()

        st.markdown("## 🎯 Min gissning")

        st.markdown(f"# 🎉 Jag tror att det är en **{prediction}!**")
        st.markdown("Gissade jag rätt? 😉")

        st.markdown("### 📊 Sannolikheter")

        chart_df = proba_df.sort_values("Siffra")

        # Visualisera den gissade siffran tydligast
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
                st.success("WOOHO! Jag är grym på siffror! 🎉🤖")

        with col_no:
            if st.button("❌ Nej, det blev fel"):
                st.info("Åh nej… 😭 Jag ska träna mer! 💪")

    else:
        st.warning("Rita en siffra först.")
