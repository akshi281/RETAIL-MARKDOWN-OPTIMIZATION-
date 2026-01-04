import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load saved model & objects
# -----------------------------
RF_model = pickle.load(open("RF_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Retail Markdown Optimization",
    page_icon="🛍️",
    layout="centered"
)
# -----------------------------
# Title
# -----------------------------
st.title("🛍️ Retail Markdown Optimization")
st.markdown("Predict **Optimal Discount (%)** using Machine Learning")
st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.header("📊 Enter Product Details")

original_price = st.number_input("Original Price", min_value=0.0, value=499.0)
competitor_price = st.number_input("Competitor Price", min_value=0.0, value=450.0)
stock_level = st.number_input("Stock Level", min_value=0, value=300)
historical_sales = st.number_input("Historical Sales", min_value=0, value=120)
seasonality_factor = st.slider("Seasonality Factor", 0.5, 2.0, 1.2)

st.markdown("---")
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(
            rgba(0, 0, 0, 0.65),
            rgba(0, 0, 0, 0.65)
        ),
        url("data:image/jpg;base64");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* 🔹 Headings & labels */
    h1, h2, h3, h4, h5, h6 {{
        color: white !important;
    }}

    label {{
        background-color: black;
        color: white !important;
        padding: 6px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 5px;
    }}

    /* 🔹 Input boxes (values area) */
    input, textarea {{
        background-color: white !important;
        color: black !important;
        border-radius: 10px;
        padding: 8px;
        font-size: 16px;
    }}

    /* 🔹 Slider background */
    .stSlider {{
        background-color: white;
        padding: 10px;
        border-radius: 10px;
    }}

    /* 🔹 Button styling */
    .stButton>button {{
        background-color: #ff4b4b;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        border: none;
    }}

    .stButton>button:hover {{
        background-color: #ff1f1f;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Background Image & Styling
# -----------------------------
def add_bg_from_local(image_file):
    import base64
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(
                rgba(0, 0, 0, 0.65),
                rgba(0, 0, 0, 0.65)
            ),
            url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        h1, h2, h3, h4, h5, h6, p, label {{
            color: white !important;
        }}

        .stNumberInput, .stSlider {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 10px;
            padding: 10px;
        }}

        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }}

        .stButton>button:hover {{
            background-color: #ff1f1f;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call background function
add_bg_from_local("Special Discount Red Text Box Vector Design Promotion Marketing Free Download, Discount Vector, Special Discount, Discount Products PNG and Vector with Transparent Background for Free Download.jfif" \
"")


# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = {col: 0 for col in feature_columns}

input_data.update({
    'Original_Price': original_price,
    'Competitor_Price': competitor_price,
    'Stock_Level': stock_level,
    'Historical_Sales': historical_sales,
    'Seasonality_Factor': seasonality_factor
})

input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict Optimal Discount"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = RF_model.predict(input_scaled)
        discount = prediction[0]

        # 🎉 Visual success effect
        st.balloons()

        # ✅ Main Result
        st.success(f"🎯 **Predicted Optimal Discount: {discount:.2f}%**")

        # 📊 Discount Interpretation
        if discount < 15:
            st.info("📉 **Low Discount Strategy**: Suitable for high-demand or premium products.")
        elif 15 <= discount <= 30:
            st.warning("📊 **Moderate Discount Strategy**: Balanced for sales and profit.")
        else:
            st.error("📈 **High Discount Strategy**: Useful for clearing excess inventory.")

        # 🧠 Business Insight
        st.markdown("### 🧠 Business Insight")
        st.markdown(
            f"""
            - Recommended discount is **{discount:.2f}%**
            - Helps optimize **sales volume and revenue**
            - Based on pricing, stock level, demand, and seasonality
            """
        )

    except Exception as e:
        st.error("❌ Prediction failed")
        st.write(e)