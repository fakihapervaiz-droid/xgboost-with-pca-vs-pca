import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Bank Marketing Prediction System")
st.write(
    "Compare XGBoost predictions with PCA and without PCA."
)

st.markdown("---")


# ============================================================
# LOAD PICKLE FILES
# ============================================================

@st.cache_resource
def load_models():

    with open("xgboost_bank_marketing_pca.pkl", "rb") as file:
        pca_data = pickle.load(file)

    with open("xgboost_bank_marketing_no_pca.pkl", "rb") as file:
        no_pca_data = pickle.load(file)

    return pca_data, no_pca_data


try:

    pca_data, no_pca_data = load_models()

    pca_model = pca_data["model"]
    pca_encoder = pca_data["encoder"]
    pca_scaler = pca_data["scaler"]
    pca = pca_data["pca"]

    no_pca_model = no_pca_data["model"]
    no_pca_encoder = no_pca_data["encoder"]
    no_pca_scaler = no_pca_data["scaler"]

    categorical_features = no_pca_data["categorical_features"]
    numerical_features = no_pca_data["numerical_features"]
    features = no_pca_data["features"]

except Exception as e:

    st.error("Unable to load the model files.")
    st.error(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Customer Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)

marital = st.sidebar.selectbox(
    "Marital Status",
    [
        "married",
        "single",
        "divorced"
    ]
)

education = st.sidebar.selectbox(
    "Education",
    [
        "primary",
        "secondary",
        "tertiary",
        "unknown"
    ]
)

default = st.sidebar.selectbox(
    "Credit Default",
    [
        "no",
        "yes"
    ]
)

balance = st.sidebar.number_input(
    "Account Balance",
    value=1000
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    [
        "no",
        "yes"
    ]
)

loan = st.sidebar.selectbox(
    "Personal Loan",
    [
        "no",
        "yes"
    ]
)

contact = st.sidebar.selectbox(
    "Contact",
    [
        "cellular",
        "telephone",
        "unknown"
    ]
)

day = st.sidebar.number_input(
    "Last Contact Day",
    min_value=1,
    max_value=31,
    value=15
)

month = st.sidebar.selectbox(
    "Last Contact Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec"
    ]
)

duration = st.sidebar.number_input(
    "Call Duration (seconds)",
    min_value=0,
    value=300
)

campaign = st.sidebar.number_input(
    "Number of Contacts During Campaign",
    min_value=1,
    value=1
)

pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    value=-1
)

previous = st.sidebar.number_input(
    "Number of Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome",
    [
        "unknown",
        "failure",
        "other",
        "success"
    ]
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "day": [day],
    "month": [month],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome]
})

# Make sure feature order is exactly correct
input_data = input_data[features]


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Customer Response",
    type="primary",
    use_container_width=True
):

    try:

        # ====================================================
        # XGBoost + PCA
        # ====================================================

        pca_cat = pca_encoder.transform(
            input_data[categorical_features]
        )

        pca_num = input_data[numerical_features].values

        pca_encoded = np.hstack([
            pca_num,
            pca_cat
        ])

        pca_scaled = pca_scaler.transform(
            pca_encoded
        )

        pca_input = pca.transform(
            pca_scaled
        )

        pca_prediction = pca_model.predict(
            pca_input
        )[0]

        pca_probability = pca_model.predict_proba(
            pca_input
        )[0][1]


        # ====================================================
        # XGBoost WITHOUT PCA
        # ====================================================

        no_pca_cat = no_pca_encoder.transform(
            input_data[categorical_features]
        )

        no_pca_num = input_data[numerical_features].values

        no_pca_encoded = np.hstack([
            no_pca_num,
            no_pca_cat
        ])

        no_pca_scaled = no_pca_scaler.transform(
            no_pca_encoded
        )

        no_pca_prediction = no_pca_model.predict(
            no_pca_scaled
        )[0]

        no_pca_probability = no_pca_model.predict_proba(
            no_pca_scaled
        )[0][1]


        # ====================================================
        # CONVERT PREDICTIONS
        # ====================================================

        pca_result = (
            "YES - Customer likely to subscribe"
            if pca_prediction == 1
            else
            "NO - Customer unlikely to subscribe"
        )

        no_pca_result = (
            "YES - Customer likely to subscribe"
            if no_pca_prediction == 1
            else
            "NO - Customer unlikely to subscribe"
        )


        # ====================================================
        # PROBABILITY DIFFERENCE
        # ====================================================

        probability_difference = abs(
            no_pca_probability - pca_probability
        )

        probability_difference_percent = (
            probability_difference * 100
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.markdown("---")

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)


        # ====================================================
        # PCA RESULT
        # ====================================================

        with col1:

            st.markdown("### XGBoost + PCA")

            if pca_prediction == 1:
                st.success(pca_result)
            else:
                st.error(pca_result)

            st.metric(
                "Subscription Probability",
                f"{pca_probability * 100:.2f}%"
            )

            st.write(
                f"PCA Components Used: **{pca.n_components_}**"
            )


        # ====================================================
        # NO PCA RESULT
        # ====================================================

        with col2:

            st.markdown("### XGBoost Without PCA")

            if no_pca_prediction == 1:
                st.success(no_pca_result)
            else:
                st.error(no_pca_result)

            st.metric(
                "Subscription Probability",
                f"{no_pca_probability * 100:.2f}%"
            )

            st.write(
                f"Original Encoded Features: "
                f"**{no_pca_scaled.shape[1]}**"
            )


        # ====================================================
        # COMPARISON
        # ====================================================

        st.markdown("---")

        st.subheader("Model Comparison")


        comparison_df = pd.DataFrame({
            "Model": [
                "XGBoost + PCA",
                "XGBoost Without PCA"
            ],
            "Prediction": [
                "YES" if pca_prediction == 1 else "NO",
                "YES" if no_pca_prediction == 1 else "NO"
            ],
            "Probability": [
                f"{pca_probability * 100:.2f}%",
                f"{no_pca_probability * 100:.2f}%"
            ]
        })

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DIFFERENCE
        # ====================================================

        st.subheader("Difference Between Models")

        diff_col1, diff_col2 = st.columns(2)

        with diff_col1:

            st.metric(
                "Probability Difference",
                f"{probability_difference_percent:.2f}%"
            )

        with diff_col2:

            if pca_prediction == no_pca_prediction:

                st.success(
                    "Both models give the same prediction."
                )

            else:

                st.warning(
                    "The models give different predictions."
                )


        # ====================================================
        # WHICH MODEL IS MORE CONFIDENT?
        # ====================================================

        st.subheader("Model Confidence")

        if pca_probability > no_pca_probability:

            st.info(
                f"XGBoost + PCA gives the higher "
                f"subscription probability "
                f"({pca_probability * 100:.2f}%)."
            )

        elif no_pca_probability > pca_probability:

            st.info(
                f"XGBoost without PCA gives the higher "
                f"subscription probability "
                f"({no_pca_probability * 100:.2f}%)."
            )

        else:

            st.info(
                "Both models give the same probability."
            )


        # ====================================================
        # INPUT DATA
        # ====================================================

        st.markdown("---")

        st.subheader("Customer Input")

        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)