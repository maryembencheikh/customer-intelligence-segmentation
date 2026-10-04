import streamlit as st
import joblib
import pandas as pd


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load trained pipeline
# --------------------------------------------------

pipeline = joblib.load("models/customer_pipeline.pkl")


# --------------------------------------------------
# Business interpretation
# --------------------------------------------------

def interpret_cluster(cluster):

    if cluster == 0:
        return {
            "Segment": "Lower Purchase Intensity",
            "Description": (
                "This customer belongs to the lower purchase-intensity "
                "segment, characterized by lower purchase frequency."
            )
        }

    else:
        return {
            "Segment": "Higher Purchase Intensity",
            "Description": (
                "This customer belongs to the higher purchase-intensity "
                "segment, characterized by higher purchase frequency."
            )
        }


# --------------------------------------------------
# Business recommendation
# --------------------------------------------------

def get_business_recommendation(cluster):

    if cluster == 0:
        return {
            "Objective": (
                "Increase engagement and purchase frequency"
            ),
            "Recommendation": (
                "Test targeted re-engagement campaigns "
                "to encourage more frequent purchases."
            )
        }

    else:
        return {
            "Objective": (
                "Maintain engagement and strengthen customer value"
            ),
            "Recommendation": (
                "Test retention and cross-sell initiatives "
                "to maintain purchasing activity."
            )
        }


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 Customer Intelligence")

st.markdown(
    """
    ### Behavioral Customer Segmentation

    This application assigns a customer to one of two behavioral segments
    based on purchasing activity and provides a business-oriented
    interpretation of the result.
    """
)

st.divider()


# --------------------------------------------------
# Customer information
# --------------------------------------------------

st.header("👤 Customer Information")

st.caption(
    "Enter the customer's observed purchasing characteristics."
)


col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    purchase_amount = st.number_input(
        "Purchase Amount (USD)",
        min_value=0.0,
        value=72.0,
        step=1.0
    )

    review_rating = st.number_input(
        "Review Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )


with col2:

    previous_purchases = st.number_input(
        "Previous Purchases",
        min_value=0,
        value=30,
        step=1
    )

    frequency = st.selectbox(
        "Frequency of Purchases",
        [
            "Weekly",
            "Bi-Weekly",
            "Fortnightly",
            "Monthly",
            "Quarterly",
            "Every 3 Months",
            "Annually"
        ]
    )

    subscription = st.selectbox(
        "Subscription Status",
        ["Yes", "No"]
    )


st.write("")


# --------------------------------------------------
# Prediction button
# --------------------------------------------------

predict_button = st.button(
    "🔍 Predict Customer Segment",
    use_container_width=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    customer = pd.DataFrame([{
        "Age": age,
        "Purchase Amount (USD)": purchase_amount,
        "Review Rating": review_rating,
        "Previous Purchases": previous_purchases,
        "Frequency of Purchases": frequency,
        "Subscription Status": subscription
    }])

    cluster = pipeline.predict(customer)[0]

    result = interpret_cluster(cluster)
    recommendation = get_business_recommendation(cluster)


    st.divider()


    # --------------------------------------------------
    # Segment result
    # --------------------------------------------------

    st.header("🎯 Predicted Segment")

    st.success(result["Segment"])

    st.write(result["Description"])


    # --------------------------------------------------
    # Customer profile
    # --------------------------------------------------

    st.subheader("Customer Profile")

    profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)

    with profile_col1:
        st.metric(
            "Purchase Frequency",
            frequency
        )

    with profile_col2:
        st.metric(
            "Previous Purchases",
            previous_purchases
        )

    with profile_col3:
        st.metric(
            "Purchase Amount",
            f"${purchase_amount:.2f}"
        )

    with profile_col4:
        st.metric(
            "Subscription",
            subscription
        )


    # --------------------------------------------------
    # Business recommendation
    # --------------------------------------------------

    st.subheader("💼 Business Recommendation")

    recommendation_col1, recommendation_col2 = st.columns(
        [1, 2]
    )

    with recommendation_col1:

        st.markdown("**Business Objective**")

        st.info(
            recommendation["Objective"]
        )

    with recommendation_col2:

        st.markdown("**Recommended Action**")

        st.info(
            recommendation["Recommendation"]
        )


    # --------------------------------------------------
    # Methodology note
    # --------------------------------------------------

    st.divider()

    st.caption(
        "Methodology: customer features are transformed and standardized "
        "through the trained pipeline before KMeans segment assignment."
    )

    st.caption(
        "The segmentation is descriptive and supports customer "
        "prioritization. It does not predict future customer behavior "
        "or establish causal effects."
    )