import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Attack Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg, #0f1117 0%, #14161f 100%);
}

.stMarkdown,
.stText,
p,
label,
span {
    color: #f1f5f9 !important;
}

div[data-testid="stWidgetLabel"] p {
    color: #f8fafc !important;
    font-weight: 600;
}

input {
    color: #ffffff !important;
    background-color: #262a3a !important;
}

div[data-testid="stSlider"] span {
    color: #ffffff !important;
}

div[data-testid="stTooltipIcon"] {
    color: #ffffff !important;
}

h3 {
    color: #ffffff !important;
    font-weight: 700;
}

.stCaption,
div[data-testid="stCaptionContainer"] {
    color: #cbd5e1 !important;
    font-size: 0.95rem !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: #e2e8f0 !important;
}

div[data-testid="stAlert"] p {
    color: #ffffff !important;
}


.header-card {
    background: linear-gradient(120deg, #b91c1c 0%, #7f1d1d 100%);
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(185,28,28,0.25);
}

.header-card h1 {
    color: white;
    margin: 0;
    font-size: 2.1rem;
    font-weight: 800;
}

.header-card p {
    color: #fde8e8;
}


.section-card {
    background: #1a1d29;
    padding: 22px 26px;
    border-radius: 16px;
    border: 1px solid #2a2e3f;
    margin-bottom: 18px;
}

.section-card h3 {
    margin-top: 0;
    border-bottom: 1px solid #2a2e3f;
    padding-bottom: 10px;
}


.result-card {
    padding: 26px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 16px;
}

.result-high {
    background: linear-gradient(135deg, #7f1d1d, #b91c1c);
}

.result-mod {
    background: linear-gradient(135deg, #78350f, #b45309);
}

.result-low {
    background: linear-gradient(135deg, #14532d, #15803d);
}

.result-card h2 {
    color: white;
}

.result-card .big-num {
    color: white;
    font-size: 3rem;
    font-weight: 800;
}

.chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.chip {
    background: #262a3a;
    color: #cbd5e1;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid #333853;
}

section[data-testid="stSidebar"] {
    background: #12141c;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)



# ------------------------------------------------------------------
# Load model + feature names
# ------------------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_attack_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, feature_names


try:
    model, FEATURE_NAMES = load_artifacts()
    LOAD_OK = True

except Exception as e:
    LOAD_OK = False
    LOAD_ERR = str(e)



# ------------------------------------------------------------------
# Feature configuration
# ------------------------------------------------------------------

FIELD_CONFIG = {
    "Age": {
        "label": "Age",
        "unit": "years",
        "min": 1,
        "max": 120,
        "default": 45,
        "step": 1,
        "help": "Patient age in years."
    },

    "BMI": {
        "label": "Body Mass Index (BMI)",
        "unit": "kg/m²",
        "min": 10.0,
        "max": 60.0,
        "default": 25.0,
        "step": 0.1,
        "help": "Weight divided by height squared."
    },

    "Systolic_BP": {
        "label": "Systolic Blood Pressure",
        "unit": "mmHg",
        "min": 70,
        "max": 250,
        "default": 120,
        "step": 1,
        "help": "Upper blood pressure value."
    },

    "Diastolic_BP": {
        "label": "Diastolic Blood Pressure",
        "unit": "mmHg",
        "min": 40,
        "max": 150,
        "default": 80,
        "step": 1,
        "help": "Lower blood pressure value."
    },

    "Total_Colesterol": {
        "label": "Total Cholesterol",
        "unit": "mg/dL",
        "min": 80,
        "max": 400,
        "default": 180,
        "step": 1,
        "help": "Total blood cholesterol level."
    },

    "C_Reactive": {
        "label": "C-Reactive Protein (CRP)",
        "unit": "mg/L",
        "min": 0.0,
        "max": 50.0,
        "default": 1.0,
        "step": 0.1,
        "help": "Inflammation marker."
    },

    "Waist_circ": {
        "label": "Waist Circumference",
        "unit": "cm",
        "min": 40,
        "max": 180,
        "default": 90,
        "step": 1,
        "help": "Measured at navel level."
    }
}

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

def risk_bucket(prob):
    if prob >= 0.66:
        return "High", "result-high", "🔴"
    elif prob >= 0.33:
        return "Moderate", "result-mod", "🟠"
    else:
        return "Low", "result-low", "🟢"



def make_gauge(prob):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={
                "suffix": "%",
                "font": {
                    "size": 46,
                    "color": "white"
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "white"
                },

                "bar": {
                    "color": "#f1f5f9",
                    "thickness": 0.25
                },

                "bgcolor": "rgba(0,0,0,0)",

                "steps": [
                    {
                        "range": [0, 33],
                        "color": "#15803d"
                    },
                    {
                        "range": [33, 66],
                        "color": "#b45309"
                    },
                    {
                        "range": [66, 100],
                        "color": "#b91c1c"
                    }
                ],

                "threshold": {
                    "line": {
                        "color": "white",
                        "width": 4
                    },
                    "value": prob * 100
                }
            }
        )
    )


    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(
            l=20,
            r=20,
            t=30,
            b=10
        )
    )

    return fig



def make_contribution_chart(model, input_dict):

    coefs = model.coef_[0]

    contributions = []

    for name, coef in zip(FEATURE_NAMES, coefs):
        contributions.append(
            coef * input_dict[name]
        )


    df = pd.DataFrame(
        {
            "Feature": [
                FIELD_CONFIG[n]["label"]
                for n in FEATURE_NAMES
            ],

            "Contribution": contributions
        }
    )


    df = df.sort_values(
        "Contribution"
    )


    colors = [
        "#b91c1c" if value > 0 else "#15803d"
        for value in df["Contribution"]
    ]


    fig = go.Figure(
        go.Bar(
            x=df["Contribution"],
            y=df["Feature"],
            orientation="h",
            marker_color=colors
        )
    )


    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )


    return fig



# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:

    st.markdown("## ❤️ About this tool")

    st.markdown(
        """
        This app estimates **heart attack risk**
        using a **Logistic Regression** model trained
        on the NHANES cardiovascular dataset.

        **Model inputs:**

        - Age
        - BMI
        - Blood Pressure
        - Total Cholesterol
        - C-Reactive Protein
        - Waist Circumference
        """
    )


    st.markdown("---")


    st.markdown("### ⚠️ Disclaimer")


    st.info(
        "This tool is an academic / educational project. "
        "It is not a medical device and should not be used "
        "for diagnosis or treatment decisions."
    )


    st.markdown("---")

    st.caption(
        "B.Sc Computer Science Project · NHANES Dataset · Streamlit"
    )



# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.markdown(
    """
    <div class="header-card">

    <h1>
    ❤️ Heart Attack Risk Predictor
    </h1>

    <p>
    Enter clinical measurements to estimate cardiovascular risk
    using machine learning.
    </p>

    </div>
    """,

    unsafe_allow_html=True
)



# ------------------------------------------------------------------
# Check Model Loading
# ------------------------------------------------------------------

if not LOAD_OK:

    st.error(
        f"""
        Could not load model files.

        Make sure:
        - heart_attack_model.pkl
        - feature_names.pkl

        are inside the same folder.

        Error:
        {LOAD_ERR}
        """
    )

    st.stop()



# ------------------------------------------------------------------
# Input Form
# ------------------------------------------------------------------

left, right = st.columns(
    [1, 1],
    gap="large"
)


input_values = {}



with left:


    st.markdown(
        '<div class="section-card"><h3>🧍 Patient Profile</h3>',
        unsafe_allow_html=True
    )


    for feat in [
        "Age",
        "BMI",
        "Waist_circ"
    ]:

        cfg = FIELD_CONFIG[feat]


        input_values[feat] = st.number_input(

            f"{cfg['label']} ({cfg['unit']})",

            min_value=float(cfg["min"]),
            max_value=float(cfg["max"]),

            value=float(cfg["default"]),

            step=float(cfg["step"]),

            help=cfg["help"],

            key=feat
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )



    st.markdown(
        '<div class="section-card"><h3>🩸 Blood Pressure</h3>',
        unsafe_allow_html=True
    )


    for feat in [
        "Systolic_BP",
        "Diastolic_BP"
    ]:

        cfg = FIELD_CONFIG[feat]


        input_values[feat] = st.slider(

            f"{cfg['label']} ({cfg['unit']})",

            min_value=int(cfg["min"]),

            max_value=int(cfg["max"]),

            value=int(cfg["default"]),

            step=int(cfg["step"]),

            help=cfg["help"],

            key=feat
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ------------------------------------------------------------------
# Right Side Inputs
# ------------------------------------------------------------------

with right:

    st.markdown(
        '<div class="section-card"><h3>🧪 Lab Markers</h3>',
        unsafe_allow_html=True
    )


    for feat in [
        "Total_Colesterol",
        "C_Reactive"
    ]:

        cfg = FIELD_CONFIG[feat]


        input_values[feat] = st.number_input(

            f"{cfg['label']} ({cfg['unit']})",

            min_value=float(cfg["min"]),

            max_value=float(cfg["max"]),

            value=float(cfg["default"]),

            step=float(cfg["step"]),

            help=cfg["help"],

            key=feat
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="section-card"><h3>📋 Summary</h3>',
        unsafe_allow_html=True
    )


    chips = "".join(

        f"""
        <span class="chip">
        <b>{FIELD_CONFIG[f]["label"]}:</b>
        {input_values[f]} {FIELD_CONFIG[f]["unit"]}
        </span>
        """

        for f in FEATURE_NAMES

    )


    st.markdown(
        f'<div class="chip-row">{chips}</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )



# ------------------------------------------------------------------
# Prediction Button
# ------------------------------------------------------------------

predict_clicked = st.button(
    "🔍 Predict Risk",
    use_container_width=True,
    type="primary"
)



# ------------------------------------------------------------------
# Prediction Output
# ------------------------------------------------------------------

if predict_clicked:


    ordered_input = np.array(
        [
            [
                input_values[f]
                for f in FEATURE_NAMES
            ]
        ]
    )


    prob = model.predict_proba(
        ordered_input
    )[0][1]


    level, css_class, emoji = risk_bucket(prob)



    st.markdown("---")



    res_col1, res_col2 = st.columns(
        [1, 1.2],
        gap="large"
    )



    with res_col1:


        st.markdown(

            f"""
            <div class="result-card {css_class}">

            <h2>
            {emoji} {level} Risk
            </h2>


            <div class="big-num">
            {prob*100:.1f}%
            </div>


            <p>
            Estimated probability of heart attack risk
            </p>


            </div>
            """,

            unsafe_allow_html=True

        )



        if level == "High":

            st.warning(
                "The model estimates elevated risk. "
                "Consider consulting a healthcare professional "
                "and reviewing lifestyle factors."
            )


        elif level == "Moderate":

            st.info(
                "The model estimates moderate risk. "
                "Regular checkups and healthy habits "
                "are recommended."
            )


        else:

            st.success(
                "The model estimates low risk based "
                "on the provided values."
            )



    with res_col2:

        st.plotly_chart(
            make_gauge(prob),
            use_container_width=True
        )



    # --------------------------------------------------------------
    # Explanation Chart
    # --------------------------------------------------------------

    st.markdown(
        "### 📊 What influenced this prediction?"
    )


    st.caption(
        "Bars show each factor's estimated push toward "
        "higher risk or lower risk based on the model."
    )


    st.plotly_chart(
        make_contribution_chart(
            model,
            input_values
        ),

        use_container_width=True
    )



else:


    st.info(
        "👈 Fill in patient details and click "
        "**Predict Risk** to see the result."
    )



# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------

st.markdown("---")


st.caption(
    "Built with Streamlit · Logistic Regression Model · "
    "For educational purposes only."
)
