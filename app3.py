# ==================================================
# Comment Toxicity Detection Streamlit App
# ==================================================

import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

# ==========================================
# Load model
# ==========================================

model = tf.keras.models.load_model(
    'model_LSTM/lstm_model.h5'
)

tokenizer = pickle.load(
    open(
        'model_LSTM/tokenizer.pkl',
        'rb'
    )
)

# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Comment Toxicity Detector",
    layout="wide"
)

st.title(
    "Comment Toxicity Detection System"
)

st.markdown(
"""
Detect toxic comments using Deep Learning models
"""
)

st.markdown("""
<style>

/* App background */
.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827
    );
    color:white;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:#1f2937;
}

/* Main title */
h1{
    text-align:center;
    color:#f8fafc;
}

/* Section headers */
h2,h3{
    color:#f59e0b;
}

/* Input box */
textarea{
    background-color:#374151 !important;
    color:white !important;
    border-radius:12px !important;
}

/* Predict button */
.stButton>button{

    background:linear-gradient(
        90deg,
        #22c55e,
        #eab308,
        #f97316,
        #ef4444
    );

    color:white;

    border:none;

    border-radius:12px;

    font-size:18px;

    font-weight:bold;

    height:50px;

    width:150px;
}

.stButton>button:hover{
    opacity:0.9;
}

/* Tables */
[data-testid="stTable"]{
    background-color:#374151;
    border-radius:10px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.header(
    "Dashboard"
)

page = st.sidebar.radio(
    "Select",
    [
        "Prediction",
        "Bulk Upload",
        "Dashboard"

    ]
)

labels = [

    'Toxic',
    'Severe Toxic',
    'Obscene',
    'Threat',
    'Insult',
    'Identity Hate'
]

# ==========================================
# Dashboard
# ==========================================

if page=="Dashboard":

    st.header(
        "Dataset Insights & Model Performance"
    )

    # ===========================
    # Dataset statistics
    # ===========================

    st.subheader(
        "Dataset Information"
    )

    total_comments = 159571
    total_labels = 6

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Total Comments",
        total_comments
    )

    col2.metric(
        "Toxic Categories",
        total_labels
    )

    col3.metric(
        "Model",
        "LSTM"
    )

    # ===========================
    # Model Metrics
    # ===========================

    st.subheader(
        "Model Performance"
    )

    metrics_df = pd.DataFrame({

        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Score":[
            0.99,
            0.82,
            0.75,
            0.78
        ]

    })

    st.table(
        metrics_df
    )


    # ===========================
    # Performance chart
    # ===========================

    fig = px.bar(

        metrics_df,

        x="Metric",

        y="Score",

        text="Score",

        color="Score"
    )

    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    fig.update_layout(
        yaxis=dict(range=[0,1.1])
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ===========================
    # Sample Test Cases
    # ===========================

    st.subheader(
        "Sample Test Cases"
    )

    sample = pd.DataFrame({

        "Comment":[

            "I love you",

            "You are stupid",

            "Have a wonderful day",

            "You idiot"
        ],

        "Expected Result":[

            "Safe",

            "Toxic",

            "Safe",

            "Toxic"
        ]
    })

    st.table(
        sample
    )

# ==========================================
# Single prediction
# ==========================================

if page=="Prediction":

    st.header(
        "Real Time Prediction"
    )

    text = st.text_area(
        "Enter comment"
    )

    if st.button(
        "Predict"
    ):

        seq = tokenizer.texts_to_sequences(
            [text]
        )

        padded = pad_sequences(
            seq,
            maxlen=100
        )

        pred = model.predict(
            padded
        )[0]

        result = {}

        # Store predictions
        for i, label in enumerate(labels):

            score = round(
                float(pred[i]),
                2
            )

            result[label] = score


        # ==============================
        # Prediction Table
        # ==============================

        st.subheader(
            "Prediction Scores"
        )


        def severity(score):

            if score < 0.30:
                return "🟢 Safe"

            elif score < 0.60:
                return "🟡 Warning"

            elif score < 0.80:
                return "🟠 High"

            else:
                return "🔴 Toxic"


        score_df = pd.DataFrame({

            "Toxicity Category":
            list(result.keys()),

            "Prediction Score":
            list(result.values()),

            "Severity":
            [severity(x)
             for x in result.values()]

        })

        st.table(
            score_df
        )

        # ==============================
        # Visualization
        # ==============================

        st.subheader(
            "Toxicity Visualization"
        )

        chart_df = pd.DataFrame({

            "Category":
            list(result.keys()),

            "Score":
            list(result.values()),

            "Severity":
            [severity(x)
            for x in result.values()]
        })


        fig = px.bar(

            chart_df,

            x="Category",

            y="Score",

            text="Score",

            color="Severity",

            color_discrete_map={

                "🟢 Safe":"green",

                "🟡 Warning":"yellow",

                "🟠 High":"orange",

                "🔴 Toxic":"red"
            }

        )

        fig.update_layout(

            yaxis=dict(
                range=[0,1.15]
            ),

            xaxis_title=
            "Toxicity Category",

            yaxis_title=
            "Prediction Score",

            margin=dict(
                t=60
            )
        )

        fig.update_traces(

            texttemplate='%{text:.2f}',

            textposition='outside',

            cliponaxis=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
)# ==========================================
# Bulk upload
# ==========================================

elif page=="Bulk Upload":

    st.header(
        "CSV Prediction"
    )

    file = st.file_uploader(
        "Upload csv",
        type=['csv']
    )

    if file:

        df = pd.read_csv(
            file
        )

        st.write(
            "Uploaded Data"
        )

        st.dataframe(
            df.head()
        )

        texts = df["comment_text"].fillna("").astype(str)

        seq = tokenizer.texts_to_sequences(texts)


        padded = pad_sequences(
            seq,
            maxlen=100
        )

        pred = model.predict(
            padded
        )

        pred_df = pd.DataFrame(
            pred,
            columns=labels
        )

        final = pd.concat(
            [df,pred_df],
            axis=1
        )

        st.dataframe(
            final.head()
        )

        csv = final.to_csv(
            index=False
        )

        st.download_button(
            "Download Results",
            csv,
            "prediction.csv"
        )