# Project_5 - Comment_Toxicity

**Project Overview**

The Comment Toxicity Detection System is an NLP and Deep Learning based application developed to identify and classify toxic comments in online text. The system performs multi-label classification and predicts multiple toxicity categories such as Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate.

This project uses Deep Learning architectures including RNN, LSTM, and experiments with Transformer-based approaches. The final deployment model uses LSTM due to better performance efficiency and lower computational cost.

The project includes a Streamlit web application that allows users to:

- Predict toxicity in real-time
- Visualize prediction scores
- View toxicity severity indicators
- Upload CSV files for bulk predictions
- Explore model metrics and sample test cases
- Problem Statement

Online platforms generate large volumes of user-generated content. Toxic comments negatively impact user experience and require automated moderation systems.

The objective of this project is to build an intelligent system capable of automatically detecting toxic content using Natural Language Processing and Deep Learning techniques

## Problem Statement

Online platforms generate large volumes of user-generated content. Toxic comments negatively impact user experience and require automated moderation systems.

The objective of this project is to build an intelligent system capable of automatically detecting toxic content using Natural Language Processing and Deep Learning techniques.

## Dataset

Dataset: Jigsaw Toxic Comment Classification Dataset

Target Labels:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## Project Workflow
- Data Collection
- Data Exploration
- Text Preprocessing
- Feature Engineering
- Model Development
- Model Evaluation
- Model Deployment
- Streamlit Application Development

The following preprocessing techniques were applied:
- Lowercasing text
- Removing punctuation
- Removing special characters
- Tokenization
- Stopword removal
- Sequence generation
- Padding sequences

## Text Preprocessing
Libraries used:
- NLTK
- TensorFlow Keras
- Pandas

## Model Development

Implemented models:

**1. Simple RNN**
- Embedding Layer
- SimpleRNN Layer
- Dropout
- Dense Output Layer

**2. LSTM**
- Embedding Layer
- LSTM Layer
- Dropout
- Dense Output Layer

**3. Transformer Experiment**
- DistilBERT
- Multi-label classification setup

## Final selected model:

✅ LSTM

**Reason:**
- Lower validation loss
- Better sequence learning capability
- Faster deployment compared to BERT
- Lower computational requirements
- Model Optimization Techniques

**Applied:**
- EarlyStopping
- ReduceLROnPlateau
- Dropout Regularization
- Binary Cross Entropy Loss

## Streamlit Features
**1. Dashboard**
- Dataset insights
- Model metrics
- Sample test cases

**2. Prediction Module**
- Real-time comment prediction
- Toxicity severity indicators
- Interactive visualizations

**3. Bulk Upload Module**
- CSV upload
- Batch predictions
- Download prediction results

## Toxicity Severity Scale

🟢 Safe

🟡 Warning

🟠 High

🔴 Toxic

## Tech Stack

**Programming Language:**
- Python

**Libraries:**
- TensorFlow
- Keras
- NLTK
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit


```Text
Comment-Toxicity/ 
├── notebooks/
│   └── Toxicity_Detection.ipynb
├── Input
|   └── Trani.csv (since the file size is high, can't able to upload it)
├── Output
|   └── Train_processed_text.csv
├── model_LSTM/
│   ├── lstm_model.h5
│   └── tokenizer.pkl
├── app.py
│
└── README.md
```
