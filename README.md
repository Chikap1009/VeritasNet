# VeritasNet 🧠🔍  
*AI-Powered Multi-Modal Bias Detection System*

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/UI-Built%20with%20Streamlit-orange)](https://streamlit.io/)

---

## 🧠 Overview

**VeritasNet** is an advanced AI system for detecting **narrative bias**, **sentiment**, and **framing** in multi-modal content — including raw text, transcripts, and YouTube videos.  
It uses fine-tuned transformer models to classify narratives as **Biased** or **Neutral** and provides **explanations** via **Captum**.

---

## 🚀 Features

- 🔍 Binary classification: **Biased vs Neutral**
- 🧠 **Explainable AI** with token-level attribution via Captum
- 🎯 Support for:
  - ✍️ Raw text
  - 📺 YouTube transcripts
  - 📄 Long-form chunked analysis
- 📊 Outputs:
  - Bias classification
  - Sentiment detection
  - Framing category
- 🌐 Interactive **Streamlit dashboard**

---

## 🗂️ Project Structure

VeritasNet/
├── app/
│ └── dashboard.py # Streamlit frontend
├── src/
│ └── narrative_bias/
│ ├── predict.py # Orchestrates full analysis
│ ├── preprocess.py # Text cleaning + chunking
│ ├── preprocess_expanded.py# Cleans scraped dataset
│ ├── video_transcriber.py # YouTube transcript downloader
│ ├── label_assistant.py # Semi-auto labeling helper
│ ├── train.py # Model training (optional)
│ └── models/ # Tokenizer/config files (weights excluded)
├── requirements.txt # Required packages
└── README.md # Project documentation

---

## 🛠️ Setup Instructions

### 🧰 Requirements

- Python 3.10+
- Git

### 📦 Install Dependencies

pip install -r requirements.txt

---

### 💻 Run the Streamlit App

streamlit run app/dashboard.py


- Supports text input or YouTube URLs
- Visualizes bias predictions, framing, sentiment, and token attribution
- Handles long-form transcripts via chunking

---

## 📚 Model Details

| Task              | Model                     |
|-------------------|---------------------------|
| Bias Detection    | BERT / RoBERTa (fine-tuned) |
| Sentiment         | DistilBERT (pretrained)   |
| Framing Detection | Custom BERT fine-tuned    |
| Explainability    | Captum (Integrated Gradients) |

> Models are trained on labeled real-world narratives using custom bias + framing tags.

---

## 🧪 Sample Prediction

### Input:

The corrupt regime has blatantly ignored the needs of the poor.


### Output:
- **Bias:** 🟥 Likely Biased  
- **Sentiment:** Negative  
- **Framing:** Political / Class-Based  
- **Attribution:** Highlights “corrupt”, “ignored”, “poor”

---

## 🧼 Git Hygiene

This repo uses a `.gitignore` to exclude:

- ✅ Large model files (`*.pt`, `*.safetensors`, `*.bin`)
- ✅ Raw datasets (`.csv`, `.json`)
- ✅ Checkpoints, logs, and HTML explanations
- ✅ Downloaded videos & transcripts

To use this project fully, you’ll need to load your own trained weights, or host them externally (e.g., Hugging Face).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Credits

- 🤗 Hugging Face Transformers  
- 📊 Captum for Explainability  
- 🌐 Streamlit for UI  
- 🎥 YouTubeTranscriptAPI  
- 🔥 PyTorch

---

## 💬 Contact

For questions or collaboration:  
📧 chiragkapoor1009@gmail.com  
GitHub: [@Chikap1009](https://github.com/Chikap1009)