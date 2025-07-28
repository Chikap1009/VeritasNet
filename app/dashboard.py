# app/dashboard.py

import streamlit as st
import sys, os
import torch

# Make sure the src/narrative_bias path is added
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'narrative_bias')))

# Core imports from narrative_bias
from predict import (
    predict_bias,
    load_bias_model,
    analyze_sentiment,
    classify_bias_framing,
    load_zero_shot_pipeline,
    load_sentiment_pipeline,
    predict_attributions,
    visualize_attributions,
    analyze_long_text,
)

# YouTube transcript processor
from video_transcriber import process_video_for_transcript

# === Load model + tokenizer ===
tok, mod, dev = load_bias_model()
if not tok or not mod:
    st.error("❌ Failed to load VeritasNet model or tokenizer. Please check your model folder.")
    st.stop()
tokenizer = tok
model = mod
device = dev

# === Streamlit App Config ===
st.set_page_config(page_title="VeritasNet Dashboard", layout="wide")
st.title("🧠 VeritasNet – Narrative Bias Analyzer")
st.markdown("""
Welcome to **VeritasNet** — an advanced AI tool that detects **bias**, **sentiment**, and **framing techniques** in news, narratives, and transcripts.
""")

# === Input Mode Selection ===
st.sidebar.header("📥 Input Options")
input_mode = st.sidebar.radio("Choose input mode:", ["Paste Text", "Upload .txt File", "YouTube URL"])

text_to_analyze = ""
transcript_path = None

# === Input Methods ===
if input_mode == "Paste Text":
    st.subheader("✍️ Paste or Type Your Narrative")
    if 'veritas_input_text' in st.session_state:
        text_to_analyze = st.session_state['veritas_input_text']
    text_to_analyze = st.text_area("Enter your text here:", value=text_to_analyze, height=200)

elif input_mode == "Upload .txt File":
    st.subheader("📄 Upload a Transcript File")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
    if uploaded_file:
        text_to_analyze = uploaded_file.read().decode("utf-8")
        st.text_area("Transcript Preview:", text_to_analyze[:1000], height=200, disabled=True)

elif input_mode == "YouTube URL":
    st.subheader("🎬 Enter YouTube Link for Analysis")
    youtube_url = st.text_input("Paste YouTube video URL:")
    if st.button("📥 Transcribe and Analyze YouTube Video"):
        if not youtube_url.strip():
            st.warning("Please enter a YouTube URL.")
        else:
            with st.spinner("Downloading & transcribing video using Whisper..."):
                try:
                    _, transcript_path, text_to_analyze = process_video_for_transcript(youtube_url)
                    st.text(f"[DEBUG] Transcript loaded: {bool(text_to_analyze)} | Length: {len(text_to_analyze) if text_to_analyze else 0}")
                    if not text_to_analyze:
                        st.error("Failed to transcribe video.")
                except Exception as e:
                    st.error(f"Error processing video: {e}")

# === ANALYSIS BLOCK ===
if (input_mode != "YouTube URL" and st.button("🔍 Analyze")) or (input_mode == "YouTube URL" and text_to_analyze):

    if not text_to_analyze.strip():
        st.warning("Please provide some input to analyze.")
    else:
        # Save to session state
        st.session_state['veritas_input_text'] = text_to_analyze

        tokenizer_tmp = tokenizer
        tokens = tokenizer_tmp.tokenize(text_to_analyze)
        max_input_tokens = 512

        if len(tokens) > max_input_tokens:
            st.warning(f"Transcript is too long ({len(tokens)} tokens). Running chunked analysis.")
            try:
                sentiment_pipeline = load_sentiment_pipeline()
                zero_shot_pipeline = load_zero_shot_pipeline()

                results = analyze_long_text(
                    text_to_analyze,
                    tokenizer,
                    model,
                    device,
                    sentiment_pipeline,
                    zero_shot_pipeline
                )

                st.session_state['veritas_chunk_results'] = results

            except Exception as e:
                st.error(f"Error in chunked analysis: {e}")
                st.stop()
        else:
            result = predict_bias(text_to_analyze, tokenizer, model, device)

            if not isinstance(result, dict) or 'overall_prediction' not in result:
                st.error("Something went wrong during bias prediction.")
                st.stop()

            label = result['overall_prediction'].upper()
            confidence = float(result['confidence'])

            sentiment_pipeline = load_sentiment_pipeline()
            sentiment_result = analyze_sentiment(text_to_analyze, sentiment_pipeline)
            sentiment_label = sentiment_result['label']
            sentiment_score = float(sentiment_result['score'])

            predicted_class_id = result['predicted_class_id']
            inputs_on_device = {k: v.to(device) for k, v in result['inputs'].items()}

            attributions, delta = predict_attributions(model, inputs_on_device, predicted_class_id)

            html_explanation = visualize_attributions(
                attributions.cpu(),
                tokenizer,
                text_to_analyze,
                label,
                confidence,
                result['inputs'],
                delta
            )

            st.success(f"Prediction: **{label}** with confidence **{confidence:.2f}**")
            st.info(f"Sentiment: **{sentiment_label}** ({sentiment_score:.2f})")
            st.markdown("---")
            st.subheader("🔍 Word-Level Explanation")
            st.components.v1.html(html_explanation, height=600, scrolling=True)

            framing_results = classify_bias_framing(text_to_analyze, load_zero_shot_pipeline())
            st.markdown("### 🧾 Specific Bias Framings Detected:")
            for tag, score in framing_results.items():
                st.markdown(f"- **{tag}** — {score}")

# === Handle chunked results if already cached ===
results = st.session_state.get('veritas_chunk_results', {})
text_to_analyze = st.session_state.get('veritas_input_text', '')

if results and 'chunks' in results:
    st.subheader("🧩 Per-Chunk Analysis")
    for i, chunk in enumerate(results.get('chunks', [])):
        chunk_str = chunk.get('text', '')
        bias_info = chunk.get('bias', {})
        sentiment_info = chunk.get('sentiment', {})
        framing = chunk.get('framing', {})

        label = bias_info.get('overall_prediction', 'unknown')
        confidence = bias_info.get('confidence', 0.0)
        sentiment = sentiment_info.get('label', 'unknown')
        sentiment_score = sentiment_info.get('score', 0.0)

        if not chunk_str or not label:
            st.warning(f"⚠️ Skipping Chunk {i+1} due to missing data.")
            continue

        if confidence >= 0.85:
            bias_marker = "🟥"
        elif confidence >= 0.5:
            bias_marker = "🟨"
        else:
            bias_marker = "🟩"

        with st.expander(f"{bias_marker} Chunk {i+1} — {label.upper()} ({confidence:.2f} confidence)"):
            st.markdown(f"**Sentiment:** {sentiment} ({sentiment_score:.2f})")
            st.markdown("**Framing Tags:**" if framing else "*No strong framing tags detected.*")
            for tag, score in framing.items():
                st.markdown(f"- **{tag}** — {float(score):.4f}")

            st.markdown("**Text Preview:**")
            st.text_area(f"Chunk Text #{i+1}", chunk_str, height=180, disabled=True, key=f"chunk_text_{i}")

            show_attr = st.toggle(f"🧠 Show Explanation for Chunk {i+1}", key=f"attr_toggle_{i}")
            if show_attr:
                try:
                    cache_key = f"explanation_html_{i}"
                    if cache_key in st.session_state:
                        st.components.v1.html(st.session_state[cache_key], height=600, scrolling=True)
                    else:
                        inputs = chunk.get('raw_inputs')
                        predicted_class_id = chunk.get('predicted_class_id')

                        if not inputs or predicted_class_id is None:
                            st.error("Missing attribution inputs or class ID.")
                            continue

                        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
                        model.eval()
                        with torch.no_grad():
                            attributions, delta = predict_attributions(model, inputs_on_device, predicted_class_id)

                        html_explanation = visualize_attributions(
                            attributions.cpu(),
                            tokenizer,
                            chunk_str,
                            label,
                            confidence,
                            inputs,
                            delta
                        )

                        st.session_state[cache_key] = html_explanation
                        st.components.v1.html(html_explanation, height=600, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to generate explanation: {e}")