# src/narrative_bias/predict.py

import torch
import os
import numpy as np
import json # New import for saving/loading results if needed
import argparse # New import for command-line arguments

# Imports for your existing model and Captum
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# New imports for sentiment and zero-shot classification
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification # Auto classes for pipelines

# --- 1. Define label mapping (MUST match train.py) ---
LABEL2ID = {'neutral': 0, 'biased': 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# --- Configuration ---
# Your existing model path. Ensure this is correct relative to predict.py
MODEL_PATH_BIAS = "C:/Users/chira/VeritasNet/src/narrative_bias/models/narrative_bias/final_model"
# Adjusted to match your path

# Max length for your fine-tuned model's input
MAX_LEN_BIAS = 512 # Retained from your provided code

# For sentiment and zero-shot, we might use slightly different lengths,
# but the pipeline handles truncation internally.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define candidate labels for zero-shot bias classification
# These are examples, you can refine them based on your specific bias taxonomy
BIAS_FRAMING_LABELS = [
    "fear appeal",
    "conspiracy framing",
    "us-versus-them mentality",
    "blame attribution",
    "exaggeration",
    "minimization",
    "cherry picking",
    "straw man argument",
    "ad hominem attack",
    "bandwagon effect",
    "appeal to authority",
    "loaded language",
    "false dilemma",
    "slippery slope",
    "personal anecdote",
    "sensationalism", # Added based on your previous discussions
    "omission of facts" # Added
]

# Global variables for lazy loading models/pipelines
_bias_tokenizer = None
_bias_model = None
_sentiment_pipeline = None
_zero_shot_pipeline = None

# --- Model Loading Functions (Existing and New) ---

def load_bias_model(model_path=MODEL_PATH_BIAS):
    """Loads the fine-tuned bias detection model and tokenizer."""
    global _bias_tokenizer, _bias_model
    if _bias_model is None:  # Lazy load
        try:
            from transformers import logging
            logging.set_verbosity_error()  # Optional: hide warnings

            abs_path = os.path.abspath(model_path)
            print(f"📦 Attempting to load model from: {abs_path}")

            tokenizer = DistilBertTokenizerFast.from_pretrained(abs_path, local_files_only=True)
            model = DistilBertForSequenceClassification.from_pretrained(abs_path, local_files_only=True)

            model.to(DEVICE)
            model.eval()

            _bias_tokenizer = tokenizer
            _bias_model = model

            print(f"✅ Model and tokenizer loaded from: {abs_path}")
        except Exception as e:
            print(f"❌ Error loading model or tokenizer from: {abs_path}")
            print(f"    Exception: {e}")
    return _bias_tokenizer, _bias_model, DEVICE


def load_sentiment_pipeline():
    """Loads a pre-trained sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None: # Lazy load
        try:
            # Using a general-purpose sentiment model. You can choose others if needed.
            _sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
            print("Sentiment analysis pipeline loaded.")
        except Exception as e:
            print(f"Error loading sentiment pipeline: {e}")
    return _sentiment_pipeline

def load_zero_shot_pipeline():
    """Loads a pre-trained zero-shot classification pipeline."""
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None: # Lazy load
        try:
            _zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
            print("Zero-shot classification pipeline loaded.")
        except Exception as e:
            print(f"Error loading zero-shot pipeline: {e}")
    return _zero_shot_pipeline

# --- Prediction Functions (Existing and New) ---

def predict_bias(text, tokenizer, model, device):
    """Predicts bias (neutral/biased) and confidence for a given text."""
    if not tokenizer or not model:
        print("Bias model or tokenizer not loaded. Cannot predict bias.")
        return "Error", 0.0, None, None # Return error values
        
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=MAX_LEN_BIAS).to(device)
    
    with torch.no_grad(): # No need to calculate gradients for inference
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_label = ID2LABEL[predicted_class_id]
    confidence = probabilities[0, predicted_class_id].item()
    
    # Also return scores for both neutral and biased classes for detailed output
    neutral_score = probabilities[0, LABEL2ID['neutral']].item()
    biased_score = probabilities[0, LABEL2ID['biased']].item()

    return {
        "overall_prediction": predicted_label,
        "confidence": confidence,
        "inputs": inputs,
        "predicted_class_id": predicted_class_id,
        "neutral_score": neutral_score,
        "biased_score": biased_score
    }

def analyze_sentiment(text, sentiment_pipeline):
    """
    Analyzes the sentiment of the text using the provided pipeline.
    Ensures input is truncated to 512 tokens (model limit) and returns safe numeric output.
    """
    if not sentiment_pipeline:
        print("Sentiment pipeline not loaded. Cannot analyze sentiment.")
        return {"label": "N/A", "score": 0.0}

    try:
        # Truncate input text to max 512 tokens before running the pipeline
        encoded = sentiment_pipeline.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        truncated_text = sentiment_pipeline.tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)

        result = sentiment_pipeline(truncated_text)

        if not result or not isinstance(result[0], dict):
            return {"label": "Unknown", "score": 0.0}

        return {
            "label": result[0].get("label", "Unknown"),
            "score": float(result[0].get("score", 0.0))
        }

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return {"label": "Error", "score": 0.0}

def classify_bias_framing(text, zero_shot_pipeline, candidate_labels=BIAS_FRAMING_LABELS, multi_label=True):
    """
    Performs zero-shot classification to identify specific bias types/framing.
    Args:
        text (str): The text to classify.
        zero_shot_pipeline: The loaded zero-shot classification pipeline.
        candidate_labels (list): A list of potential bias/framing labels.
        multi_label (bool): If True, allows multiple labels to be predicted.
    Returns:
        dict: A dictionary of predicted labels and their scores.
    """
    if not zero_shot_pipeline:
        print("Zero-shot pipeline not loaded. Cannot classify bias framing.")
        return {"error": "Zero-shot pipeline not loaded."}
    
    # Zero-shot classification can be sensitive to very long inputs.
    # We might need to consider segmenting for very long transcripts in future.
    # For now, let the pipeline handle truncation up to its model's limit.
    try:
        result = zero_shot_pipeline(text, candidate_labels, multi_label=multi_label)
        
        # Filter and sort results by score
        # You can adjust the threshold or number of top N biases to report
        top_biases = {label: round(score, 4) for label, score in zip(result['labels'], result['scores']) if score > 0.15}

        if not top_biases:
            top_biases = {"No strong specific bias detected (threshold > 0.15)": "N/A"} # Or adjust this message

        return dict(sorted(top_biases.items(), key=lambda item: item[1], reverse=True)) # Sort by score
    except Exception as e:
        print(f"Error during zero-shot classification: {e}")
        return {"error": "Error during classification", "details": str(e)}

# --- Existing Explainability Functions ---
def predict_attributions(model, inputs, target_label_idx):
    """
    Computes Integrated Gradients attributions for input tokens.
    Uses LayerIntegratedGradients on the embedding layer.
    """
    # Define a custom forward function for LayerIntegratedGradients
    def forward_with_target(input_ids_tensor, attention_mask_tensor):
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        return outputs.logits[:, target_label_idx]

    # Initialize LayerIntegratedGradients with the adapted forward function and the embedding layer
    lig = LayerIntegratedGradients(forward_with_target, model.distilbert.embeddings)

    # Prepare inputs and baselines for the attribution method
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask'] 
    
    ref_input_ids = torch.zeros_like(input_ids).to(input_ids.device)
    ref_attention_mask = torch.zeros_like(attention_mask).to(attention_mask.device) 
    
    # Compute attributions
    attributions, delta = lig.attribute(
        (input_ids, attention_mask),               # Inputs to the forward_func
        baselines=(ref_input_ids, ref_attention_mask), # Baselines for inputs
        return_convergence_delta=True
    )
    
    # Sum attributions across the embedding dimension to get a single score per token
    attributions = attributions.sum(dim=-1).squeeze(0) # Shape: (sequence_length,)
    
    return attributions, delta

def _format_word_importances_html(tokens, attributions, predicted_label, cmap_name='bwr'):
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    # Strong Blue → White → Strong Red
    colors = [(0.0, 0.0, 0.8), (1.0, 1.0, 1.0), (0.8, 0.0, 0.0)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # Normalize attributions
    max_attr = max(attributions.max().item(), -attributions.min().item())
    norm_attributions = attributions / max_attr if max_attr != 0 else attributions * 0

    # Merge subword tokens (##ism → ism)
    merged_tokens = []
    merged_attributions = []
    current_token = ""
    current_attr = []

    for token, attr in zip(tokens, norm_attributions):
        if token.startswith("##"):
            current_token += token[2:]
            current_attr.append(attr.item())
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_attributions.append(np.mean(current_attr))
            current_token = token
            current_attr = [attr.item()]
    if current_token:
        merged_tokens.append(current_token)
        merged_attributions.append(np.mean(current_attr))

    # HTML rendering
    html_parts = []
    for token, attr in zip(merged_tokens, merged_attributions):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            html_parts.append(f'<span style="opacity:0.7; font-weight:bold; font-size: 14px;">{token}</span>')
            continue

        # Attribution to color
        color_value = (-attr + 1) / 2 if predicted_label.lower() == 'neutral' else (attr + 1) / 2
        rgba = cmap(color_value)
        hex_color = '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

        # Font color based on luminance
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        luminance = (0.299*r + 0.587*g + 0.114*b) / 255
        font_color = 'white' if luminance < 0.5 else 'black'

        # Bold + size bump for strong attribution
        style = "font-weight:700; font-size:17px;" if abs(attr) > 0.85 else "font-weight:600; font-size:16px;"

        html_parts.append(
            f'<span title="{token} ({attr:+.2f})" '
            f'style="background-color:{hex_color}; color:{font_color}; '
            f'padding: 2px 6px; margin: 2px; border-radius: 3px; '
            f'display:inline-block; {style}">{token}</span>'
        )

    return ' '.join(html_parts)

def visualize_attributions(attributions, tokenizer, text, predicted_label, confidence, inputs, delta):
    """Generates a complete HTML page with prediction and colored token attributions."""
    indexed_tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')[0].tolist()
    all_tokens_for_viz = tokenizer.convert_ids_to_tokens(indexed_tokens)

    colored_words_html = _format_word_importances_html(all_tokens_for_viz, attributions.cpu(), predicted_label)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VeritasNet Bias Explanation</title>
        <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 16px;
            color: #f0f0f0;
            background-color: #0d1117;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 15px;
        }}
        th, td {{
            border: 1px solid #444;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #161b22;
            color: #ffffff;
        }}
        td {{
            background-color: #1e242c;
        }}
        .legend {{
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #555;
            background-color: #161b22;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 15px;
            color: #ffffff;
        }}
        .color-box {{
            width: 20px;
            height: 20px;
            border: 1px solid #000;
            display: inline-block;
            vertical-align: middle;
            margin-right: 5px;
        }}
        .prediction-box {{
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            margin-top: 20px;
            display: inline-block;
            font-size: 20px;
        }}
        .prediction-neutral {{
            background-color: #ADD8E6;
            color: black;
        }}
        .prediction-biased {{
            background-color: #FF4040;
            color: white;
        }}
        </style>
    </head>
    <body>
        <h1>VeritasNet Bias Explanation</h1>
        <div class="prediction-box prediction-{predicted_label.lower()}">
            Predicted Label: {predicted_label.upper()} (Confidence: {confidence:.4f})
        </div>

        <table>
            <tr>
                <th>Original Text</th>
                <td>{text}</td>
            </tr>
            <tr>
                <th>Attribution Label</th>
                <td>{predicted_label.upper()}</td>
            </tr>
            <tr>
                <th>Attribution Score</th>
                <td>{confidence:.4f}</td>
            </tr>
            <tr>
                <th>Convergence Score</th>
                <td>{delta.item():.4f}</td>
            </tr>
            <tr>
                <th>Word Importance</th>
                <td>{colored_words_html}</td>
            </tr>
        </table>

        <div class="legend">
            <h3>Legend:</h3>
            <div class="legend-item">
                <span class="color-box" style="background-color: rgb(0,0,255);"></span> Strong Neutral Contribution
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: rgb(255,255,255);"></span> Neutral/No Contribution
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: rgb(255,0,0);"></span> Strong Biased Contribution
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# --- New Orchestration Function for Transcript Analysis ---
def analyze_transcript(transcript_path):
    """
    Reads a transcript file and performs comprehensive bias, sentiment,
    and zero-shot bias framing analysis.
    """
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found at {transcript_path}")
        return None

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read()

    if not transcript_text.strip():
        print("Error: Transcript file is empty.")
        return None

    print(f"\n--- Analyzing Transcript: {os.path.basename(transcript_path)} ---")

    # Load all models/pipelines (lazy loading handled by individual functions)
    tokenizer, model, device = load_bias_model() # Your fine-tuned BERT model
    sentiment_analyzer = load_sentiment_pipeline()
    zero_shot_classifier = load_zero_shot_pipeline()

    if not all([tokenizer, model, sentiment_analyzer, zero_shot_classifier]):
        print("One or more analysis models/pipelines failed to load. Aborting analysis.")
        return None

    # 1. Overall Bias Prediction (using your existing function)
    bias_result_dict = predict_bias(transcript_text, tokenizer, model, device)
    predicted_label = bias_result_dict["overall_prediction"]
    confidence = bias_result_dict["confidence"]
    inputs = bias_result_dict["inputs"]
    predicted_class_id = bias_result_dict["predicted_class_id"]

    print(f"\nOverall Bias Prediction: {predicted_label.upper()} (Confidence: {confidence:.4f})")
    print(f"  Neutral Score: {bias_result_dict['neutral_score']:.4f}, Biased Score: {bias_result_dict['biased_score']:.4f}")

    # 2. Sentiment Analysis
    sentiment_result = analyze_sentiment(transcript_text, sentiment_analyzer)
    print(f"\nSentiment Analysis: Label='{sentiment_result['label']}' (Score={sentiment_result['score']:.4f})")

    # 3. Specific Bias/Framing Classification (Zero-Shot)
    print("\nSpecific Bias/Framing Detection (Zero-Shot):")
    framing_results = classify_bias_framing(transcript_text, zero_shot_classifier, BIAS_FRAMING_LABELS)
    for label, score in framing_results.items():
        print(f"  - {label}: {score}")

    # You can also generate an HTML explanation for the full transcript,
    # but be aware that for very long transcripts, the HTML might be huge
    # and Captum attributions might be less interpretable over many tokens.
    # For now, we'll focus on the console output for full transcripts.
    # If you want HTML for the full transcript, uncomment and adapt:
    # attributions, delta = predict_attributions(model, inputs.to(device), predicted_class_id)
    # html_output = visualize_attributions(attributions.cpu(), tokenizer, transcript_text, predicted_label, confidence, inputs, delta)
    # output_html_file = f"transcript_explanation_{os.path.basename(transcript_path).replace('.txt', '')}.html"
    # with open(output_html_file, "w", encoding="utf-8") as f:
    #     f.write(html_output)
    # print(f"\nFull transcript explanation (HTML) saved to {output_html_file}")

    return {
        "transcript_path": transcript_path,
        "overall_bias": {
            "prediction": predicted_label,
            "confidence": f"{confidence:.4f}",
            "neutral_score": f"{bias_result_dict['neutral_score']:.4f}",
            "biased_score": f"{bias_result_dict['biased_score']:.4f}"
        },
        "sentiment": sentiment_result,
        "specific_biases": framing_results
    }

def chunk_text(text, tokenizer, max_length=512, stride=256):
    """
    Splits long text into overlapping chunks using the tokenizer.
    Returns a list of dictionaries with chunk metadata:
        - text: Decoded chunk text
        - start_token: Starting token index
        - end_token: Ending token index
    """
    encodings = tokenizer(text, truncation=False, return_offsets_mapping=True, return_tensors='pt')
    input_ids = encodings['input_ids'][0]
    total_tokens = len(input_ids)

    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + max_length, total_tokens)
        chunk_ids = input_ids[start:end]
        chunk_str = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        chunks.append({
            "text": chunk_str,
            "start_token": start,
            "end_token": end
        })

        if end == total_tokens:
            break
        start += stride
    return chunks

def analyze_long_text(text, tokenizer, model, device, sentiment_pipeline, zero_shot_pipeline):
    """
    Analyzes long text by chunking and running bias, sentiment, and framing analysis on each chunk.
    Returns:
        - Overall bias and sentiment
        - Aggregated framing scores
        - All chunk-wise results
    """
    chunks = chunk_text(text, tokenizer)
    all_chunk_results = []

    for chunk in chunks:
        chunk_str = chunk["text"]

        bias = predict_bias(chunk_str, tokenizer, model, device)
        sentiment = analyze_sentiment(chunk_str, sentiment_pipeline)
        framing = classify_bias_framing(chunk_str, zero_shot_pipeline)

        all_chunk_results.append({
            "text": chunk_str,
            "start_token": chunk["start_token"],
            "end_token": chunk["end_token"],
            "bias": bias,
            "sentiment": sentiment,
            "framing": framing,
            "raw_inputs": bias.get("inputs"),
            "predicted_class_id": bias.get("predicted_class_id")
        })

    # Safely extract confidences and sentiment scores
    bias_confidences = [float(c["bias"].get("confidence", 0.0)) for c in all_chunk_results]
    sentiment_scores = [float(c["sentiment"].get("score", 0.0)) for c in all_chunk_results]
    bias_labels = [c["bias"].get("overall_prediction", "unknown") for c in all_chunk_results]
    sentiment_labels = [c["sentiment"].get("label", "unknown") for c in all_chunk_results]

    # Aggregate framing
    aggregated_framing = {}
    for c in all_chunk_results:
        for tag, score in c["framing"].items():
            try:
                numeric_score = float(score)
                aggregated_framing[tag] = aggregated_framing.get(tag, 0.0) + numeric_score
            except:
                continue

    for tag in aggregated_framing:
        aggregated_framing[tag] /= len(all_chunk_results)

    # Overall decisions
    overall_bias = max(set(bias_labels), key=bias_labels.count)
    overall_sentiment = max(set(sentiment_labels), key=sentiment_labels.count)
    avg_confidence = np.mean(bias_confidences)
    avg_sentiment_score = np.mean(sentiment_scores)
    top_confidence_chunk = max(all_chunk_results, key=lambda c: float(c["bias"].get("confidence", 0.0)))

    return {
        "bias": {
            "label": overall_bias,
            "confidence": float(avg_confidence)
        },
        "sentiment": {
            "label": overall_sentiment,
            "score": float(avg_sentiment_score)
        },
        "framing": aggregated_framing,
        "chunks": all_chunk_results
    }

# --- Main execution block (Modified) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text or a video transcript for bias, sentiment, and specific framing with explainability.")
    parser.add_argument("--text", type=str, help="A single text string to analyze for bias and generate explanation.")
    parser.add_argument("--transcript_file", type=str,
                        help="Path to a .txt file containing the video transcript for comprehensive analysis.")
    
    args = parser.parse_args()

    # Load all pipelines once globally when the script starts if any analysis is requested
    # These will be loaded lazily the first time they are called
    load_bias_model()
    load_sentiment_pipeline()
    load_zero_shot_pipeline()

    if args.text:
        print("\n--- Analyzing Single Text with Explainability ---")
        text_to_analyze = args.text
        
        # Predict bias
        bias_result_dict = predict_bias(text_to_analyze, _bias_tokenizer, _bias_model, DEVICE)
        predicted_label = bias_result_dict["overall_prediction"]
        confidence = bias_result_dict["confidence"]
        inputs = bias_result_dict["inputs"]
        predicted_class_id = bias_result_dict["predicted_class_id"]

        print(f"Predicted Label: {predicted_label.upper()} (Confidence: {confidence:.4f})")
        
        # Generate and visualize attributions
        if inputs is not None: # Check if inputs were successfully created
            inputs_on_device = {k: v.to(DEVICE) for k,v in inputs.items()}
            attributions, delta = predict_attributions(_bias_model, inputs_on_device, predicted_class_id)
            
            html_output = visualize_attributions(attributions.cpu(), _bias_tokenizer, text_to_analyze, predicted_label, confidence, inputs, delta)
            
            output_html_file = f"explanation_single_text_{predicted_label.lower()}.html" 
            with open(output_html_file, "w", encoding="utf-8") as f:
                f.write(html_output)
            print(f"Explanation saved to {output_html_file}")
            print("Please open this HTML file in your web browser to view the explanation.")
        else:
            print("Could not generate explainability HTML due to prediction error.")

    elif args.transcript_file:
        analysis_results = analyze_transcript(args.transcript_file)
        if analysis_results:
            print("\n--- Comprehensive Transcript Analysis Complete ---")
            # You can save the full results dictionary to a JSON file if desired
            # with open("comprehensive_transcript_analysis_results.json", "w", encoding="utf-8") as f:
            #     json.dump(analysis_results, f, indent=4)
            # print("Full analysis results saved to comprehensive_transcript_analysis_results.json")
        else:
            print("\n--- Comprehensive Transcript Analysis Failed ---")
    else:
        print("\n--- VeritasNet Prediction & Explanation (Example Usage) ---")
        print("No --text or --transcript_file argument provided. Running example tests.")
        
        test_texts = [
            "The new policy was implemented on Tuesday after a lengthy discussion by the committee.", # Neutral example
            "They want to control your thoughts, but we will fight for freedom against their tyrannical agenda.", # Biased example
            "Scientists discovered a new exoplanet, significantly expanding our understanding of the universe.", # Another neutral
            "The mainstream media consistently distorts the truth to fit their hidden agenda, don't believe a word they say!" # Another biased
        ]

        for i, text_to_analyze in enumerate(test_texts):
            print(f"\nAnalyzing example text {i+1}: '{text_to_analyze}'")
            
            # Predict bias
            bias_result_dict = predict_bias(text_to_analyze, _bias_tokenizer, _bias_model, DEVICE)
            predicted_label = bias_result_dict["overall_prediction"]
            confidence = bias_result_dict["confidence"]
            inputs = bias_result_dict["inputs"]
            predicted_class_id = bias_result_dict["predicted_class_id"]

            print(f"Predicted Label: {predicted_label.upper()} (Confidence: {confidence:.4f})")
            
            # Generate and visualize attributions
            if inputs is not None:
                inputs_on_device = {k: v.to(DEVICE) for k,v in inputs.items()}
                attributions, delta = predict_attributions(_bias_model, inputs_on_device, predicted_class_id)
                
                output_html_file = f"explanation_example_{predicted_label.lower()}_{i+1}.html" 
                html_output = visualize_attributions(attributions.cpu(), _bias_tokenizer, text_to_analyze, predicted_label, confidence, inputs, delta)
                with open(output_html_file, "w", encoding="utf-8") as f:
                    f.write(html_output)
                print(f"Explanation saved to {output_html_file}")
                print("Please open this HTML file in your web browser to view the explanation.")
            else:
                print("Could not generate explainability HTML due to prediction error.")