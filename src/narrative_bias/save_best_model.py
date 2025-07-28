from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Source checkpoint and destination
src = "models/narrative_bias/checkpoint-850"
dst = "models/narrative_bias/best_model"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(src)
tokenizer = AutoTokenizer.from_pretrained(src)

# Save them cleanly
model.save_pretrained(dst)
tokenizer.save_pretrained(dst)

print(f"✅ Best model saved to {dst}")