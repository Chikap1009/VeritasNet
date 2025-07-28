import pandas as pd
import nltk
import re
import os
from tqdm import tqdm
nltk.download("punkt")

from nltk.tokenize import sent_tokenize

INPUT_FILE = "expanded_dataset.csv"
OUTPUT_FILE = "final_dataset.csv"

# Minimum and maximum length for a chunk (in words)
MIN_LEN = 5
MAX_LEN = 50

def clean_text(text):
    """Remove bad encodings, HTML junk, weird spacing."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\[[^\]]*\]', '', text)  # Remove [brackets]
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = text.encode('ascii', errors='ignore').decode()  # Remove non-ascii
    return text.strip()

def split_into_chunks(text):
    """Split article into smaller chunks of 2–3 sentences."""
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), 2):  # Slide every 2 sentences
        chunk = " ".join(sentences[i:i+2])
        word_count = len(chunk.split())
        if MIN_LEN <= word_count <= MAX_LEN:
            chunks.append(chunk)
    return chunks

def process_dataset(df):
    final_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = row.get("label", "").strip().lower()
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        clean = clean_text(text)
        chunks = split_into_chunks(clean)
        for chunk in chunks:
            final_rows.append({"text": chunk, "label": label})
    return pd.DataFrame(final_rows)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        return

    print("🔍 Reading expanded_dataset.csv...")
    df = pd.read_csv(INPUT_FILE)

    print("✨ Cleaning and chunking text...")
    cleaned_df = process_dataset(df)

    print(f"✅ Cleaned rows: {len(cleaned_df)}")
    cleaned_df.to_csv(OUTPUT_FILE, index=False)
    print(f"📦 Saved clean dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()