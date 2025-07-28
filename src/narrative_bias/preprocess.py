# src/narrative_bias/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset

# 1️⃣ Define label mapping
LABEL2ID = {'neutral': 0, 'biased': 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 2️⃣ Load dataset from CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['label'] = df['label'].map(LABEL2ID)
    return df

# 3️⃣ Custom PyTorch dataset class
class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels  # <– keep as-is if already 0 and 1 integers

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 4️⃣ Master function to prepare train & test datasets
def prepare_datasets(csv_path, test_size=0.2):
    df = load_data(csv_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=test_size,
        stratify=df['label'],
        random_state=42
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_dataset = BiasDataset(train_texts, train_labels, tokenizer)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer)

    return train_dataset, val_dataset, tokenizer