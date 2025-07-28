import os
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.nn import CrossEntropyLoss # Crucial for custom loss with weights
from torch.utils.data import Dataset # Base class for custom dataset
from collections import Counter # To count labels for weighting
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1️⃣ Define label mapping ---
# Ensure this mapping is consistent with your dataset: 0 for neutral, 1 for biased.
LABEL2ID = {'neutral': 0, 'biased': 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# --- 2️⃣ Custom PyTorch dataset class ---
class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        # Tokenize all texts upfront for efficiency
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512) # Added max_length
        self.labels = labels

    def __getitem__(self, idx):
        # Return tokenized inputs and corresponding label for a given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.labels)

# --- 3️⃣ Master function to prepare train & test datasets and calculate weights ---
def prepare_datasets(csv_path, test_size=0.2):
    # Load data from the specified CSV file
    df = pd.read_csv(csv_path)

    # Map string labels to integer IDs using LABEL2ID
    df['label'] = df['label'].map(LABEL2ID)
    # Drop any rows where label mapping failed (e.g., unexpected label strings)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int) # Ensure labels are integer type

    # Stratified split: important to maintain the proportion of labels in train/val sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=test_size,
        stratify=df['label'], # This ensures balanced splits for imbalanced data
        random_state=42 # For reproducibility
    )

    # Initialize the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Create custom Dataset objects for training and validation
    train_dataset = BiasDataset(train_texts, train_labels, tokenizer)
    val_dataset = BiasDataset(val_texts, val_labels, tokenizer)

    # Calculate class weights for the training set to handle potential imbalance
    # This helps the model pay more attention to the minority class
    label_counts = Counter(train_labels)
    total_samples = sum(label_counts.values())
    num_classes = len(LABEL2ID)

    # Calculate inverse class weights: smaller class count gets a higher weight
    # Formula: total_samples / (num_classes * count_of_class)
    class_weights = torch.tensor([
        total_samples / (num_classes * label_counts[LABEL2ID['neutral']]), # Weight for neutral (0)
        total_samples / (num_classes * label_counts[LABEL2ID['biased']])   # Weight for biased (1)
    ], dtype=torch.float)

    # Move weights to GPU if available, as the model will be on GPU
    if torch.cuda.is_available():
        class_weights = class_weights.to('cuda')

    print(f"Calculated Class Weights (Neutral: {class_weights[0]:.4f}, Biased: {class_weights[1]:.4f})")

    return train_dataset, val_dataset, tokenizer, class_weights


# --- 4️⃣ Custom Trainer to apply class weights in the loss function ---
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize class weights; default to ones if not provided
        self.class_weights = class_weights if class_weights is not None else torch.ones(self.model.config.num_labels)
        # Ensure weights are on the same device as the model for computation
        if self.model.device.type == 'cuda':
             self.class_weights = self.class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels from inputs
        labels = inputs.pop("labels")
        # Forward pass through the model
        outputs = model(**inputs)
        logits = outputs.logits
        # Use CrossEntropyLoss with the calculated class weights
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        # Compute loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # Return loss and outputs
        return (loss, outputs) if return_outputs else loss


# --- 5️⃣ Define compute_metrics for evaluation ---
def compute_metrics(p):
    # Predictions are logits, labels are true labels
    predictions = p.predictions
    labels = p.label_ids

    # Get predicted class by taking argmax of logits
    preds = np.argmax(predictions, axis=1)

    # Calculate F1-score (binary for two classes) and accuracy
    f1 = f1_score(labels, preds, average='binary')
    accuracy = accuracy_score(labels, preds)

    # Return metrics as a dictionary
    return {"f1": f1, "accuracy": accuracy}


# --- Main execution block ---
if __name__ == "__main__":
    # Define the path to your dataset CSV file
    DATASET_PATH = "FINAL_dataset_complete.csv" # <--- IMPORTANT: This should point to your cleaned CSV!

    # Prepare datasets, tokenizer, and get calculated class weights
    train_dataset, val_dataset, tokenizer, class_weights = prepare_datasets(DATASET_PATH)

    # Load the pre-trained DistilBERT model for sequence classification
    # num_labels must match the number of classes (neutral, biased)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(LABEL2ID))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',                          # Directory to save logs and checkpoints
        num_train_epochs=10,                             # Total number of training epochs (can be stopped early by callback)
        per_device_train_batch_size=16,                  # Batch size per device during training
        per_device_eval_batch_size=16,                   # Batch size for evaluation
        warmup_steps=500,                                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                               # Strength of weight decay (L2 regularization)
        logging_dir='./logs',                            # Directory for storing TensorBoard logs
        logging_steps=100,                               # Log training metrics every 100 steps
        evaluation_strategy="epoch",                     # Evaluate model at the end of each epoch
        save_strategy="epoch",                           # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True,                     # Load the best model (based on metric_for_best_model) when training finishes
        metric_for_best_model="f1",                      # <--- CRITICAL: Use F1 score to determine the best model
        greater_is_better=True,                          # For F1, higher is better
        save_total_limit=1,                              # Only saves the best model, deleting older checkpoints
        seed=42,                                         # For reproducibility
        fp16=torch.cuda.is_available(),                  # Enable mixed precision training if GPU is available
    )

    # Initialize the custom Trainer with the model, arguments, datasets, metrics, weights, and callbacks
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,                     # Pass the calculated class weights to the custom trainer
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # <--- CRITICAL: Stop training if F1 doesn't improve for 3 epochs
    )

    # Start the training process
    print("Starting model training for VeritasNet...")
    trainer.train()
    print("Training complete.")

    # Save the final best model and its tokenizer
    # The best model is automatically loaded into `model` due to `load_best_model_at_end=True`
    # --- ✅ FINAL SAVE BLOCK (overwrite old one) ---
    output_model_dir = "models/narrative_bias/final_model"
    os.makedirs(output_model_dir, exist_ok=True)

    # Force saving with .bin instead of .safetensors
    model.save_pretrained(output_model_dir, safe_serialization=False)  # ✅ critical fix
    tokenizer.save_pretrained(output_model_dir)

    print(f"✅ Model and tokenizer saved to {output_model_dir} in .bin format.")