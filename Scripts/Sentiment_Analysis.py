# Sentiment_Analysis.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# Label mapping
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
label_map_reverse = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ✅ Singleton pattern: cache model/tokenizer
_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        model_id = "albert-base-v2"
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3).to(device)
        _model.eval()
    return _tokenizer, _model

# ✅ Predict a single sentence
def predict_sentiment(sentence):
    tokenizer, model = load_model()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return label_map_reverse[prediction]

# ✅ Only load dataset and evaluate if script is run directly
def evaluate(model, test_loader):
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    print(f"\n🎯 Accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Neutral', 'Positive']))

# ✅ Run full evaluation only when directly executed
if __name__ == "__main__":
    # Load and clean dataset
    df = pd.read_csv(
        r'C:/Users/anils/Documents - Copy/MlOps/Sentiment-Analysis/Data/raw/tweet_eval_sentiment.csv',
        names=['text', 'label'],
        skiprows=1,
        encoding='utf-8'
    )
    df['label'] = df['label'].astype(str).str.strip().map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 🔽 Reduce dataset size to 2000 samples
    df = df.iloc[:1000].reset_index(drop=True)

    test_texts = df['text'].tolist()
    test_labels = df['label'].tolist()

    # Load model/tokenizer
    tokenizer, model = load_model()

    # Tokenize
    encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    class SentimentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    test_dataset = SentimentDataset(encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Evaluate
    evaluate(model, test_loader)
