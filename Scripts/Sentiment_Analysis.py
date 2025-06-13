import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# Load and clean dataset
df = pd.read_csv(
    r'Data/raw/Meta_Hinglish_annotated.csv',
    names=['text', 'label'],
    skiprows=1,
    encoding='utf-8'
)

df['label'] = df['label'].astype(str).str.strip()
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
label_map_reverse = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['label'] = df['label'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Shuffle dataset
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
test_texts = df['text'].tolist()
test_labels = df['label'].tolist()

# Load model and tokenizer
model_id = "prajjwal1/bert-mini"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3).to(device)
model.eval()

# Tokenize inputs
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

# DataLoader
test_dataset = SentimentDataset(encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=1)

# Evaluation
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

# Predict a single sentence
def predict_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return label_map_reverse[prediction]

# Evaluate
evaluate(model, test_loader)

# Sample predictions
print("\n--- Sample Predictions ---")
samples = [
    "Yeh movie bahut acchi thi!",
    "Mujhe yeh jagah bilkul pasand nahi aayi.",
    "Main kal shopping gaya tha.",
    "vaha bahut kharab admi hai",
    "apple mujhe pasand hai",
    "kal milte hai fir",
    "tu chutya hye",
    "tu bhekar hye"
]

for s in samples:
    print(f"📌 {s} → {predict_sentiment(s)}")
