import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load and clean dataset
df = pd.read_csv(
    r'Data/raw/Meta_Hinglish_annotated.csv',
    names=['text', 'label'],
    skiprows=1,
    encoding='utf-8'
)

# Label mapping
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

# Load IndicBERTv2-tiny
model_id = "ai4bharat/IndicBERTv2-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3).to(device)
model.eval()

# Tokenization
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

# DataLoader with batch size = 1
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
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Neutral', 'Positive']))

# Single sentence prediction
def predict_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return label_map_reverse[prediction]

# Run evaluation
evaluate(model, test_loader)

# Sample predictions
print("\n--- Sample Predictions ---")
test_sentences = [
    "Yeh movie bahut acchi thi!",
    "Mujhe yeh jagah bilkul pasand nahi aayi.",
    "Main kal shopping gaya tha.",
    "vaha bahut kharab admi hai",
    "apple mujhe pasand hai",
    "kal milte hai fir",
    "tu chutya hye",
    "tu bhekar hye"
]

for sentence in test_sentences:
    print(f"Sentence: {sentence} -> Sentiment: {predict_sentiment(sentence)}")

