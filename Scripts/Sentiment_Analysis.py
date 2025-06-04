import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Dataset
df = pd.read_csv(
    r'C:/Users/anils/OneDrive/Documents/MlOps/mlops_project/Data/raw/Meta_Hinglish_annotated.csv',
    names=['text', 'label'],
    skiprows=1,  # Skip header
    encoding='utf-8'
)

# Clean and inspect labels
df['label'] = df['label'].astype(str).str.strip()
print("Unique labels before mapping:", df['label'].unique())

# Define label mapping
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
label_map_reverse = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Apply mapping and drop invalid rows
df['label'] = df['label'].map(label_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"✅ Total cleaned samples: {len(df)}")
print(df.head())

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-mlm-only")

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

# Create datasets and loaders
train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "ai4bharat/IndicBERTv2-mlm-only",
    num_labels=3
).to(device)

# Optimizer & scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_loader) * 3
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training function
def train_model(model, train_loader):
    model.train()
    for epoch in range(3):
        print(f"--- Epoch {epoch+1} ---")
        loop = tqdm(train_loader)
        for batch in loop:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            loop.set_postfix(loss=loss.item())

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Neutral', 'Positive']))

# Predict function
def predict_sentiment(sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
    return label_map_reverse[prediction]

# Run training and evaluation
train_model(model, train_loader)
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







