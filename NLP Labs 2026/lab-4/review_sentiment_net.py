# ============================================================
# IMDB Sentiment Analysis — RNN / LSTM / GRU Benchmark
# Loads the IMDB review dataset, builds a frequency-capped
# vocabulary, and trains three recurrent architectures in
# sequence before reporting precision, recall, F1, and
# confusion matrices for each.
# ============================================================

import re
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import TensorDataset, DataLoader

# ---- Global Configuration ----
VOCAB_LIMIT = 10000
EMBED_DIM = 100
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
PAD_LEN = 256

# ---- Data Loading ----

sentiment_df = pd.read_csv("IMDB Dataset.csv")
sentiment_df["sentiment"] = sentiment_df["sentiment"].apply(
    lambda label: 1 if label == "positive" else 0
)

raw_reviews, _, raw_labels, _ = [None] * 4  # placeholder alignment
raw_reviews = sentiment_df["review"]
raw_labels = sentiment_df["sentiment"]

train_reviews, test_reviews, train_labels, test_labels = train_test_split(
    raw_reviews, raw_labels, test_size=0.2, random_state=42
)


# ---- Tokenisation & Vocabulary ----

def word_tokenize(raw_text):
    """Lowercases, strips punctuation, and drops tokens shorter than 3 chars."""
    lowered = raw_text.lower()
    tokens = re.findall(r"\b\w+\b", lowered)
    return [t for t in tokens if len(t) > 2]


vocab_counter = Counter()
for review_text in train_reviews:
    vocab_counter.update(word_tokenize(review_text))

frequent_words = vocab_counter.most_common(VOCAB_LIMIT - 2)
word_map = {word: idx + 2 for idx, (word, _) in enumerate(frequent_words)}
word_map["<PAD>"] = 0
word_map["<OOV>"] = 1


def encode_review(raw_text):
    """Converts a review string into a list of integer token IDs."""
    tokens = word_tokenize(raw_text)
    return [word_map.get(tok, word_map["<OOV>"]) for tok in tokens]


def normalize_length(seq):
    """Clips or right-pads a sequence to exactly PAD_LEN tokens."""
    if len(seq) > PAD_LEN:
        return seq[:PAD_LEN]
    return seq + [0] * (PAD_LEN - len(seq))


train_enc = [encode_review(r) for r in train_reviews]
test_enc = [encode_review(r) for r in test_reviews]

x_train_pad = [normalize_length(s) for s in train_enc]
x_test_pad = [normalize_length(s) for s in test_enc]

x_train = torch.tensor(x_train_pad, dtype=torch.long)
y_train = torch.tensor(train_labels.values, dtype=torch.float)
x_test = torch.tensor(x_test_pad, dtype=torch.long)
y_test = torch.tensor(test_labels.values, dtype=torch.float)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# ---- Network Architectures ----

class BasicRNN(nn.Module):
    """Single-layer vanilla RNN sentiment classifier."""

    def __init__(self, vocab_sz, embed_dim, hidden_dim):
        super(BasicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.sigmoid(self.fc(out[:, -1, :]))


class MemoryNet(nn.Module):
    """LSTM-based classifier that handles long-range review context."""

    def __init__(self, vocab_sz, embed_dim, hidden_dim):
        super(MemoryNet, self).__init__()
        self.embedding = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.sigmoid(self.fc(out[:, -1, :]))


class GatedRNN(nn.Module):
    """GRU variant; fewer parameters than LSTM with comparable capacity."""

    def __init__(self, vocab_sz, embed_dim, hidden_dim):
        super(GatedRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_sz, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        return self.sigmoid(self.fc(out[:, -1, :]))


# ---- Evaluation Routine ----

def compute_metrics(net, data_iter, device):
    """
    Runs inference on the full test split and returns the
    standard binary classification metrics along with the
    confusion matrix.
    """
    net.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in data_iter:
            x_batch = x_batch.long().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            preds = (net(x_batch) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prec, rec, f1, cm


# ---- Training Routine ----

def run_training(net, train_iter, test_iter, lr=LEARNING_RATE, epochs=NUM_EPOCHS):
    """
    Optimises a binary classification model with BCE loss and
    returns performance metrics evaluated on the test split.
    """
    net.to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    bce_loss = nn.BCELoss()

    for ep in range(epochs):
        net.train()
        total_loss = 0.0

        for x_batch, y_batch in train_iter:
            x_batch = x_batch.long().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            optimiser.zero_grad()
            step_loss = bce_loss(net(x_batch), y_batch)
            step_loss.backward()
            optimiser.step()
            total_loss += step_loss.item()

        avg_loss = total_loss / len(train_iter)
        print("Epoch {}/{} - Loss: {:.4f}".format(ep + 1, epochs, avg_loss))

    return compute_metrics(net, test_iter, device)


# ---- Device & Model Registry ----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_zoo = {
    "RNN": BasicRNN(VOCAB_LIMIT, EMBED_DIM, HIDDEN_DIM).to(device),
    "LSTM": MemoryNet(VOCAB_LIMIT, EMBED_DIM, HIDDEN_DIM).to(device),
    "GRU": GatedRNN(VOCAB_LIMIT, EMBED_DIM, HIDDEN_DIM).to(device),
}

score_table = {}

# ---- Sequential Training & Evaluation ----

for arch_name, net in model_zoo.items():
    print("\nTraining {} on {}".format(arch_name, device))
    acc, prec, rec, f1, cm = run_training(
        net, train_loader, test_loader, epochs=NUM_EPOCHS
    )
    score_table[arch_name] = (acc, prec, rec, f1, cm)

# ---- Results Summary ----

for arch_name, (acc, prec, rec, f1, cm) in score_table.items():
    print("\n{} Results".format(arch_name))
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", cm)
