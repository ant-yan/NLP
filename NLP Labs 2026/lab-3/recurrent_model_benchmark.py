# ============================================================
# RNN vs LSTM Language Model Comparison
# Both architectures are trained on the same Frankenstein
# corpus; their loss curves and generated text are compared
# side-by-side to illustrate the long-range dependency gap.
# ============================================================

import re
import time
import torch
import torch.nn as nn
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from torch.utils.data import TensorDataset, DataLoader

# ---- Shared Hyperparameters ----
EMBED_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
BATCH_SIZE = 32
CONTEXT_WINDOW = 100
SEQ_LEN = CONTEXT_WINDOW - 1

# ---- Data Source ----
corpus_url = "https://www.gutenberg.org/cache/epub/84/pg84-images.html"
request_headers = {"User-Agent": "Mozilla/5.0"}

# ---- Evaluation Seeds ----
test_seeds = [
    "i will explore",
    "my dream was",
    "i felt a strange",
    "at night",
]


# ---- Network Architectures ----

class RecurrentNetwork(nn.Module):
    """
    Vanilla RNN that maps a token sequence to a next-word distribution.
    Uses only the last time-step output for prediction.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RecurrentNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        return self.fc(rnn_out[:, -1, :])


class LSTMNetwork(nn.Module):
    """
    LSTM variant of the same architecture; the gating mechanism
    helps the model retain relevant context over longer sequences.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ---- Training Utility ----

def fit_network(net, data_iter, epochs, runtime_device, tag):
    """
    Runs the full training loop for a given model, saves the
    best weights, and returns per-epoch loss values.
    """
    net.to(runtime_device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_history = []

    for epoch_num in range(epochs):
        net.train()
        running_loss = 0.0

        for batch_x, batch_y in data_iter:
            batch_x = batch_x.to(runtime_device)
            batch_y = batch_y.to(runtime_device)
            optimiser.zero_grad()
            out = net(batch_x)
            step_loss = loss_fn(out, batch_y)
            step_loss.backward()
            optimiser.step()
            running_loss += step_loss.item()

        epoch_avg = running_loss / len(data_iter)
        loss_history.append(epoch_avg)
        print("{} Epoch [{}/{}] Loss: {:.4f}".format(
            tag, epoch_num + 1, epochs, epoch_avg))

    torch.save(net.state_dict(), "{}_model.pth".format(tag))
    print("{} saved!\n".format(tag))
    return loss_history


# ---- Text Generation Utility ----

def continue_text(net, prompt, num_tokens, runtime_device):
    """
    Produces a word sequence following the given prompt using
    greedy decoding; halts at sentence boundary sentinel.
    """
    net.eval()
    tokens = prompt.lower().split()
    tokens = [t for t in tokens if t in word_index_map]

    if not tokens:
        return "Seed words not found."

    input_seq = [word_index_map[t] for t in tokens]

    for _ in range(num_tokens):
        if len(input_seq) < SEQ_LEN:
            context = [0] * (SEQ_LEN - len(input_seq)) + input_seq
        else:
            context = input_seq[-SEQ_LEN:]

        x_tensor = torch.tensor([context], dtype=torch.long).to(runtime_device)

        with torch.no_grad():
            pred_idx = torch.argmax(net(x_tensor), dim=1).item()
            next_word = index_word_map[pred_idx]

        tokens.append("." if next_word == "<END>" else next_word)
        input_seq.append(pred_idx)

    return " ".join(tokens)


# ---- Corpus Download & Parse ----

server_response = requests.get(corpus_url, headers=request_headers)
html_soup = BeautifulSoup(server_response.text, "html.parser")
document = html_soup.get_text()

book_start = "To Mrs. Saville, England."
book_end = (
    "*** END OF THE PROJECT GUTENBERG EBOOK "
    "FRANKENSTEIN; OR, THE MODERN PROMETHEUS ***"
)

s_pos = document.find(book_start)
e_pos = document.find(book_end)
if s_pos != -1:
    document = document[s_pos + len(book_start):]
if e_pos != -1:
    document = document[:e_pos]

# ---- Text Cleaning ----

document = re.sub(
    r"\n\s*(letter|chapter)\s+\d+\s*\n", "\n", document, flags=re.IGNORECASE
)
document = re.sub(
    r"(^|\n)[A-Za-z.\- ]*,?\s*"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\,?\s*(\d{1,2}(st|nd|rd|th)?)?,?\s*(17—\.)?\s*",
    r"\1",
    document,
)
document = document.lower()
document = re.sub(r"\s+", " ", document).strip()
document = re.sub(r"([.!?])", r"\1 <END>", document)
document = re.sub(r"[^\w\s<>]", "", document)

# ---- Vocabulary Construction ----

word_tokens = document.split()
print("Total tokens:", len(word_tokens))

sorted_vocab = sorted(set(word_tokens))
word_index_map = {word: idx for idx, word in enumerate(sorted_vocab)}
index_word_map = {idx: word for word, idx in word_index_map.items()}
vocab_size = len(sorted_vocab)
print("Vocab size:", vocab_size)

numeric_seq = [word_index_map[w] for w in word_tokens]

# ---- Sliding Window Dataset ----

input_windows, target_words = [], []
for i in range(len(numeric_seq) - CONTEXT_WINDOW):
    input_windows.append(numeric_seq[i: i + SEQ_LEN])
    target_words.append(numeric_seq[i + SEQ_LEN])

x_tensor = torch.tensor(input_windows, dtype=torch.long)
y_tensor = torch.tensor(target_words, dtype=torch.long)
print(x_tensor.shape, y_tensor.shape)

corpus_dataset = TensorDataset(x_tensor, y_tensor)
corpus_loader = DataLoader(corpus_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Device & Model Setup ----

runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vanilla_rnn = RecurrentNetwork(vocab_size, EMBED_DIM, HIDDEN_DIM)
memory_rnn = LSTMNetwork(vocab_size, EMBED_DIM, HIDDEN_DIM)

# ---- Sequential Training ----

rnn_loss_history = fit_network(vanilla_rnn, corpus_loader, NUM_EPOCHS, runtime_device, "RNN")
lstm_loss_history = fit_network(memory_rnn, corpus_loader, NUM_EPOCHS, runtime_device, "LSTM")

# ---- Loss Curve Visualization ----

plt.figure()
plt.plot(rnn_loss_history, label="RNN")
plt.plot(lstm_loss_history, label="LSTM")
plt.title("RNN vs LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Comparative Generation ----

print("\nTEXT GENERATION COMPARISON\n")
for prompt in test_seeds:
    rnn_output = continue_text(vanilla_rnn, prompt, 50, runtime_device)
    lstm_output = continue_text(memory_rnn, prompt, 50, runtime_device)
    print("Seed: {}".format(prompt))
    print("RNN :", rnn_output)
    print("LSTM:", lstm_output)
    print("-" * 50)
