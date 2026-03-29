# ============================================================
# Character-Level RNN Language Model (Word Granularity)
# Trains a vanilla recurrent network on Frankenstein scraped
# from Project Gutenberg, then generates continuations from
# user-supplied seed phrases using greedy argmax decoding.
# ============================================================

import re
import torch
import torch.nn as nn
import requests
from bs4 import BeautifulSoup
from torch.utils.data import TensorDataset, DataLoader

# ---- Configuration ----
EMBED_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
EPOCH_COUNT = 30
BATCH_SIZE = 64
CONTEXT_WINDOW = 100
SEQ_LEN = CONTEXT_WINDOW - 1

# ---- Corpus Source ----
corpus_url = "https://www.gutenberg.org/cache/epub/84/pg84-images.html"
request_headers = {"User-Agent": "Mozilla/5.0"}

# ---- Seed Phrases for Generation ----
seed_phrases = [
    "i will explore",
    "i world like to know",
    "my dream was",
    "i felt a strange",
    "at night",
]


# ---- Model Definition ----

class SimpleRNN(nn.Module):
    """
    Single-layer vanilla RNN that reads a sequence of embedded tokens
    and predicts the next word from the final hidden state.
    """

    def __init__(self, vocabulary_size, embed_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        rnn_out, _ = self.rnn(embedded)
        final_hidden = rnn_out[:, -1, :]
        return self.fc(final_hidden)


# ---- Generation Utility ----

def generate_continuation(net, prompt, num_words, runtime_device):
    """
    Extends a seed phrase word-by-word using greedy decoding.
    Stops early if the <END> sentinel is encountered.
    """
    net.eval()

    word_list = prompt.lower().split()
    word_list = [w for w in word_list if w in word_index_map]

    if not word_list:
        return "No seed words found in vocabulary."

    input_seq = [word_index_map[w] for w in word_list]

    for _ in range(num_words):
        if len(input_seq) < SEQ_LEN:
            padded = [0] * (SEQ_LEN - len(input_seq)) + input_seq
        else:
            padded = input_seq[-SEQ_LEN:]

        step_tensor = torch.tensor([padded], dtype=torch.long).to(runtime_device)

        with torch.no_grad():
            logits = net(step_tensor)
            next_idx = torch.argmax(logits, dim=1).item()
            next_word = index_word_map[next_idx]

        if next_word == "<END>":
            word_list.append(".")
            break

        word_list.append(next_word)
        input_seq.append(next_idx)

    return " ".join(word_list)


# ---- Data Acquisition ----

server_response = requests.get(corpus_url, headers=request_headers)
html_parser = BeautifulSoup(server_response.text, "html.parser")
document = html_parser.get_text()

corpus_begin = "To Mrs. Saville, England."
corpus_end = (
    "*** END OF THE PROJECT GUTENBERG EBOOK "
    "FRANKENSTEIN; OR, THE MODERN PROMETHEUS ***"
)

start_pos = document.find(corpus_begin)
end_pos = document.find(corpus_end)

if start_pos != -1:
    document = document[start_pos + len(corpus_begin):]
if end_pos != -1:
    document = document[:end_pos]

# ---- Text Preprocessing ----

document = re.sub(
    r"\n\s*(letter|chapter)\s+\d+\s*\n", "\n", document, flags=re.IGNORECASE
)
document = re.sub(
    r"(^|\n)[A-Za-z.\- ]*,?\s*"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\,?\s*(\d{{1,2}}(st|nd|rd|th)?)?,?\s*(17—\.)?\s*",
    r"\1",
    document,
)
document = document.lower()
document = re.sub(r"\s+", " ", document).strip()

document = re.sub(r"([.!?])", r"\1 <END>", document)
document = re.sub(r"[^\w\s<>]", "", document)

# ---- Tokenization & Vocabulary ----

word_tokens = document.split()
print("Total tokens:", len(word_tokens))

sorted_vocab = sorted(set(word_tokens))
word_index_map = {word: idx for idx, word in enumerate(sorted_vocab)}
index_word_map = {idx: word for word, idx in word_index_map.items()}
vocabulary_size = len(sorted_vocab)
print("Vocab size:", vocabulary_size)

numeric_seq = [word_index_map[w] for w in word_tokens]

# ---- Sliding Window Sequences ----

sequence_inputs = []
next_word_targets = []

for i in range(len(numeric_seq) - CONTEXT_WINDOW):
    sequence_inputs.append(numeric_seq[i: i + SEQ_LEN])
    next_word_targets.append(numeric_seq[i + SEQ_LEN])

print("Sequence count:", len(sequence_inputs))

# ---- Tensor Construction & DataLoader ----

input_tensor = torch.tensor(sequence_inputs, dtype=torch.long)
target_tensor = torch.tensor(next_word_targets, dtype=torch.long)

seq_dataset = TensorDataset(input_tensor, target_tensor)
seq_loader = DataLoader(seq_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- Training Setup ----

runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", runtime_device)

rnn_net = SimpleRNN(vocabulary_size, EMBED_DIM, HIDDEN_DIM).to(runtime_device)
cross_entropy_loss = nn.CrossEntropyLoss()
adam_opt = torch.optim.Adam(rnn_net.parameters(), lr=LEARNING_RATE)

# ---- Training Loop ----

for epoch_idx in range(EPOCH_COUNT):
    rnn_net.train()
    epoch_loss = 0.0

    for batch_x, batch_y in seq_loader:
        batch_x = batch_x.to(runtime_device)
        batch_y = batch_y.to(runtime_device)

        adam_opt.zero_grad()
        logits = rnn_net(batch_x)
        loss = cross_entropy_loss(logits, batch_y)
        loss.backward()
        adam_opt.step()
        epoch_loss += loss.item()

    avg = epoch_loss / len(seq_loader)
    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch_idx + 1, EPOCH_COUNT, avg))

# ---- Text Generation ----

for idx, phrase in enumerate(seed_phrases):
    generated = generate_continuation(rnn_net, phrase, length=10, runtime_device=runtime_device)
    print("Sentence {}:".format(idx + 1), generated)
