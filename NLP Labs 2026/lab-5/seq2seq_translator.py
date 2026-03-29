# ============================================================
# English → Russian Neural Machine Translation (Seq2Seq)
# Implements an encoder-decoder architecture with LSTM cells.
# Beam search decoding is used at inference time to improve
# translation quality over greedy argmax selection.
# ============================================================

import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

# ---- Hyperparameters ----
BATCH_SIZE = 32
EMBED_DIM = 256
HIDDEN_DIM = 1024
N_LAYERS = 1
EPOCHS = 10
GRAD_CLIP = 1
BEAM_WIDTH = 5

# ---- Source Data ----
source_file = "rus.txt"

# ---- Demo Sentences ----
demo_sentences = [
    "hello how are you",
    "i love programming",
    "where is the cinema",
    "this is a good idea",
    "please show me the path",
    "can you help me",
]


# ---- Preprocessing Utilities ----

def sanitize_text(raw):
    """Strips non-alphabetic characters and normalises whitespace."""
    lowered = raw.lower()
    stripped = re.sub(r"[^a-zа-яё\s]", "", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def make_vocabulary(sentence_list):
    """
    Builds a word-to-index mapping from a list of whitespace-
    tokenised strings; reserves slots 0 and 1 for padding and OOV.
    """
    freq = Counter()
    for sent in sentence_list:
        freq.update(sent.split())
    vocab = {"<pad>": 0, "<oov>": 1}
    for word in freq:
        vocab[word] = len(vocab)
    return vocab


def encode_sentence(sentence, vocab):
    """Maps each token in sentence to its integer index."""
    tokens = sentence.split() if isinstance(sentence, str) else sentence
    return [vocab.get(tok, vocab["<oov>"]) for tok in tokens]


# ---- Dataset & Collation ----

class ParallelCorpus(Dataset):
    """Stores aligned source-target integer sequence pairs."""

    def __init__(self, pair_list):
        self.pairs = pair_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return torch.tensor(src), torch.tensor(tgt)


def pad_batch(batch):
    """Pads variable-length sequences within a batch to a common length."""
    src_seqs, tgt_seqs = zip(*batch)
    return (
        pad_sequence(src_seqs, padding_value=0),
        pad_sequence(tgt_seqs, padding_value=0),
    )


# ---- Model Components ----

class SourceEncoder(nn.Module):
    """
    Encodes a source sequence into a fixed-size context vector
    represented by the LSTM's final hidden and cell states.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers)

    def forward(self, src):
        embedded = self.embedding(src)
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class TargetDecoder(nn.Module):
    """
    Generates target tokens one step at a time, conditioned on
    the encoder context passed through hidden and cell states.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, token, hidden, cell):
        token = token.unsqueeze(0)
        emb = self.embedding(token)
        out, (hidden, cell) = self.lstm(emb, (hidden, cell))
        prediction = self.fc(out.squeeze(0))
        return prediction, hidden, cell


class NeuralTranslator(nn.Module):
    """
    Connects encoder and decoder; applies teacher forcing during
    training to stabilise gradient flow across the decoder.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_sz = trg.shape[1]
        trg_steps = trg.shape[0]
        tgt_vocab_sz = self.decoder.output_dim

        step_outputs = torch.zeros(trg_steps, batch_sz, tgt_vocab_sz).to(self.device)
        hidden, cell = self.encoder(src)
        current_token = trg[0, :]

        for t in range(1, trg_steps):
            pred, hidden, cell = self.decoder(current_token, hidden, cell)
            step_outputs[t] = pred
            top_token = pred.argmax(1)
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            current_token = trg[t] if use_teacher else top_token

        return step_outputs


# ---- Training & Validation Functions ----

def train_epoch(net, loader, optimiser, loss_fn, clip_val, device):
    """One full pass over the training data with gradient clipping."""
    net.train()
    total_loss = 0.0

    for src_batch, trg_batch in loader:
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)
        optimiser.zero_grad()

        preds = net(src_batch, trg_batch)
        vocab_sz = preds.shape[-1]
        flat_preds = preds[1:].view(-1, vocab_sz)
        flat_trg = trg_batch[1:].reshape(-1)

        loss = loss_fn(flat_preds, flat_trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
        optimiser.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(net, loader, loss_fn, device):
    """One full pass over the validation data without gradient updates."""
    net.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src_batch, trg_batch in loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)
            preds = net(src_batch, trg_batch, teacher_forcing_ratio=0)
            vocab_sz = preds.shape[-1]
            flat_preds = preds[1:].view(-1, vocab_sz)
            flat_trg = trg_batch[1:].reshape(-1)
            total_loss += loss_fn(flat_preds, flat_trg).item()

    return total_loss / len(loader)


# ---- Beam Search Decoder ----

def beam_translate(sentence, net, src_vocab, tgt_vocab, device,
                   beam_size=5, max_len=50):
    """
    Translates a sentence using beam search; accumulates log-probabilities
    across candidate sequences and returns the highest-scoring path.
    """
    net.eval()
    tokens = sentence.lower().split() if isinstance(sentence, str) else sentence
    src_ids = [src_vocab.get(t, src_vocab["<oov>"]) for t in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = net.encoder(src_tensor)

    beams = [([tgt_vocab["<sos>"]], 0.0, hidden, cell)]

    for _ in range(max_len):
        new_beams = []
        for seq, score, h, c in beams:
            last = seq[-1]
            if last == tgt_vocab["<eos>"]:
                new_beams.append((seq, score, h, c))
                continue

            tok_tensor = torch.LongTensor([last]).to(device)
            with torch.no_grad():
                out, h_new, c_new = net.decoder(tok_tensor, h, c)

            log_probs = F.log_softmax(out, dim=1)
            top_probs, top_idxs = log_probs.topk(beam_size)

            for k in range(beam_size):
                token = top_idxs[0][k].item()
                new_beams.append((
                    seq + [token],
                    score + top_probs[0][k].item(),
                    h_new, c_new,
                ))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    best_seq = beams[0][0]
    inv_vocab = {v: k for k, v in tgt_vocab.items()}
    return [inv_vocab[i] for i in best_seq][1:]


# ---- Data Loading ----

raw_pairs = []
with open(source_file, "r", encoding="utf-8") as fh:
    for line in fh:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            raw_pairs.append((parts[0], parts[1]))

print("Total pairs:", len(raw_pairs))

sanitized_pairs = [(sanitize_text(e), sanitize_text(r)) for e, r in raw_pairs]

prepared_pairs = [
    (eng, "<sos> {} <eos>".format(rus)) for eng, rus in sanitized_pairs
]

english_sentences = [p[0] for p in prepared_pairs]
russian_sentences = [p[1] for p in prepared_pairs]

english_vocab = make_vocabulary(english_sentences)
russian_vocab = make_vocabulary(russian_sentences)
print("English vocabulary size:", len(english_vocab))
print("Russian vocabulary size:", len(russian_vocab))

indexed_pairs = [
    (encode_sentence(e, english_vocab), encode_sentence(r, russian_vocab))
    for e, r in prepared_pairs
]

train_split, val_split = train_test_split(indexed_pairs, test_size=0.2, random_state=42)
print("Train size:", len(train_split), "| Val size:", len(val_split))

train_corpus = ParallelCorpus(train_split)
val_corpus = ParallelCorpus(val_split)

train_batches = DataLoader(
    train_corpus, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch
)
val_batches = DataLoader(
    val_corpus, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_batch
)

# ---- Model Instantiation ----

src_vocab_size = len(english_vocab)
tgt_vocab_size = len(russian_vocab)

src_encoder = SourceEncoder(src_vocab_size, EMBED_DIM, HIDDEN_DIM, N_LAYERS)
tgt_decoder = TargetDecoder(tgt_vocab_size, EMBED_DIM, HIDDEN_DIM, N_LAYERS)

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator = NeuralTranslator(src_encoder, tgt_decoder, compute_device).to(compute_device)

optim_fn = optim.Adam(translator.parameters(), lr=0.001)
padding_token = russian_vocab["<pad>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=padding_token)

# ---- Training Loop ----

best_score = float("inf")

for epoch_num in range(EPOCHS):
    t_start = time.time()
    tr_loss = train_epoch(translator, train_batches, optim_fn, loss_fn, GRAD_CLIP, compute_device)
    vl_loss = eval_epoch(translator, val_batches, loss_fn, compute_device)
    t_end = time.time()

    if vl_loss < best_score:
        best_score = vl_loss
        torch.save(translator.state_dict(), "best_model_{}.pt".format(epoch_num))

    print("Epoch {}".format(epoch_num + 1))
    print("Train loss:", tr_loss)
    print("Val loss:", vl_loss)

# ---- Inference Demo ----

for sent in demo_sentences:
    result = beam_translate(
        sent, translator, english_vocab, russian_vocab, compute_device, beam_size=BEAM_WIDTH
    )
    print("English:", sent)
    print("Russian:", " ".join(result))
    print()
