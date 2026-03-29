# ============================================================
# Seq2Seq Translation with BLEU Evaluation & Decoding Study
# Extends the baseline translator with a from-scratch BLEU
# implementation and a runtime comparison of greedy decoding
# vs beam search at widths 3, 5, and 10.
# ============================================================

import re
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

# ---- Configuration ----
BATCH_SIZE = 32
EMBED_DIM = 256
HIDDEN_DIM = 1024
N_LAYERS = 1
EPOCHS = 10
GRAD_CLIP = 1

# ---- Data Path ----
source_file = "rus.txt"

# ---- Demo Sentences ----
demo_sentences = [
    "hello how are you",
    "i love programming",
    "where is the cinema",
    "this is a good idea",
    "please show me the path",
    "can you help me",
    "this is interesting",
]


# ---- Text Preprocessing ----

def sanitize_text(raw):
    """Normalises a string to lowercase ASCII/Cyrillic, strips punctuation."""
    lowered = raw.lower()
    cleaned = re.sub(r"[^a-zа-яё\s]", "", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def make_vocabulary(sentences):
    """Constructs a token-to-index map from a list of sentences."""
    freq = Counter()
    for sent in sentences:
        freq.update(sent.split())
    vocab = {"<pad>": 0, "<oov>": 1}
    for token in freq:
        vocab[token] = len(vocab)
    return vocab


def encode_sentence(sentence, vocab):
    """Converts a sentence string or token list to integer indices."""
    tokens = sentence.split() if isinstance(sentence, str) else sentence
    return [vocab.get(t, vocab["<oov>"]) for t in tokens]


# ---- BLEU Metric Implementation ----

def count_ngrams(token_list, n):
    """Returns a frequency Counter of all n-grams in token_list."""
    return Counter(
        tuple(token_list[i: i + n]) for i in range(len(token_list) - n + 1)
    )


def clipped_prec(reference, hypothesis, n):
    """
    Computes modified n-gram precision: hypothesis counts are clipped
    to the reference maximum before dividing by total hypothesis n-grams.
    """
    ref_ng = count_ngrams(reference, n)
    hyp_ng = count_ngrams(hypothesis, n)
    total = sum(hyp_ng.values())
    if total == 0:
        return 0
    clipped = sum(min(cnt, ref_ng.get(ng, 0)) for ng, cnt in hyp_ng.items())
    return clipped / total


def length_penalty(reference, hypothesis):
    """
    Applies a brevity penalty when the hypothesis is shorter than
    the reference; returns 1.0 when hypothesis length is sufficient.
    """
    r_len = len(reference)
    h_len = len(hypothesis)
    if h_len == 0:
        return 0
    return 1 if h_len > r_len else math.exp(1 - r_len / h_len)


def compute_bleu(reference, hypothesis, max_n=4):
    """
    Full BLEU calculation with uniform n-gram weights up to max_n.
    Returns 0 if any precision is zero (avoids log(0)).
    """
    weights = [1 / max_n] * max_n
    precisions = [clipped_prec(reference, hypothesis, n) for n in range(1, max_n + 1)]
    if min(precisions) == 0:
        return 0
    log_avg = sum(w * math.log(p) for w, p in zip(weights, precisions))
    return length_penalty(reference, hypothesis) * math.exp(log_avg)


# ---- Vocabulary Decoding Helpers ----

def ids_to_words(indices, vocab):
    """Converts integer indices back to tokens, skipping special symbols."""
    inv = {v: k for k, v in vocab.items()}
    result = []
    for i in indices:
        word = inv.get(i, "<oov>")
        if word == "<eos>":
            break
        if word not in {"<sos>", "<pad>"}:
            result.append(word)
    return result


# ---- Dataset ----

class ParallelCorpus(Dataset):
    """Wraps aligned source/target index pairs for batch iteration."""

    def __init__(self, pair_list):
        self.pairs = pair_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        return torch.tensor(s), torch.tensor(t)


def pad_batch(batch):
    """Pads a batch of variable-length sequence pairs."""
    src_seqs, tgt_seqs = zip(*batch)
    return (
        pad_sequence(src_seqs, padding_value=0),
        pad_sequence(tgt_seqs, padding_value=0),
    )


# ---- Model Architecture ----

class SourceEncoder(nn.Module):
    """LSTM encoder that compresses the source sequence to hidden states."""

    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers)

    def forward(self, src):
        emb = self.embedding(src)
        _, (hidden, cell) = self.lstm(emb)
        return hidden, cell


class TargetDecoder(nn.Module):
    """LSTM decoder that generates target tokens from encoder context."""

    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, token, hidden, cell):
        emb = self.embedding(token.unsqueeze(0))
        out, (hidden, cell) = self.lstm(emb, (hidden, cell))
        return self.fc(out.squeeze(0)), hidden, cell


class NeuralTranslator(nn.Module):
    """Full Seq2Seq model with optional teacher forcing."""

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


# ---- Training & Evaluation ----

def train_epoch(net, loader, optimiser, loss_fn, clip, device):
    """Single training epoch with gradient clipping."""
    net.train()
    total_loss = 0.0
    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        optimiser.zero_grad()
        preds = net(src, trg)
        flat_preds = preds[1:].view(-1, preds.shape[-1])
        flat_trg = trg[1:].reshape(-1)
        loss = loss_fn(flat_preds, flat_trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(net, loader, loss_fn, device):
    """Validation pass without parameter updates."""
    net.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            preds = net(src, trg, teacher_forcing_ratio=0)
            flat_preds = preds[1:].view(-1, preds.shape[-1])
            flat_trg = trg[1:].reshape(-1)
            total_loss += loss_fn(flat_preds, flat_trg).item()
    return total_loss / len(loader)


# ---- Beam Decoding ----

def decode_beam(net, sentence, src_vocab, tgt_vocab, device,
                beam_width=5, max_len=50):
    """
    Beam search translation; completed hypotheses (ending with <eos>)
    are collected separately and compete with active beams at the end.
    """
    net.eval()
    tokens = sentence.lower().split() if isinstance(sentence, str) else sentence
    src_ids = [src_vocab.get(t, src_vocab["<oov>"]) for t in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = net.encoder(src_tensor)

    beams = [([tgt_vocab["<sos>"]], 0.0, hidden, cell)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for seq, score, h, c in beams:
            last = seq[-1]
            if last == tgt_vocab["<eos>"]:
                completed.append((seq, score))
                continue
            tok_tensor = torch.LongTensor([last]).to(device)
            with torch.no_grad():
                out, h_new, c_new = net.decoder(tok_tensor, h, c)
            log_probs = F.log_softmax(out, dim=1)
            top_probs, top_idxs = log_probs.topk(beam_width)
            for k in range(beam_width):
                new_beams.append((
                    seq + [top_idxs[0][k].item()],
                    score + top_probs[0][k].item(),
                    h_new, c_new,
                ))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    completed += [(seq, score) for seq, score, _, _ in beams]
    best = sorted(completed, key=lambda x: x[1], reverse=True)[0][0]
    inv_vocab = {v: k for k, v in tgt_vocab.items()}

    result = []
    for i in best:
        word = inv_vocab[i]
        if word == "<eos>":
            break
        if word not in {"<sos>", "<pad>"}:
            result.append(word)
    return result


def decode_greedy(sentence):
    return decode_beam(translator, sentence, english_vocab, russian_vocab,
                       compute_device, beam_width=1)


def decode_w3(sentence):
    return decode_beam(translator, sentence, english_vocab, russian_vocab,
                       compute_device, beam_width=3)


def decode_w5(sentence):
    return decode_beam(translator, sentence, english_vocab, russian_vocab,
                       compute_device, beam_width=5)


def decode_w10(sentence):
    return decode_beam(translator, sentence, english_vocab, russian_vocab,
                       compute_device, beam_width=10)


# ---- BLEU Evaluation Utilities ----

def bleu_evaluation(net, eval_data, src_vocab, tgt_vocab, device, decode_fn):
    """Computes average BLEU-1, BLEU-2, and BLEU-4 over a data subset."""
    scores_1, scores_2, scores_4 = [], [], []
    for src_ids, tgt_ids in eval_data:
        src_sent = ids_to_words(src_ids, src_vocab)
        ref_sent = ids_to_words(tgt_ids, tgt_vocab)
        hyp_tokens = decode_fn(" ".join(src_sent))
        scores_1.append(compute_bleu(ref_sent, hyp_tokens, max_n=1))
        scores_2.append(compute_bleu(ref_sent, hyp_tokens, max_n=2))
        scores_4.append(compute_bleu(ref_sent, hyp_tokens, max_n=4))
    n = len(scores_1)
    return sum(scores_1) / n, sum(scores_2) / n, sum(scores_4) / n


def timed_inference(decode_fn, data, n_samples=100):
    """Measures average decoding time per sentence over n_samples."""
    t0 = time.time()
    for i in range(n_samples):
        src, _ = data[i]
        decode_fn(" ".join(ids_to_words(src, english_vocab)))
    return (time.time() - t0) / n_samples


# ---- Data Loading ----

raw_pairs = []
with open(source_file, "r", encoding="utf-8") as fh:
    for line in fh:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            raw_pairs.append((parts[0], parts[1]))

sanitized_pairs = [(sanitize_text(e), sanitize_text(r)) for e, r in raw_pairs]
prepared_pairs = [
    (eng, "<sos> {} <eos>".format(rus)) for eng, rus in sanitized_pairs
]

english_sentences = [p[0] for p in prepared_pairs]
russian_sentences = [p[1] for p in prepared_pairs]

english_vocab = make_vocabulary(english_sentences)
russian_vocab = make_vocabulary(russian_sentences)

indexed_pairs = [
    (encode_sentence(e, english_vocab), encode_sentence(r, russian_vocab))
    for e, r in prepared_pairs
]

train_set, val_set = train_test_split(indexed_pairs, test_size=0.2, random_state=42)

train_corpus = ParallelCorpus(train_set)
val_corpus = ParallelCorpus(val_set)
train_batches = DataLoader(train_corpus, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch)
val_batches = DataLoader(val_corpus, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_batch)

# ---- Build & Train Model ----

src_vocab_size = len(english_vocab)
tgt_vocab_size = len(russian_vocab)

src_encoder = SourceEncoder(src_vocab_size, EMBED_DIM, HIDDEN_DIM, N_LAYERS)
tgt_decoder = TargetDecoder(tgt_vocab_size, EMBED_DIM, HIDDEN_DIM, N_LAYERS)

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator = NeuralTranslator(src_encoder, tgt_decoder, compute_device).to(compute_device)

optimiser = optim.Adam(translator.parameters(), lr=0.001)
pad_token = russian_vocab["<pad>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)

best_val = float("inf")
for ep in range(EPOCHS):
    tr_loss = train_epoch(translator, train_batches, optimiser, loss_fn, GRAD_CLIP, compute_device)
    vl_loss = eval_epoch(translator, val_batches, loss_fn, compute_device)
    if vl_loss < best_val:
        best_val = vl_loss
        torch.save(translator.state_dict(), "best_model_{}.pt".format(ep))
    print("Epoch {}\nTrain loss: {}\nVal loss: {}".format(ep + 1, tr_loss, vl_loss))

# ---- Load Best Checkpoint ----

translator.load_state_dict(torch.load("best_model_2.pt", map_location=compute_device))
translator.eval()

# ---- BLEU Scores (Greedy) ----

eval_sample = val_set[:1000]
b1, b2, b4 = bleu_evaluation(
    translator, eval_sample, english_vocab, russian_vocab, compute_device, decode_greedy
)
print("Greedy BLEU-1:", b1)
print("Greedy BLEU-2:", b2)
print("Greedy BLEU-4:", b4)

# ---- Decoding Speed Benchmark ----

t_greedy = timed_inference(decode_greedy, eval_sample)
t_w3 = timed_inference(decode_w3, eval_sample)
t_w5 = timed_inference(decode_w5, eval_sample)
t_w10 = timed_inference(decode_w10, eval_sample)

print("Greedy Time:", t_greedy)
print("Beam3 Time:", t_w3)
print("Beam5 Time:", t_w5)
print("Beam10 Time:", t_w10)

# ---- Qualitative Translation Samples ----

for sent in demo_sentences:
    greedy_out = decode_greedy(sent)
    beam5_out = decode_w5(sent)
    print("EN:", sent)
    print("Greedy:", " ".join(greedy_out))
    print("Beam5:", " ".join(beam5_out))
