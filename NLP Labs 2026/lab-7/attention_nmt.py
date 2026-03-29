# ============================================================
# Attention-Augmented Seq2Seq Machine Translation
# Implements Bahdanau-style additive attention so the decoder
# can dynamically focus on relevant encoder positions at each
# generation step, addressing the information bottleneck of
# fixed-size context vectors used in vanilla Seq2Seq models.
# ============================================================

import re
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---- Global Hyperparameters ----
BATCH_SIZE = 32
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0005
TOTAL_EPOCHS = 20
CLIP_NORM = 1

# ---- Data File ----
source_file = "rus.txt"

# ---- Sample Inputs for Demonstration ----
sample_inputs = [
    "hello how are you",
    "i love walking",
    "that is wonderful",
    "this is a good idea",
    "please show me the way",
    "can you help me",
    "i need help",
    "interesting story",
    "that is a difficult question",
    "i will tell you",
]


# ---- Preprocessing ----

def sanitize_text(raw):
    """Strips non-alphabetic characters and folds to lowercase."""
    lowered = raw.lower()
    cleaned = re.sub(r"[^a-zа-яё\s]", "", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def make_vocabulary(sentences):
    """Builds a word-index mapping from a corpus of sentences."""
    freq = Counter()
    for sent in sentences:
        freq.update(sent.split())
    vocab = {"<pad>": 0, "<oov>": 1}
    for token in freq:
        vocab[token] = len(vocab)
    return vocab


def encode_sentence(sentence, vocab):
    """Maps string tokens to their integer IDs."""
    tokens = sentence.split() if isinstance(sentence, str) else sentence
    return [vocab.get(t, vocab["<oov>"]) for t in tokens]


# ---- Dataset ----

class ParallelCorpus(Dataset):
    """Stores pairs of source/target integer sequences."""

    def __init__(self, pair_list):
        self.pairs = pair_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        return torch.tensor(s), torch.tensor(t)


def pad_batch(batch):
    """Collates variable-length sequence pairs into padded tensors."""
    src_seqs, tgt_seqs = zip(*batch)
    return (
        pad_sequence(src_seqs, padding_value=0),
        pad_sequence(tgt_seqs, padding_value=0),
    )


# ---- Model Components ----

class SequenceEncoder(nn.Module):
    """
    LSTM encoder that returns all hidden states (not just the final one)
    so the attention mechanism can align over the full source sequence.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0
        )

    def forward(self, src):
        embedded = self.embedding(src)
        enc_outputs, (hidden, cell) = self.lstm(embedded)
        return enc_outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention: combines the current decoder
    hidden state with each encoder output through a learned linear
    projection, then scores with a single-unit output layer.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs, mask=None):
        src_len = enc_outputs.shape[0]
        dec_hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        enc_perm = enc_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, enc_perm), dim=2)))
        scores = self.v(energy).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)


class AttentionDecoder(nn.Module):
    """
    LSTM decoder conditioned on a weighted sum of encoder outputs
    (context vector) computed via attention at each decoding step.
    The final projection concatenates the LSTM output, context, and
    the current embedding to give the full output vocabulary distribution.
    """

    def __init__(self, output_dim, emb_dim, hidden_dim, attention, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(emb_dim + hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, cell, enc_outputs, mask=None):
        token = token.unsqueeze(0)
        embedded = self.dropout(self.embedding(token))
        attn_weights = self.attention(hidden, enc_outputs, mask).unsqueeze(1)
        context = torch.bmm(attn_weights, enc_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        rnn_in = torch.cat((embedded, context), dim=2)
        out, (hidden, cell) = self.rnn(rnn_in, (hidden, cell))
        proj_in = torch.cat(
            (out.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1
        )
        return self.fc_out(proj_in), hidden, cell


class AttentionTranslator(nn.Module):
    """
    Full attention Seq2Seq pipeline; the encoder produces a set of
    context vectors and the decoder attends to them at each step.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5, mask=None):
        batch_sz = trg.shape[1]
        trg_steps = trg.shape[0]
        tgt_vocab_sz = self.decoder.output_dim

        all_outputs = torch.zeros(trg_steps, batch_sz, tgt_vocab_sz).to(self.device)
        enc_outputs, hidden, cell = self.encoder(src)
        current_token = trg[0]

        for t in range(1, trg_steps):
            pred, hidden, cell = self.decoder(
                current_token, hidden, cell, enc_outputs, mask
            )
            all_outputs[t] = pred
            top_token = pred.argmax(1)
            force = random.random() < teacher_forcing_ratio
            current_token = trg[t] if force else top_token

        return all_outputs


def initialize_weights(module):
    """Applies small normal initialisation to weights; zeros biases."""
    for name, param in module.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# ---- Training & Validation Passes ----

def train_pass(net, loader, optimiser, loss_fn, clip, device, epoch):
    """Executes one epoch of supervised training with teacher forcing."""
    net.train()
    epoch_loss = 0.0
    force_ratio = 0.5

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)
        optimiser.zero_grad()
        preds = net(src, trg, teacher_forcing_ratio=force_ratio)
        flat_preds = preds[1:].view(-1, preds.shape[-1])
        flat_trg = trg[1:].view(-1)
        loss = loss_fn(flat_preds, flat_trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimiser.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def val_pass(net, loader, loss_fn, device):
    """Evaluates the model on the validation split without gradients."""
    net.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            preds = net(src, trg, teacher_forcing_ratio=0)
            flat_preds = preds[1:].view(-1, preds.shape[-1])
            flat_trg = trg[1:].view(-1)
            epoch_loss += loss_fn(flat_preds, flat_trg).item()

    return epoch_loss / len(loader)


# ---- Attention-Aware Beam Search ----

def translate_with_beam(sentence, net, src_vocab, tgt_vocab, device,
                        beam_size=5, max_len=50, min_len=3):
    """
    Beam search over the attention decoder; penalises <eos> emission
    before min_len tokens to avoid trivially short outputs.
    """
    net.eval()
    tokens = sentence.lower().split() if isinstance(sentence, str) else sentence
    src_ids = [src_vocab.get(t, src_vocab["<oov>"]) for t in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)

    with torch.no_grad():
        enc_outputs, hidden, cell = net.encoder(src_tensor)

    beams = [([tgt_vocab["<sos>"]], 0.0, hidden, cell)]

    for step in range(max_len):
        new_beams = []
        for seq, score, h, c in beams:
            last = seq[-1]
            if last == tgt_vocab["<eos>"]:
                new_beams.append((seq, score, h, c))
                continue

            tok_tensor = torch.LongTensor([last]).to(device)
            with torch.no_grad():
                out, h_new, c_new = net.decoder(tok_tensor, h, c, enc_outputs)

            log_probs = F.log_softmax(out, dim=1)
            if len(seq) < min_len:
                log_probs[0][tgt_vocab["<eos>"]] = -1e9

            top_probs, top_idxs = log_probs.topk(beam_size)
            for k in range(beam_size):
                new_beams.append((
                    seq + [top_idxs[0][k].item()],
                    score + top_probs[0][k].item(),
                    h_new, c_new,
                ))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(seq[-1] == tgt_vocab["<eos>"] for seq, _, _, _ in beams):
            break

    best_seq = beams[0][0]
    inv_vocab = {v: k for k, v in tgt_vocab.items()}

    result = []
    for idx in best_seq[1:]:
        word = inv_vocab[idx]
        if word == "<eos>":
            break
        result.append(word)
    return result


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

# ---- Model Assembly ----

source_vocab_size = len(english_vocab)
target_vocab_size = len(russian_vocab)

attention_layer = BahdanauAttention(HIDDEN_DIM)
sequence_encoder = SequenceEncoder(source_vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE)
attention_decoder = AttentionDecoder(target_vocab_size, EMBED_DIM, HIDDEN_DIM, attention_layer, DROPOUT_RATE)

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attn_translator = AttentionTranslator(sequence_encoder, attention_decoder, compute_device).to(compute_device)
attn_translator.apply(initialize_weights)

adam_optimizer = optim.Adam(attn_translator.parameters(), lr=LEARNING_RATE)
pad_token_idx = russian_vocab["<pad>"]
nll_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_idx)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    adam_optimizer, mode="min", patience=2, factor=0.5
)

# ---- Training Loop ----

best_val = float("inf")

for epoch_num in range(TOTAL_EPOCHS):
    tr_loss = train_pass(
        attn_translator, train_batches, adam_optimizer,
        nll_criterion, CLIP_NORM, compute_device, epoch_num
    )
    vl_loss = val_pass(attn_translator, val_batches, nll_criterion, compute_device)
    lr_scheduler.step(vl_loss)

    if vl_loss < best_val:
        best_val = vl_loss
        torch.save(attn_translator.state_dict(), "best_att_model_final.pt")
        print("  Best model saved (val loss: {:.4f})".format(vl_loss))

    print("Epoch {:02d} | Train: {:.4f} | Val: {:.4f} | LR: {:.6f}".format(
        epoch_num + 1, tr_loss, vl_loss,
        adam_optimizer.param_groups[0]["lr"]
    ))

# ---- Load Best Checkpoint & Run Inference ----

attn_translator.load_state_dict(
    torch.load("best_att_model_final.pt", map_location=compute_device)
)
attn_translator.eval()

for sent in sample_inputs:
    greedy_out = translate_with_beam(
        sent, attn_translator, english_vocab, russian_vocab,
        compute_device, beam_size=1
    )
    beam2_out = translate_with_beam(
        sent, attn_translator, english_vocab, russian_vocab,
        compute_device, beam_size=2
    )
    beam5_out = translate_with_beam(
        sent, attn_translator, english_vocab, russian_vocab,
        compute_device, beam_size=5
    )
    print("EN:", sent)
    print("Greedy:", " ".join(greedy_out))
    print("Beam2:", " ".join(beam2_out))
    print("Beam5:", " ".join(beam5_out))
    print("-" * 50)
