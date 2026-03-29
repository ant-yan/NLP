# ============================================================
# Bag-of-Words Embedding Classifier
# Encodes documents as averaged word vectors, then routes
# them through a two-layer MLP for binary categorization.
# Embedding shifts are tracked before and after optimization.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- Global Hyperparameters ----
EMBED_DIM = 50
HIDDEN_DIM = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TOTAL_EPOCHS = 50
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"

# ---- Probe Words for Embedding Analysis ----
similarity_probes = ["team", "game", "player", "software", "quantum"]

# ---- Visualization Token List ----
visualization_tokens = [
    "team", "game", "player", "coach", "championship",
    "match", "goal", "athlete",
    "software", "quantum", "algorithm", "technology",
    "innovation", "digital", "computer", "AI"
]

token_colors = ["blue"] * 8 + ["red"] * 8

# ---- Raw Labeled Corpus ----
training_corpus = [
    "Lakers win championship after amazing comeback",
    "Football team scores three goals in final match",
    "Tennis star wins grand slam tournament",
    "Olympic athlete breaks world record",
    "Basketball game ends in dramatic overtime",
    "Soccer team advances to finals",
    "Runner wins marathon in record time",
    "Baseball team clinches playoff spot",
    "Swimmer wins gold medal at Olympics",
    "Hockey team defeats rival in shootout",
    "Gymnast performs perfect routine at competition",
    "Boxer wins title fight in final round",
    "Cycling champion wins Tour de France stage",
    "Golfer sinks incredible putt to win tournament",
    "Figure skater lands difficult jump perfectly",
    "Wrestling champion defends title successfully",
    "Cricket team wins test match series",
    "Rugby team scores last minute try",
    "Volleyball team wins championship match",
    "Badminton player wins international tournament",
    "Team celebrates victory after tough game",
    "Players train hard for upcoming season",
    "Coach announces new strategy for playoffs",
    "Stadium packed with fans for final game",
    "Athletes compete in regional championship",
    "Team captain leads squad to victory",
    "Young player shows great potential",
    "Fans celebrate team winning streak",
    "Manager signs new contract with club",
    "Training camp begins for new season",
    "Team wins away game against rivals",
    "Player receives award for performance",
    "Championship game draws huge crowd",
    "Team prepares for important match",
    "Coach praises players after victory",
    "Athletes set new records at event",
    "Team qualifies for next round",
    "Player scores hat trick in match",
    "Championship trophy awarded to winners",
    "Team celebrates historic achievement",
    "Players excited for tournament start",
    "Coach announces starting lineup today",
    "Team practices before big game",
    "Victory parade celebrates championship win",
    "Players sign autographs for fans",
    "Team mascot entertains crowd at game",
    "Championship banner raised at stadium",
    "Team dominates in playoff series",
    "Player injury sidelines star athlete",
    "Team rebuilds roster for season",
    "Coach implements new training program",
    "Players bond during team retreat",
    "Stadium renovations completed for season",
    "Team announces ticket prices for games",
    "Player traded to rival team",
    "Coach resigns after disappointing season",
    "Team practices penalty kicks for game",
    "Championship ring ceremony held today",
    "Team wins tournament in overtime",
    "Player breaks scoring record in game",
    "Coach gives motivational speech to team",
    "Team celebrates winning season finale",
    "Athletes train for upcoming competition",
    "Player signs endorsement deal today",
    "Team unveils new uniform design",
    "Championship parade draws massive crowd",
    "Coach analyzes game film with players",
    "Team defeats defending champions convincingly",
    "Player receives sportsmanship award today",
    "Team holds charity event for fans",
    "Athletes prepare for championship match",
    "Coach praises team effort after win",
    "Team advances in tournament bracket",
    "Player scores winning goal in match",
    "Championship celebrations continue all week",
    "Team announces preseason schedule today",
    "Players work hard during practice",
    "Coach happy with team performance",
    "Team wins decisive game at home",
    "Athletes excited for season opener",
    "Player makes incredible save in game",
    "Team rallies from behind for victory",
    "Coach confident before important match",
    "Championship run inspires young athletes",
    "Team prepares strategy for rivals",
    "Player demonstrates leadership on field",
    "Team celebrates milestone victory today",
    "Coach reviews tactics with players",
    "Athletes compete in regional finals",
    "Player achieves personal best in event",
    "Team practices formations for game",
    "Championship trophy displayed at stadium",
    "Coach motivates team before playoffs",
    "Team wins thrilling match in finale",
    "Player earns spot on national team",
    "Athletes train for international competition",
    "Team holds press conference today",
    "New smartphone features advanced AI technology",
    "Tech company releases latest software update",
    "Scientists develop breakthrough quantum computer",
    "Artificial intelligence system improves healthcare",
    "New app helps users learn programming",
    "Electric vehicle company announces new model",
    "Researchers create faster internet connection",
    "Social media platform adds new features",
    "Cloud computing service expands globally",
    "Cybersecurity system protects against attacks",
    "Virtual reality headset launches next month",
    "Machine learning algorithm solves complex problem",
    "New programming language released by developers",
    "Tech startup raises millions in funding",
    "5G network expands to more cities",
    "Robotics company builds autonomous system",
    "Data center uses renewable energy",
    "Blockchain technology improves security",
    "New laptop features powerful processor",
    "Software update fixes major bugs",
    "Company develops innovative tech solution",
    "Algorithm improves search accuracy online",
    "Digital platform streamlines business operations",
    "Innovation drives tech industry forward",
    "Startup creates app for education",
    "Computer chip breakthrough announced today",
    "Technology advances medical research capabilities",
    "New software helps developers code faster",
    "Tech firm invests in AI research",
    "Digital transformation changes business landscape",
    "Scientists program robot for tasks",
    "Tech company expands into new markets",
    "Innovation lab opens in Silicon Valley",
    "New technology revolutionizes communication industry",
    "Software engineers develop better tools",
    "Tech conference showcases latest innovations",
    "Startup creates platform for collaboration",
    "Computer vision system recognizes objects accurately",
    "Technology enables remote work solutions",
    "Digital security measures protect data",
    "Algorithm optimizes supply chain operations",
    "Tech giant announces quarterly earnings today",
    "Innovation accelerates in artificial intelligence",
    "Software update enhances user experience",
    "Tech startup disrupts traditional industry",
    "Computer scientists breakthrough in research",
    "Digital platform connects developers worldwide",
    "Technology transforms healthcare delivery system",
    "New app simplifies complex tasks",
    "Innovation drives efficiency in business",
    "Tech company partners with university",
    "Software development tools improve productivity",
    "Algorithm processes data faster now",
    "Technology enables smart home systems",
    "Digital innovation changes customer experience",
    "Tech firm releases open source software",
    "Computer program automates repetitive work",
    "Innovation creates new tech opportunities",
    "Software engineers collaborate on project",
    "Technology advances renewable energy solutions",
    "Digital tools help students learn",
    "Tech startup solves infrastructure problem",
    "Algorithm improves recommendation system accuracy",
    "Innovation drives semiconductor industry growth",
    "Software platform integrates multiple services",
    "Technology enables precision medicine approach",
    "Digital assistant becomes more intelligent",
    "Tech company invests in quantum research",
    "Computer network expands bandwidth capacity",
    "Innovation transforms financial services industry",
    "Software developers build mobile applications",
    "Technology improves agricultural productivity significantly",
    "Digital marketplace connects buyers sellers",
    "Tech firm announces new partnership",
    "Algorithm detects patterns in data",
    "Innovation accelerates autonomous vehicle development",
    "Software update adds security features",
    "Technology enables virtual collaboration tools",
    "Digital platform supports remote learning",
    "Tech startup creates innovative solution",
    "Computer system processes information quickly",
    "Innovation drives biotechnology research forward",
    "Software engineers optimize application performance",
    "Technology transforms manufacturing processes today",
    "Digital tools enhance creative workflows",
    "Tech company expands research division",
    "Algorithm improves translation accuracy significantly",
    "Innovation creates sustainable technology solutions",
    "Software platform enables data analysis",
    "Technology advances space exploration capabilities",
    "Digital infrastructure supports cloud services",
    "Tech firm develops edge computing solution",
    "Computer scientists research neural networks",
    "Innovation drives telecommunications industry growth",
    "Software development becomes more accessible",
    "Technology enables personalized learning experiences",
    "Digital security protects online transactions",
]

evaluation_corpus = [
    "Team wins game in final seconds",
    "Player scores amazing goal today",
    "Coach proud of team performance",
    "Athletes compete in championship finals",
    "Team defeats rivals in playoff",
    "Player breaks record in tournament",
    "Championship game ends in victory",
    "Team prepares for important match",
    "Athletes train for upcoming season",
    "Player receives award for excellence",
    "Team celebrates winning streak today",
    "Coach announces strategy for game",
    "Championship trophy presented to team",
    "Player scores in overtime win",
    "Team advances to next round",
    "Athletes perform well at competition",
    "Coach motivates players before match",
    "Team wins decisive playoff game",
    "Player demonstrates skill on field",
    "Championship victory celebrated by fans",
    "Team dominates in tournament play",
    "Athletes excited for season start",
    "Player achieves milestone in career",
    "Team practices before big match",
    "Championship parade honors winning team",
    "Software company releases new product",
    "Algorithm improves system performance significantly",
    "Tech startup develops innovative platform",
    "Digital tools enhance productivity today",
    "Innovation drives technology sector forward",
    "Computer program solves difficult problem",
    "Technology transforms business operations completely",
    "Software engineers create better solutions",
    "Tech firm announces breakthrough research",
    "Digital platform connects users globally",
    "Algorithm processes information efficiently now",
    "Innovation accelerates in tech industry",
    "Software update improves functionality greatly",
    "Technology enables new capabilities today",
    "Tech company invests in development",
    "Digital system automates complex tasks",
    "Innovation creates technology opportunities now",
    "Software platform integrates services seamlessly",
    "Technology advances research capabilities significantly",
    "Tech startup solves industry challenge",
    "Algorithm optimizes operations effectively today",
    "Innovation drives digital transformation forward",
    "Software developers build applications efficiently",
    "Technology improves user experience greatly",
    "Digital innovation changes industry landscape",
]


# ---- Text Processing Utilities ----

def normalize_and_split(raw_text):
    """Lowercases the input and returns individual word tokens."""
    lowered = raw_text.lower()
    return lowered.split()


def map_to_indices(token_list, lexicon):
    """Converts a list of string tokens into their integer indices."""
    oov_idx = lexicon[OOV_TOKEN]
    return [lexicon.get(tok, oov_idx) for tok in token_list]


def apply_padding(sequences, target_len, fill_value=0):
    """Right-pads each sequence with fill_value to reach target_len."""
    return [seq + [fill_value] * (target_len - len(seq)) for seq in sequences]


def retrieve_embedding(word, embed_matrix):
    """Looks up the embedding vector for a given word token."""
    lookup_idx = lexicon.get(word, lexicon[OOV_TOKEN])
    return embed_matrix[lookup_idx]


def vector_similarity(vec_a, vec_b):
    """Computes cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()


# ---- Neural Architecture ----

class BagOfWordsNet(nn.Module):
    """
    Represents documents as masked mean-pooled embeddings,
    then classifies via a sigmoid-activated hidden layer.
    """

    def __init__(self, lexicon_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=lexicon_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )
        self.fc_hidden = nn.Linear(embed_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, 2)

    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        pad_mask = (token_ids != 0).float().unsqueeze(2)
        emb = emb * pad_mask
        summed = emb.sum(dim=1)
        word_count = pad_mask.sum(dim=1)
        mean_emb = summed / word_count
        hidden_act = torch.sigmoid(self.fc_hidden(mean_emb))
        return self.fc_output(hidden_act)


class CorpusDataset(Dataset):
    """Wraps feature tensors and label tensors into an iterable dataset."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ---- Data Preparation ----

tokenized_corpus = [normalize_and_split(doc) for doc in training_corpus]

token_frequency = Counter()
for token_list in tokenized_corpus:
    token_frequency.update(token_list)

lexicon = {PAD_TOKEN: 0, OOV_TOKEN: 1}
for word in token_frequency:
    lexicon[word] = len(lexicon)

id_to_token = {idx: word for word, idx in lexicon.items()}

indexed_corpus = [map_to_indices(tokens, lexicon) for tokens in tokenized_corpus]

maximum_sequence_len = max(len(seq) for seq in indexed_corpus)

padded_corpus = apply_padding(indexed_corpus, maximum_sequence_len, fill_value=lexicon[PAD_TOKEN])

category_labels = [0] * 97 + [1] * 97

feature_tensor = torch.tensor(padded_corpus, dtype=torch.long)
label_tensor = torch.tensor(category_labels, dtype=torch.long)

training_set = CorpusDataset(feature_tensor, label_tensor)
sample_generator = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)

# ---- Model Instantiation ----

lexicon_size = len(lexicon)
classifier = BagOfWordsNet(lexicon_size)
loss_criterion = nn.CrossEntropyLoss()
param_optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

# Snapshot embeddings before any gradient updates
pre_train_emb = classifier.embedding.weight.data.clone()

# ---- Training Loop ----

for round_num in range(1, TOTAL_EPOCHS + 1):
    classifier.train()
    cumulative_loss = 0.0

    for batch_features, batch_labels in sample_generator:
        param_optimizer.zero_grad()
        predictions = classifier(batch_features)
        batch_loss = loss_criterion(predictions, batch_labels)
        batch_loss.backward()
        param_optimizer.step()
        cumulative_loss += batch_loss.item()

    batch_avg_loss = cumulative_loss / len(sample_generator)
    print("Epoch {}, Loss: {:.4f}".format(round_num, batch_avg_loss))

# Snapshot embeddings after training completes
post_train_emb = classifier.embedding.weight.data.clone()

torch.save(classifier.state_dict(), "text_classifier.pth")
print("Saved weights to text_classifier.pth")

# ---- Embedding Similarity Comparison ----

print("{:<10}{:<10}{:<10}{:<10}".format("Word 1", "Word 2", "Before", "After"))
for i in range(len(similarity_probes)):
    for j in range(i + 1, len(similarity_probes)):
        w1, w2 = similarity_probes[i], similarity_probes[j]
        before = vector_similarity(
            retrieve_embedding(w1, pre_train_emb),
            retrieve_embedding(w2, pre_train_emb)
        )
        after = vector_similarity(
            retrieve_embedding(w1, post_train_emb),
            retrieve_embedding(w2, post_train_emb)
        )
        print("{:<10}{:<10}{:<10.4f}{:<10.4f}".format(w1, w2, before, after))

# ---- 2D PCA Visualization of Learned Embeddings ----

embedding_stack = torch.stack(
    [retrieve_embedding(w, post_train_emb) for w in visualization_tokens]
).detach().numpy()

pca_reducer = PCA(n_components=2)
projected_embeddings = pca_reducer.fit_transform(embedding_stack)

plt.figure(figsize=(8, 8))
for i, word in enumerate(visualization_tokens):
    px, py = projected_embeddings[i]
    plt.scatter(px, py, color=token_colors[i], s=100)
    plt.text(px + 0.01, py + 0.01, word, fontsize=10)

plt.title("2D PCA of Word Embeddings (Post-Training)")
plt.grid(True)
plt.tight_layout()
plt.show()
