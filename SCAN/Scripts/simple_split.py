import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import numpy as np
import math

# Define Fields
SRC = Field(tokenize="spacy", tokenizer_language="en_core_web_sm", lower=True, init_token="<sos>", eos_token="<eos>")
TRG = Field(tokenize="spacy", tokenizer_language="en_core_web_sm", lower=True, init_token="<sos>", eos_token="<eos>")

# Load and split dataset
train_data, test_data = TabularDataset.splits(
    path="D:/IIITH/my_scan_project/data/simple_split/",
    train="tasks_train_simple.txt",
    test="tasks_test_simple.txt",
    format="tsv",
    fields=[('src', SRC), ('trg', TRG)]
)

# Build vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Universal Transformer model
class UniversalTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(UniversalTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.positional_encoding = PositionalEncoding(emb_dim, dropout)
        self.transformer_layers = nn.ModuleList([TransformerLayer(emb_dim, hid_dim, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.embedding(src) * np.sqrt(self.embedding.weight.size(1))
        embedded = self.positional_encoding(embedded)
        embedded = self.dropout(embedded)

        for layer in self.transformer_layers:
            embedded = layer(embedded)

        output = self.fc_out(embedded)
        return output

# Define training process
def train(model, iterator, optimizer, criterion):
    model.train()
    correct_preds = 0
    total_preds = 0

    for batch in iterator:
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()
        output = model(src)
        preds = output.argmax(dim=-1)
        correct_preds += (preds == trg).sum().item()
        total_preds += trg.numel()

        loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
        loss.backward()
        optimizer.step()

    accuracy = correct_preds / total_preds
    return accuracy

# Define evaluation process
def evaluate(model, iterator, criterion):
    model.eval()
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg

            output = model(src)
            preds = output.argmax(dim=-1)
            correct_preds += (preds == trg).sum().item()
            total_preds += trg.numel()

    accuracy = correct_preds / total_preds
    return accuracy

# Model parameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 6
DROPOUT = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 10

# Initialize model, optimizer, and criterion
model = UniversalTransformer(INPUT_DIM, OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

# Define iterators
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort=False
)

# Train and evaluate the model
for epoch in range(N_EPOCHS):
    train_accuracy = train(model, train_iterator, optimizer, criterion)
    test_accuracy = evaluate(model, test_iterator, criterion)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Accuracy: {train_accuracy*100:.2f}%')
    print(f'\tTest Accuracy: {test_accuracy*100:.2f}%')
