import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define preprocessing function
def preprocess_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split('OUT:')
            input_seq = parts[0].split('IN:')[1].strip().split()
            target_seq = parts[1].strip().split()
            data.append((input_seq, target_seq))
    return data

# Load and preprocess the dataset
train_data = preprocess_dataset('D:/IIITH/my_scan_project/data/add_prim_split/tasks_train_addprim_jump.txt')
test_data = preprocess_dataset('D:/IIITH/my_scan_project/data/add_prim_split/tasks_test_addprim_jump.txt')

print("Number of training samples:", len(train_data))
print("Number of test samples:", len(test_data))
class ActionDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.input_vocab = self.build_vocab([x[0] for x in data])
        self.output_vocab = self.build_vocab([x[1] for x in data])
    
    def build_vocab(self, sequences):
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        for seq in sequences:
            for token in seq:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab
    
    def encode_sequence(self, seq, vocab):
        return [vocab.get(token, vocab['<UNK>']) for token in seq]
    
    def decode_sequence(self, seq, vocab):
        reverse_vocab = {v: k for k, v in vocab.items()}
        return [reverse_vocab.get(idx, '<UNK>') for idx in seq]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        input_ids = self.encode_sequence(input_seq, self.input_vocab)
        target_ids = self.encode_sequence(target_seq, self.output_vocab)
        return torch.tensor(input_ids), torch.tensor(target_ids)

# Define collate function for padding
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

# Hyperparameters
batch_size = 32
embedding_dim = 256
hidden_dim = 100
num_epochs = 20
learning_rate = 0.001

# Create datasets and dataloaders
train_dataset = ActionDataset(train_data)
test_dataset = ActionDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
class UniversalTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(UniversalTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class UniversalTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(UniversalTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_dim)
        
        self.encoder_layers = nn.ModuleList([UniversalTransformerLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([UniversalTransformerLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        
        self.output_layer = nn.Linear(embed_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt):
        encoder_embedded = self.encoder_embedding(src).transpose(0, 1)
        decoder_embedded = self.decoder_embedding(tgt).transpose(0, 1)
        
        for layer in self.encoder_layers:
            encoder_embedded = layer(encoder_embedded)
        
        for layer in self.decoder_layers:
            decoder_embedded = layer(decoder_embedded)
        
        output = self.output_layer(decoder_embedded)
        
        return output.transpose(0, 1)
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            outputs = model(inputs, targets[:, :-1])
            loss = criterion(outputs.reshape(-1, output_vocab_size), targets[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs, -1)
            train_total += targets.size(0) * targets.size(1)
            train_correct += (predicted == targets[:, 1:]).sum().item()
        
        train_accuracy = train_correct / train_total
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Train Accuracy: {train_accuracy}')

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            outputs = model(inputs, targets[:, :-1])
            loss = criterion(outputs.reshape(-1, output_vocab_size), targets[:, 1:].reshape(-1))
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, -1)
            test_total += targets.size(0) * targets.size(1)
            test_correct += (predicted == targets[:, 1:]).sum().item()
    
    test_accuracy = test_correct / test_total
    print(f'Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy}')

# Instantiate the model
input_vocab_size = len(train_dataset.input_vocab)
output_vocab_size = len(train_dataset.output_vocab)

embed_dim = 256
num_heads = 8
ff_dim = 512
num_layers = 4

model = UniversalTransformer(input_vocab_size, output_vocab_size, embed_dim, num_heads, ff_dim, num_layers)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and evaluate the model
train_model(model, train_loader, criterion, optimizer, num_epochs)
evaluate_model(model, test_loader, criterion)
