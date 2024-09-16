import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Example parameters
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
learning_rate = 0.001
batch_size = 32
num_epochs = 10

# Function to remove non-integer tokens from a list of sentences
def remove_non_integer_tokens(data):
    cleaned_data = []
    for sentence in data:
        cleaned_sentence = []
        for token in sentence.split():
            try:
                int(token)  # Check if token can be converted to an integer
                cleaned_sentence.append(token)
            except ValueError:
                continue  # Skip non-integer tokens
        cleaned_data.append(" ".join(cleaned_sentence))
    return cleaned_data

# Function to read data from file
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

# Paths to your data files
TRAIN_SOURCE_FILE = r"C:\Users\Admin\Downloads\COGS\PCFG\train_src.txt"
TRAIN_TARGET_FILE = r"C:\Users\Admin\Downloads\COGS\PCFG\train_tgt.txt"

# Load source and target data
source_data = read_data(TRAIN_SOURCE_FILE)
target_data = read_data(TRAIN_TARGET_FILE)

# Remove non-integer tokens from source and target data
cleaned_source_data = remove_non_integer_tokens(source_data)
cleaned_target_data = remove_non_integer_tokens(target_data)

# Tokenization and Vocabulary creation
class Vocabulary:
    def __init__(self, data):
        self.word2idx = {}
        self.idx2word = {}
        self.build_vocab(data)
        
    def build_vocab(self, data):
        special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        for i, token in enumerate(special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
        
        idx = len(special_tokens)
        for sentence in data:
            for token in sentence.split():
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token
                    idx += 1
    
    def encode_sentence(self, sentence):
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in sentence.split()]
    
    def decode_sentence(self, encoded_sentence):
        return " ".join([self.idx2word[idx] for idx in encoded_sentence])

# Create vocabularies for source and target data
source_vocab = Vocabulary(cleaned_source_data)
target_vocab = Vocabulary(cleaned_target_data)

# Convert sentences to sequences of integers using vocabulary
encoded_source_data = [source_vocab.encode_sentence(sentence) for sentence in cleaned_source_data]
encoded_target_data = [target_vocab.encode_sentence(sentence) for sentence in cleaned_target_data]

# Padding sequences to ensure uniform length
def pad_sequences(data, max_length):
    padded_data = []
    for sequence in data:
        if len(sequence) < max_length:
            padded_sequence = sequence + [source_vocab.word2idx['<pad>']] * (max_length - len(sequence))
        else:
            padded_sequence = sequence[:max_length]
        padded_data.append(padded_sequence)
    return padded_data

# Determine max sequence length
max_source_length = max(len(sequence) for sequence in encoded_source_data)
max_target_length = max(len(sequence) for sequence in encoded_target_data)

# Pad sequences
padded_source_data = pad_sequences(encoded_source_data, max_source_length)
padded_target_data = pad_sequences(encoded_target_data, max_target_length)

# Convert to PyTorch tensors
source_tensor = torch.tensor(padded_source_data, dtype=torch.long)
target_tensor = torch.tensor(padded_target_data, dtype=torch.long)

# Define dataset class
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_tensor, tgt_tensor):
        self.src_tensor = src_tensor
        self.tgt_tensor = tgt_tensor
        assert len(self.src_tensor) == len(self.tgt_tensor), "Source and target lengths must be equal"
    
    def __len__(self):
        return len(self.src_tensor)
    
    def __getitem__(self, index):
        source = self.src_tensor[index]
        target = self.tgt_tensor[index]
        return source, target

# Create datasets and data loaders
dataset = TranslationDataset(source_tensor, target_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # Ensure src and tgt have shape [seq_len, batch_size, d_model]
        src = src.permute(1, 0)  # [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
        tgt = tgt.permute(1, 0)  # [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
        
        memory = self.transformer(src, tgt)
        output = self.fc(memory)
        return output

# Instantiate the model
model = TransformerModel(
    src_vocab_size=len(source_vocab.word2idx),
    tgt_vocab_size=len(target_vocab.word2idx),
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for src_batch, tgt_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(src_batch, tgt_batch)
        
        # Flatten outputs and targets for the loss function
        outputs_flat = outputs.view(-1, tgt_vocab_size)
        targets_flat = tgt_batch.view(-1).long()
        
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
