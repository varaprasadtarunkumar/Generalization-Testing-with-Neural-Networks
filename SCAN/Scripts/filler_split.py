# evaluate.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm
import sacrebleu

# Define the preprocessing function
def preprocess_translation_dataset(file_path):
    data = []
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return data
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(' OUT: ')
            if len(parts) == 2:
                input_seq = parts[0].replace('IN: ', '').strip().split()
                target_seq = parts[1].strip().split()
                data.append((input_seq, target_seq))
            else:
                print(f"Line not in correct format: {line}")
    print(f"Loaded {len(data)} samples from {file_path}")
    return data

# Define the dataset class
class TranslationDataset(Dataset):
    def __init__(self, data, input_vocab, output_vocab):
        self.data = data
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
    
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

def evaluate_model(model, test_loader, criterion, dataset):
    model.eval()
    test_loss = 0
    predictions = []
    references = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            outputs = model(inputs, targets[:, :-1])
            loss = criterion(outputs.reshape(-1, output_vocab_size), targets[:, 1:].reshape(-1))
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, -1)
            predictions.extend(predicted.cpu().numpy())
            references.extend(targets[:, 1:].cpu().numpy())
    
    print(f'Test Loss: {test_loss / len(test_loader)}')
    
    # Convert predictions and references to sentences
    pred_sentences = [' '.join(dataset.decode_sequence(pred, dataset.output_vocab)) for pred in predictions]
    ref_sentences = [[' '.join(dataset.decode_sequence(ref, dataset.output_vocab))] for ref in references]

    # Calculate BLEU score
    bleu_score = sacrebleu.corpus_bleu(pred_sentences, ref_sentences)
    print(f'BLEU Score: {bleu_score.score}')

# Load and preprocess the dataset
test_data = preprocess_translation_dataset('D:/IIITH/my_scan_project/data/simple_split/tasks_test_simple.txt')

print("Number of test samples:", len(test_data))

# Load vocabularies
input_vocab = torch.load('input_vocab.pth')
output_vocab = torch.load('output_vocab.pth')

# Create datasets and dataloaders
test_dataset = TranslationDataset(test_data, input_vocab, output_vocab)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

input_vocab_size = len(test_dataset.input_vocab)
output_vocab_size = len(test_dataset.output_vocab)

embed_dim = 256
num_heads = 8
ff_dim = 512
num_layers = 4

# Load the model
model = UniversalTransformer(input_vocab_size, output_vocab_size, embed_dim, num_heads, ff_dim, num_layers)
model.load_state_dict(torch.load('model.pth'))
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Evaluate the model
evaluate_model(model, test_loader, criterion, test_dataset)
