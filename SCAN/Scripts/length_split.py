import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Path to your dataset files
train_data_path = r'D:\IIITH\my_scan_project\data\length_split\tasks_train_length.txt'
test_data_path = r'D:\IIITH\my_scan_project\data\length_split\tasks_test_length.txt'

# Load and preprocess data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    # Split commands into inputs and outputs
    inputs = []
    outputs = []
    for line in data:
        parts = line.strip().split('OUT: ')
        inputs.append(parts[0].strip().split())
        outputs.append(parts[1].strip().split())
    return inputs, outputs

train_inputs, train_outputs = load_data(train_data_path)
test_inputs, test_outputs = load_data(test_data_path)

# Tokenize data
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False)
tokenizer.fit_on_texts(train_inputs + train_outputs)

# Convert text sequences to token sequences
train_inputs_seq = tokenizer.texts_to_sequences(train_inputs)
train_outputs_seq = tokenizer.texts_to_sequences(train_outputs)
test_inputs_seq = tokenizer.texts_to_sequences(test_inputs)
test_outputs_seq = tokenizer.texts_to_sequences(test_outputs)

# Pad sequences
max_length = max(max(len(seq) for seq in train_inputs_seq),
                 max(len(seq) for seq in test_inputs_seq),
                 max(len(seq) for seq in train_outputs_seq),
                 max(len(seq) for seq in test_outputs_seq))

train_inputs_pad = pad_sequences(train_inputs_seq, maxlen=max_length, padding='post')
train_outputs_pad = pad_sequences(train_outputs_seq, maxlen=max_length, padding='post')
test_inputs_pad = pad_sequences(test_inputs_seq, maxlen=max_length, padding='post')
test_outputs_pad = pad_sequences(test_outputs_seq, maxlen=max_length, padding='post')

# Define Universal Transformer model
class UniversalTransformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_length):
        super(UniversalTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_encoding = self.positional_encoding(max_length, embed_dim)
        
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]
        
        self.dropout = Dropout(0.1)
        self.final_layer = Dense(vocab_size, activation='softmax')
    
    def call(self, inputs, training=True):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_encoding[:, :seq_len, :]
        
        x = self.embedding(inputs) + positions
        x = self.dropout(x, training=training)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        x = self.final_layer(x)
        return x
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

# Define Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, inputs, training=True):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Hyperparameters
vocab_size = len(tokenizer.word_index) + 1
embed_dim = 256
num_heads = 8
ff_dim = 512
num_layers = 4
batch_size = 32
epochs = 10

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs_pad, train_outputs_pad)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs_pad, test_outputs_pad)).batch(batch_size)

# Initialize model
model = UniversalTransformer(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
                             ff_dim=ff_dim, num_layers=num_layers, max_length=max_length)
loss_fn = SparseCategoricalCrossentropy()
optimizer = Adam()

# Training loop
for epoch in range(epochs):
    print(f'Starting epoch {epoch + 1}/{epochs}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        if step % 100 == 0:
            print(f'Epoch {epoch + 1}, Step {step}, Loss: {loss_value.numpy()}')

# Evaluation
accuracy_metric = SparseCategoricalAccuracy()

for x_batch_test, y_batch_test in test_dataset:
    test_logits = model(x_batch_test, training=False)
    accuracy_metric.update_state(y_batch_test, test_logits)

test_accuracy = accuracy_metric.result().numpy()
print(f'Test Accuracy: {test_accuracy}')
