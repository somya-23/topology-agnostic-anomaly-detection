import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import io
import math
import time
import os
import math

# --- 1. GPU and System Optimization ---
# Setup device for GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# ==============================================================================
# 2. MODEL ARCHITECTURE (High-Fidelity Implementation)
# ==============================================================================

# --- Components for Transformer Module (as per Fig. 3) ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # This calculation is the core of the problem. Let's fix it.
        # We create a div_term for half the dimensions.
        half_d = d_model // 2
        div_term = torch.exp(torch.arange(half_d).float() * (-math.log(10000.0) / half_d))

        # Assign sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Assign cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CustomTransformerEncoderLayer(nn.Module):
    """ Implements the custom Transformer block from Figure 3 of the paper. """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Conv1D layers as per the paper's diagram
        self.conv1 = nn.Conv1d(d_model, dim_feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(dim_feedforward, d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            # Multi-Head Attention Path
            # The nn.MultiheadAttention layer CAN use these masks if provided
            src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout(src2)
            src = self.norm1(src)

            # Convolutional Path
            src2 = src.permute(1, 2, 0) 
            src2 = F.relu(self.conv1(src2))
            src2 = self.conv2(src2)
            src2 = src2.permute(2, 0, 1)
            
            src = src + self.dropout(src2)
            src = self.norm2(src)
            return src

# In your simba_model.py script

class TransformerModule(nn.Module):
    """ Implements the Time Series Transformer module from Figure 7. """
    def __init__(self, in_features, num_heads, num_layers, hidden_dim, model_dim=32): # Added model_dim
        super(TransformerModule, self).__init__()
        
        # --- FIX 1: Project input features (5) to a valid model dimension (32) ---
        self.input_projection = nn.Linear(in_features, model_dim)
        
        # Use the even model_dim for PositionalEncoding and the EncoderLayer
        self.pos_encoder = PositionalEncoding(model_dim)
        
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = model_dim # The output of this module is now model_dim

    def forward(self, src):
        seq_len, batch_size, num_nodes, in_features = src.shape[1], src.shape[0], src.shape[2], src.shape[3]
        
        src_reshaped = src.permute(1, 0, 2, 3).reshape(seq_len, batch_size * num_nodes, in_features)
        
        # Apply the projection
        src_projected = self.input_projection(src_reshaped)
        
        src_pos = self.pos_encoder(src_projected)
        output = self.transformer_encoder(src_pos)
        output_pooled = output.mean(dim=0)
        
        # Reshape to the new output dimension
        output_final = output_pooled.reshape(batch_size, num_nodes, self.output_dim)
        
        return output_final

# --- Components for Spatial Module (as per Fig. 8 and Eqs. 3-5) ---

class GraphLearningModule(nn.Module):
    """ Implements the Graph Learning (GL) module based on equations 3-5. """
    def __init__(self, num_nodes, embedding_dim, top_k, alpha=3.0):
        super(GraphLearningModule, self).__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.alpha = alpha
        self.embedding1 = nn.Embedding(num_nodes, embedding_dim)
        self.embedding2 = nn.Embedding(num_nodes, embedding_dim)
        self.theta1 = nn.Linear(embedding_dim, embedding_dim)
        self.theta2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, node_indices):
        e1 = self.embedding1(node_indices)
        e2 = self.embedding2(node_indices)
        
        m1 = torch.tanh(self.alpha * self.theta1(e1))
        m2 = torch.tanh(self.alpha * self.theta2(e2))
        
        adj_matrix = F.relu(torch.tanh(self.alpha * (torch.matmul(m1, m2.transpose(1, 2)) - torch.matmul(m2, m1.transpose(1, 2)))))
        
        if self.top_k is not None and self.top_k > 0:
            top_k_values, _ = torch.topk(adj_matrix, self.top_k, dim=-1)
            mask = adj_matrix >= top_k_values[..., -1].unsqueeze(-1)
            adj_matrix = adj_matrix * mask.float()
            
        return adj_matrix

class MixHopPropagationLayer(nn.Module):
    """ Implements the mix-hop propagation layer from Figure 8b. """
    def __init__(self, in_features, out_features, num_hops):
        super(MixHopPropagationLayer, self).__init__()
        self.num_hops = num_hops
        self.feature_selectors = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_hops + 1)])

    def forward(self, features, adj_matrix):
        hop_features = []
        current_hop_features = features
        for i in range(self.num_hops + 1):
            selected_features = self.feature_selectors[i](current_hop_features)
            hop_features.append(selected_features)
            if i < self.num_hops:
                current_hop_features = torch.bmm(adj_matrix, current_hop_features)
        aggregated_features = torch.stack(hop_features, dim=0).sum(dim=0)
        return aggregated_features

class GCLayer(nn.Module):
    """ Implements the Graph Convolution (GC) layer from Figure 8a. """
    def __init__(self, in_features, out_features, num_hops):
        super(GCLayer, self).__init__()
        self.outflow_prop = MixHopPropagationLayer(in_features, out_features, num_hops)
        self.inflow_prop = MixHopPropagationLayer(in_features, out_features, num_hops)

    def forward(self, features, adj_matrix):
        adj_matrix_t = adj_matrix.transpose(1, 2)
        outflow_features = self.outflow_prop(features, adj_matrix)
        inflow_features = self.inflow_prop(features, adj_matrix_t)
        return outflow_features + inflow_features


# --- Full Simba Model (as per Fig. 7) ---

class Simba(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, seq_len,
                 gl_embedding_dim=10, top_k=5, gc_num_hops=2, gc_out_channels=32,
                 transformer_heads=4, transformer_layers=2, transformer_hidden=64,
                 final_ff_dim=128):
        super(Simba, self).__init__()
        self.num_nodes = num_nodes
        self.graph_learner = GraphLearningModule(num_nodes, gl_embedding_dim, top_k)
        self.gc_layer = GCLayer(in_features, gc_out_channels, gc_num_hops)
        
        # The TransformerModule now has its own internal dimension
        self.transformer_module = TransformerModule(in_features, transformer_heads, transformer_layers, transformer_hidden)
        
        # --- FIX 2: Calculate combined_dim using the transformer's TRUE output dimension ---
        transformer_out_features = self.transformer_module.output_dim 
        combined_dim = gc_out_channels + transformer_out_features # This will be 32 + 32 = 64
        
        self.feed_forward = nn.Sequential(
            # This will now correctly be nn.Linear(64 * 7, ...) which is nn.Linear(448, 128)
            nn.Linear(combined_dim * num_nodes, final_ff_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(final_ff_dim, out_features),
        )
        self.node_indices = nn.Parameter(torch.arange(0, num_nodes, dtype=torch.long), requires_grad=False)

    def forward(self, x):
        # Input shape: [Batch, SeqLen, NumNodes, Features]
        x = x.permute(0, 2, 1, 3) # -> [Batch, NumNodes, SeqLen, Features]

        # Spatial Path
        x_spatial = x[:, :, -1, :] # Use most recent time step [Batch, NumNodes, Features]
        batch_node_indices = self.node_indices.unsqueeze(0).expand(x.shape[0], -1)
        adj_matrix = self.graph_learner(batch_node_indices)
        spatial_embedding = self.gc_layer(x_spatial, adj_matrix) # [Batch, NumNodes, GC_Out_Channels]

        # Temporal Path
        x_temporal = x.permute(0, 2, 1, 3) # -> [Batch, SeqLen, NumNodes, Features]
        temporal_embedding = self.transformer_module(x_temporal) # [Batch, NumNodes, In_Features]

        # Combination and Classification
        combined_embedding = torch.cat([spatial_embedding, temporal_embedding], dim=2)
        
        # Flatten across all nodes for a single system-wide prediction
        combined_flat = combined_embedding.reshape(combined_embedding.shape[0], -1)
        
        output = self.feed_forward(combined_flat)
        return output

# ==============================================================================
# 3. DATA PROCESSING PIPELINE
# ==============================================================================
class RANAnomalyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_sequences(df, seq_length, feature_columns, target_column):
    sequences, labels = [], []
    num_nodes = df['BS_ID'].nunique()
    time_grouped = df.groupby('Time')
    timestamps = sorted(df['Time'].unique())
    
    for i in range(len(timestamps) - seq_length):
        window_timestamps = timestamps[i : i + seq_length]
        target_timestamp = timestamps[i + seq_length]
        
        input_sequence_df = df[df['Time'].isin(window_timestamps)]
        target_df = df[df['Time'] == target_timestamp]
        
        if len(input_sequence_df['Time'].unique()) < seq_length or len(target_df) < num_nodes:
            continue
            
        input_sequence_df = input_sequence_df.sort_values(by=['Time', 'BS_ID'])
        x_features = input_sequence_df[feature_columns].values.reshape(seq_length, num_nodes, len(feature_columns))
        
        target_df = target_df.sort_values(by='BS_ID')
        # Label for the whole system is the max fault type (0=Normal, 1/2=Fault)
        y_label = target_df[target_column].max()
        
        sequences.append(x_features)
        labels.append(y_label)
        
    return np.array(sequences), np.array(labels)

def prepare_data(data_file_path, seq_length, test_size=0.2, val_size=0.2):
    """
    Loads data from a file path, preprocesses, and splits it chronologically.
    """
    # --- Step 1: Load the CSV from the file path ---
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"FATAL ERROR: The data file was not found at the specified path: {data_file_path}")
        print("Please ensure the CSV file exists and the path is correct.")
        exit() # Exit the script if the data can't be found

    # --- Step 2: DEBUGGING - Check what was loaded ---
    print("\n--- Data Loading Debug Info ---")
    print(f"Successfully loaded {len(df)} rows from {data_file_path}")
    print("Columns found in the DataFrame:", df.columns.tolist())
    print("First 3 rows of the loaded data:\n", df.head(3))
    print("-----------------------------\n")

    # --- Step 3: Proceed with your logic ---
    # Now, when the script fails on the line below, the print statements above will have told you why.
    
    # Cleaning
    df.fillna(0, inplace=True)
    
    # Feature Selection
    feature_columns = ['Avg_RSRP_dBm', 'Avg_RSRQ_dB', 'Avg_SINR_dB', 'Avg_Throughput_bps', 'Avg_Distance_m']
    target_column = 'FaultType'
    
    # Chronological Split
    unique_timestamps = sorted(df['Time'].unique()) # The error happens here
    train_end_idx = int(len(unique_timestamps) * (1 - test_size - val_size))
    val_end_idx = int(len(unique_timestamps) * (1 - test_size))

    train_ts = unique_timestamps[:train_end_idx]
    val_ts = unique_timestamps[train_end_idx:val_end_idx]
    test_ts = unique_timestamps[val_end_idx:]

    train_df = df[df['Time'].isin(train_ts)]
    val_df = df[df['Time'].isin(val_ts)]
    test_df = df[df['Time'].isin(test_ts)]

    # Normalization (Fit on train ONLY)
    scaler = StandardScaler()
    train_df.loc[:, feature_columns] = scaler.fit_transform(train_df[feature_columns])
    val_df.loc[:, feature_columns] = scaler.transform(val_df[feature_columns])
    test_df.loc[:, feature_columns] = scaler.transform(test_df[feature_columns])
    
    # Create Sequences
    X_train, y_train = create_sequences(train_df, seq_length, feature_columns, target_column)
    X_val, y_val = create_sequences(val_df, seq_length, feature_columns, target_column)
    X_test, y_test = create_sequences(test_df, seq_length, feature_columns, target_column)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler

# ==============================================================================
# 4. TRAINING AND EVALUATION LOOP
# ==============================================================================

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # --- Load Your Data ---
    # This data is very small. For a real run, you need much more data
    # to create enough sequences for training, validation, and testing.
    # The current sample will likely result in empty validation/test sets.
    DATA_FILE_PATH = 'data/5g_uav_anomaly_dataset_integer_labels.csv'
    
    # --- Hyperparameters from Paper (Table III) ---
    SEQ_LENGTH = 5  # Paper uses 5, but our tiny dataset needs a smaller value.
    NUM_EPOCHS = 20 # Train for more epochs
    BATCH_SIZE = 2000 # Paper uses 2000, but our dataset is tiny.
    LEARNING_RATE = 0.0003
    
    # --- Data Preparation ---
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_data(DATA_FILE_PATH, SEQ_LENGTH, test_size=0.3, val_size=0.3)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at '{DATA_FILE_PATH}'. Please ensure the file exists in the same directory as the script.")
        exit()
    
    if len(X_train) == 0:
        raise ValueError("Training data is empty. The provided CSV data is too small for the chosen SEQ_LENGTH and train/val/test split.")

    train_dataset = RANAnomalyDataset(X_train, y_train)
    val_dataset = RANAnomalyDataset(X_val, y_val)
    # Use pin_memory and num_workers for faster data loading on GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2 if device.type == 'cuda' else 0)

    # --- Model Initialization ---
    num_nodes = X_train.shape[2]
    num_features = X_train.shape[3]
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    
    model = Simba(
        num_nodes=num_nodes,
        in_features=num_features,
        out_features=num_classes,
        seq_len=SEQ_LENGTH,
        top_k=5, # From paper
        gc_out_channels=32, # From paper
        transformer_heads=2, # Assuming based on common practice
        transformer_layers=2, # Assuming based on common practice
    ).to(device)

    # --- Loss Function with Class Weights ---
    class_counts = np.bincount(y_train, minlength=num_classes)
    weights = 1. / (class_counts + 1e-9) # Add epsilon to avoid division by zero
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    print(f"Using class weights for loss function: {weights.cpu().numpy()}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        
        if len(val_loader) > 0:
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, loss_fn, device)
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {elapsed_time:.2f}s | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        else:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | (Validation set is empty)")

    print("--- Training Finished ---")

    # 1. Define the directory and create it if it doesn't exist
    MODEL_DIR = 'model'
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Define file paths for the model and the scaler
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'simba_model.pth')
    SCALER_SAVE_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    
    # 3. Save the model's state dictionary
    # This saves all the learned weights and biases.
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # 4. Save the scaler object using pickle
    # This is crucial for correctly preprocessing the test data later.
    import pickle
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"\n--- Model and Scaler Saved ---")
    print(f"Model state dictionary saved to: {MODEL_SAVE_PATH}")
    print(f"Data scaler saved to: {SCALER_SAVE_PATH}")
    print("You can now use these files for inference on your test set.")
    # --- END OF SAVING BLOCK ---