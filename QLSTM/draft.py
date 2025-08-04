import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')

class ClassicalLSTM(nn.Module):
    """
    Classical LSTM implementation with 7 hidden units and 288 parameters
    """
    def __init__(self, input_size=1, hidden_size=7, output_size=1):
        super(ClassicalLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM gates: forget, input, candidate, output
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # Apply linear layer to all time steps to match QLSTM output
        outputs = []
        for t in range(lstm_out.shape[1]):
            output = self.linear(lstm_out[:, t, :])
            outputs.append(output.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class VariationalQuantumCircuit:
    """
    Variational Quantum Circuit with 7 qubits, depth 2, and 3 rotations
    """
    def __init__(self, n_qubits=7, depth=2):
        self.n_qubits = n_qubits
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
    def circuit(self, inputs, weights):
        """
        VQC with data encoding, variational layers, and measurement
        Following the figure's architecture with 4 qubits and specific entanglement pattern
        """
        # Data encoding layer using angle encoding
        for i in range(self.n_qubits):
            if i < len(inputs):
                # Angle encoding with arctan (as shown in figure)
                theta1 = np.arctan(inputs[i])
                theta2 = np.arctan(inputs[i]**2)
                qml.Hadamard(wires=i)
                qml.RY(theta1, wires=i)
                qml.RZ(theta2, wires=i)
            else:
                # Initialize unused qubits
                qml.Hadamard(wires=i)
        
        # First entanglement layer (as shown in figure)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        
        # Second entanglement layer (reversed pattern as shown in figure)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        
        # Variational layers with rotation gates R(α_i, β_i, γ_i)
        weight_idx = 0
        for d in range(self.depth):
            # Single qubit rotations (R(α_i, β_i, γ_i) as shown in figure)
            for i in range(self.n_qubits):
                qml.Rot(weights[weight_idx], weights[weight_idx+1], weights[weight_idx+2], wires=i)
                weight_idx += 3
        
        # Measurement layer - return Pauli Z expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

class QLSTM(nn.Module):
    """
    Quantum-Enhanced LSTM with 6 VQCs replacing classical gates
    """
    def __init__(self, input_size=1, hidden_size=7, output_size=1, n_qubits=7):
        super(QLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        
        # Initialize 6 VQCs for LSTM gates
        self.vqc1 = VariationalQuantumCircuit(n_qubits)  # Forget gate
        self.vqc2 = VariationalQuantumCircuit(n_qubits)  # Input gate
        self.vqc3 = VariationalQuantumCircuit(n_qubits)  # Update gate
        self.vqc4 = VariationalQuantumCircuit(n_qubits)  # Output gate
        self.vqc5 = VariationalQuantumCircuit(n_qubits)  # Cell to hidden
        self.vqc6 = VariationalQuantumCircuit(n_qubits)  # Hidden to output
        
        # Parameters for each VQC: 6 VQCs × 7 qubits × 2 depth × 3 rotations + 2 scaling = 254 params
        self.vqc_params = nn.Parameter(torch.randn(6, n_qubits * 2 * 3 + 2))
        
        # Scaling parameters for final output
        self.scale_params = nn.Parameter(torch.randn(2))
        
    def quantum_gate(self, vqc, inputs, params):
        """Execute quantum circuit and return classical output"""
        batch_size = inputs.shape[0]
        outputs = []
        
        # Add progress bar for quantum operations (only for large batches)
        if batch_size > 10:
            pbar = tqdm(range(batch_size), desc="Quantum processing", leave=False)
        else:
            pbar = range(batch_size)
        
        for b in pbar:
            # Get single sample
            inputs_np = inputs[b].detach().numpy()
            params_np = params.detach().numpy()
            
            # Ensure inputs are 1D for quantum circuit
            if len(inputs_np.shape) > 1:
                inputs_np = inputs_np.flatten()
            
            # Execute quantum circuit
            @qml.qnode(vqc.dev)
            def circuit(inputs, weights):
                return vqc.circuit(inputs, weights)
            
            # Get quantum measurements
            measurements = circuit(inputs_np, params_np)
            outputs.append(measurements)
        
        return torch.tensor(outputs, dtype=torch.float32)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        
        for t in range(seq_len):
            # Concatenate hidden state and input (vt = [ht-1, xt])
            input_t = x[:, t, :].squeeze(-1) if x[:, t, :].dim() > 1 else x[:, t, :]
            if input_t.dim() == 1:
                input_t = input_t.unsqueeze(1)
            vt = torch.cat([h, input_t], dim=1)
            
            # VQC1: Forget gate (ft = σ(VQC1(vt)))
            ft = torch.sigmoid(self.quantum_gate(self.vqc1, vt, self.vqc_params[0]))
            
            # VQC2: Input gate (it = σ(VQC2(vt)))
            it = torch.sigmoid(self.quantum_gate(self.vqc2, vt, self.vqc_params[1]))
            
            # VQC3: Candidate cell state (C̃t = tanh(VQC3(vt)))
            Ct_tilde = torch.tanh(self.quantum_gate(self.vqc3, vt, self.vqc_params[2]))
            
            # Update cell state (ct = ft ∗ ct-1 + it ∗ C̃t)
            c = ft * c + it * Ct_tilde
            
            # VQC4: Output gate (ot = σ(VQC4(vt)))
            ot = torch.sigmoid(self.quantum_gate(self.vqc4, vt, self.vqc_params[3]))
            
            # Hidden state update (ht = VQC5(ot ∗ tanh(ct)))
            ht = self.quantum_gate(self.vqc5, ot * torch.tanh(c), self.vqc_params[4])
            
            # Final output (yt+1 = VQC6(ht))
            yt_plus_1 = self.quantum_gate(self.vqc6, ht, self.vqc_params[5])
            
            # Apply scaling parameters
            yt_plus_1 = self.scale_params[0] * yt_plus_1 + self.scale_params[1]
            
            # Take the first element as the output
            yt_plus_1 = yt_plus_1[:, 0].unsqueeze(1)
            
            outputs.append(yt_plus_1)
            h = ht
        
        # Return predictions for all time steps (matching the figure's recurrent structure)
        return torch.cat(outputs, dim=1)

class StockPricePredictor:
    """
    Complete stock price prediction system with data preprocessing and model training
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def load_and_preprocess_data(self, file_path=None):
        """
        Load and preprocess stock data
        Supports real data from CSV files or generates synthetic data for demonstration
        """
        if file_path and os.path.exists(file_path):
            # Load actual data from CSV file
            print(f"📊 Loading real data from {file_path}")
            data = pd.read_csv(file_path)
            
            # Handle different column names
            if 'Close' in data.columns:
                prices = data['Close'].values
            elif 'close' in data.columns:
                prices = data['close'].values
            else:
                print("⚠️  No 'Close' column found. Using first numeric column.")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    prices = data[numeric_cols[0]].values
                else:
                    raise ValueError("No numeric columns found in data file")
            
            print(f"✅ Loaded {len(prices)} data points")
            
        else:
            # Generate synthetic Apple-like stock data for demonstration
            print("📊 Generating synthetic data for demonstration...")
            np.random.seed(42)
            dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
            dates = dates[dates.weekday < 5]  # Weekdays only
            
            # Generate realistic stock price movements
            initial_price = 150.0
            prices = [initial_price]
            for i in range(len(dates) - 1):
                # Random walk with trend
                change = np.random.normal(0, 2) + 0.1  # Small upward trend
                new_price = prices[-1] + change
                prices.append(max(new_price, 50))  # Minimum price floor
            
            prices = np.array(prices)
            print(f"✅ Generated {len(prices)} synthetic data points")
        
        # Normalize data to [-1, 1] range
        prices_normalized = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Create sequences for time series prediction
        X, y = [], []
        for i in range(len(prices_normalized) - self.sequence_length):
            X.append(prices_normalized[i:i + self.sequence_length])
            y.append(prices_normalized[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data: 70% training, 30% testing
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train, y_train, epochs=50, lr=0.01):
        """Train the model using Adam optimizer and MSE loss"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        
        # Add progress bar
        pbar = tqdm(range(epochs), desc=f"Training {model.__class__.__name__}")
        
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train)
            
            # Handle different output shapes
            if len(outputs.shape) > 2:
                # If outputs have multiple time steps, use the last one
                outputs = outputs[:, -1, :]
            
            loss = criterion(outputs, y_train)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Update progress bar with loss
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return train_losses
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance using RMSE and Accuracy metrics"""
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            
            # For time series prediction, we typically use the last prediction
            # or average across all predictions
            if len(predictions.shape) > 2:
                # If predictions have multiple time steps, use the last one
                predictions = predictions[:, -1, :]
            elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
                # If predictions have multiple columns, use the last one
                predictions = predictions[:, -1:]
            
            # Ensure shapes match
            if predictions.shape != y_test.shape:
                predictions = predictions[:, :y_test.shape[1]]
            
            # Calculate RMSE
            mse = mean_squared_error(y_test.numpy(), predictions.numpy())
            rmse = np.sqrt(mse)
            
            # Calculate Accuracy (as defined in the paper)
            accuracy = 1 - np.mean(np.abs(y_test.numpy() - predictions.numpy()))
            
            return rmse, accuracy
    
    def plot_results(self, train_losses, model_name):
        """Plot training loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title(f'{model_name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def main():
    """Main execution function"""
    print("=== Hybrid Quantum-Classical Model for Stock Price Prediction ===")
    print("Implementing QLSTM vs Classical LSTM comparison\n")
    
    # Initialize predictor
    predictor = StockPricePredictor(sequence_length=10)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data()
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Train Classical LSTM
    print("\n=== Training Classical LSTM ===")
    classical_lstm = ClassicalLSTM(input_size=1, hidden_size=7, output_size=1)
    classical_losses = predictor.train_model(classical_lstm, X_train, y_train)
    
    # Evaluate Classical LSTM
    classical_rmse, classical_accuracy = predictor.evaluate_model(classical_lstm, X_test, y_test)
    print(f"Classical LSTM - RMSE: {classical_rmse:.4f}, Accuracy: {classical_accuracy:.4f}")
    
    # Train QLSTM
    print("\n=== Training QLSTM ===")
    qlstm = QLSTM(input_size=1, hidden_size=7, output_size=1, n_qubits=7)
    qlstm_losses = predictor.train_model(qlstm, X_train, y_train)
    
    # Evaluate QLSTM
    qlstm_rmse, qlstm_accuracy = predictor.evaluate_model(qlstm, X_test, y_test)
    print(f"QLSTM - RMSE: {qlstm_rmse:.4f}, Accuracy: {qlstm_accuracy:.4f}")
    
    # Print comparison
    print("\n=== Results Comparison ===")
    print(f"Classical LSTM - RMSE: {classical_rmse:.4f}, Accuracy: {classical_accuracy:.4f}")
    print(f"QLSTM - RMSE: {qlstm_rmse:.4f}, Accuracy: {qlstm_accuracy:.4f}")
    
    rmse_improvement = ((classical_rmse - qlstm_rmse) / classical_rmse) * 100
    accuracy_improvement = ((qlstm_accuracy - classical_accuracy) / classical_accuracy) * 100
    
    print(f"\nQLSTM RMSE Improvement: {rmse_improvement:.2f}%")
    print(f"QLSTM Accuracy Improvement: {accuracy_improvement:.2f}%")
    
    # Plot training losses
    predictor.plot_results(classical_losses, "Classical LSTM")
    predictor.plot_results(qlstm_losses, "QLSTM")
    
    # Parameter count comparison
    classical_params = sum(p.numel() for p in classical_lstm.parameters())
    qlstm_params = sum(p.numel() for p in qlstm.parameters())
    
    print(f"\n=== Parameter Count ===")
    print(f"Classical LSTM parameters: {classical_params}")
    print(f"QLSTM parameters: {qlstm_params}")
    print(f"Parameter reduction: {((classical_params - qlstm_params) / classical_params) * 100:.1f}%")

if __name__ == "__main__":
    import os
    main()
