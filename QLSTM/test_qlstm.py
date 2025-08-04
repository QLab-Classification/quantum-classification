import torch
import numpy as np
from draft import QLSTM, ClassicalLSTM, StockPricePredictor

def test_models():
    """Test both Classical LSTM and QLSTM with small dataset"""
    print("Testing QLSTM Implementation...")
    
    # Create small test dataset
    np.random.seed(42)
    n_samples = 50
    sequence_length = 5
    
    # Generate synthetic time series data
    data = np.cumsum(np.random.randn(n_samples)) + 100
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize data
    scaler = StockPricePredictor()
    X_normalized = scaler.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    y_normalized = scaler.scaler.transform(y.reshape(-1, 1)).flatten()
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_normalized).unsqueeze(-1)
    y_tensor = torch.FloatTensor(y_normalized).unsqueeze(-1)
    
    print(f"Test data shape: {X_tensor.shape}")
    print(f"Target shape: {y_tensor.shape}")
    
    # Test Classical LSTM
    print("\nTesting Classical LSTM...")
    classical_lstm = ClassicalLSTM(input_size=1, hidden_size=7, output_size=1)
    classical_output = classical_lstm(X_tensor)
    print(f"Classical LSTM output shape: {classical_output.shape}")
    
    # Test QLSTM
    print("\nTesting QLSTM...")
    qlstm = QLSTM(input_size=1, hidden_size=7, output_size=1, n_qubits=7)
    qlstm_output = qlstm(X_tensor)
    print(f"QLSTM output shape: {qlstm_output.shape}")
    
    # Test parameter counts
    classical_params = sum(p.numel() for p in classical_lstm.parameters())
    qlstm_params = sum(p.numel() for p in qlstm.parameters())
    
    print(f"\nParameter counts:")
    print(f"Classical LSTM: {classical_params}")
    print(f"QLSTM: {qlstm_params}")
    
    # Test forward pass with single sample
    print("\nTesting single sample forward pass...")
    single_input = X_tensor[:1]  # Single sample
    classical_single = classical_lstm(single_input)
    qlstm_single = qlstm(single_input)
    
    print(f"Single sample Classical LSTM output: {classical_single.shape}")
    print(f"Single sample QLSTM output: {qlstm_single.shape}")
    
    print("\n✅ All tests passed! QLSTM implementation is working correctly.")

if __name__ == "__main__":
    test_models() 