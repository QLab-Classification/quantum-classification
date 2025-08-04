# Quantum-Enhanced Long Short-Term Memory (QLSTM) for Stock Price Prediction

This implementation provides a hybrid quantum-classical model for stock price prediction, specifically the Quantum-Enhanced Long Short-Term Memory (QLSTM) model as described in the research paper.

## Overview

The QLSTM model integrates classical Long Short-Term Memory (LSTM) networks with Variational Quantum Circuits (VQCs) to achieve superior performance in time-series prediction tasks, particularly for stock price forecasting.

## Key Features

### Classical LSTM Component
- **Hidden Size**: 7 units
- **Parameters**: 288 total parameters
- **Architecture**: Standard LSTM with forget, input, candidate, and output gates
- **Training**: Adam optimizer, MSE loss, 50 epochs

### Quantum Component (QLSTM)
- **VQCs**: 6 Variational Quantum Circuits replacing classical LSTM gates
- **Qubits**: 7 qubits per VQC
- **Depth**: 2 variational layers
- **Rotations**: 3 rotations per qubit per layer
- **Parameters**: 254 total parameters (fewer than classical LSTM)
- **Encoding**: Angle encoding with arctan function

### Data Processing
- **Dataset**: Apple Inc. stock data (Jan 2022 - Jan 2023)
- **Preprocessing**: Normalization to [-1, 1] range
- **Split**: 70% training, 30% testing
- **Sequence Length**: 10 time steps for prediction

## Model Architecture

### QLSTM Gates (6 VQCs)
1. **VQC1**: Forget gate - determines what to discard from cell state
2. **VQC2**: Input gate - determines new information to add
3. **VQC3**: Update gate - generates candidate cell state
4. **VQC4**: Output gate - controls output based on cell state
5. **VQC5**: Cell to hidden state conversion
6. **VQC6**: Hidden to output conversion

### Quantum Circuit Structure
- **Data Encoding Layer**: Hadamard gates + angle encoding (arctan)
- **Variational Layer**: 2 layers with 3 rotations per qubit + CNOT entanglement
- **Measurement Layer**: Pauli Z expectation values

## Mathematical Formulations

### Classical LSTM
```
ft = σ(Wf [ht-1, xt] + bf)
it = σ(Wi[ht-1, xt] + bi)
C̃t = tanh(WC[ht-1, xt] + bC)
ct = ft ∗ ct-1 + it ∗ C̃t
ot = σ(Wo[ht-1, xt] + bo)
ht = ot ∗ tanh(ct)
```

### QLSTM
```
vt = [ht-1, xt]
ft = σ(VQC1(vt))
it = σ(VQC2(vt))
C̃t = tanh(VQC3(vt))
ct = ft ∗ ct-1 + it ∗ C̃t
ot = σ(VQC4(vt))
ht = VQC5(ot ∗ tanh(ct))
yt = VQC6(ht)
```

## Expected Performance

Based on the research paper:
- **RMSE Improvement**: ~50% reduction compared to classical models
- **Accuracy Improvement**: ~10% enhancement
- **Parameter Efficiency**: Fewer parameters (254 vs 288) with better performance
- **Convergence**: Faster adaptation to data patterns

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Run the complete implementation
python draft.py
```

## Output

The implementation will:
1. Load and preprocess stock data
2. Train both Classical LSTM and QLSTM models
3. Compare performance metrics (RMSE and Accuracy)
4. Display training loss plots
5. Show parameter count comparison

## Dependencies

- **PyTorch**: Deep learning framework
- **PennyLane**: Quantum machine learning library
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting
- **Scikit-learn**: Data preprocessing and metrics

## Key Implementation Details

### Quantum Circuit Design
- **Angle Encoding**: Uses arctan function for data encoding
- **Entanglement**: CNOT gates between adjacent qubits
- **Measurement**: Pauli Z expectation values for classical output

### Training Parameters
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 50

### Evaluation Metrics
- **RMSE**: Root Mean Square Error
- **Accuracy**: 1 - mean absolute error

## Research Context

This implementation follows the paper "A Hybrid Quantum-Classical Model for Stock Price Prediction" which demonstrates:
- Superior performance of quantum-enhanced models
- Parameter efficiency with fewer parameters
- Robust performance even with quantum noise
- Applicability to real-world financial time series

## Notes

- The implementation includes synthetic data generation for demonstration
- Real stock data can be provided via CSV file
- Quantum simulation is used (noiseless environment)
- Results may vary due to quantum circuit randomness 