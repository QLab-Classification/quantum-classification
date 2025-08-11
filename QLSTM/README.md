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
- **Parameters**: 266 total parameters (fewer than classical LSTM)
- **Encoding**: Angle encoding with arctan function

### Data Processing
- **Dataset**: Multiple stock datasets including Apple, Microsoft, Google, Tesla, Amazon
- **Preprocessing**: Normalization to [-1, 1] range
- **Split**: 70% training, 30% testing
- **Sequence Length**: Configurable time steps for prediction

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
- **Parameter Efficiency**: Fewer parameters (266 vs 288) with better performance
- **Convergence**: Faster adaptation to data patterns

## Installation

```bash
pip install -r requirements.txt
```

## File Structure

```
QLSTM/
├── helpers/                          # Core implementation files
│   ├── draft.py                     # Main QLSTM implementation
│   ├── data_sources.py              # Data collection utilities
│   └── plot_predictions.py          # Prediction visualization
├── docs/                            # Documentation and research
│   ├── ARCHITECTURE_COMPARISON.md   # Paper alignment verification
│   ├── IMPLEMENTATION_SUMMARY.md    # Technical implementation details
│   └── QLSTM paper1.pdf            # Research paper reference
├── data/                            # Stock datasets
│   ├── AAPL_2022-01-01_2023-01-01.csv    # Apple stock data
│   ├── MSFT_2022-01-01_2023-01-01.csv    # Microsoft stock data
│   ├── GOOGL_2022-01-01_2023-01-01.csv   # Google stock data
│   ├── TSLA_2022-01-01_2023-01-01.csv    # Tesla stock data
│   ├── AMZN_2022-01-01_2023-01-01.csv    # Amazon stock data
│   └── sample_stock_data.csv              # Synthetic data for testing
├── run_with_real_data.py            # Execute with real stock data
├── quick_demo.py                    # Fast demonstration with progress bars
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## Usage

### Quick Start
```bash
# Run fast demonstration (5 epochs, with progress bars)
python quick_demo.py

# Run with real stock data
python run_with_real_data.py

# Run with specific data source
python run_with_real_data.py download
```

### Core Implementation
```python
# Import from helpers directory
from helpers.draft import QLSTM, ClassicalLSTM, StockPricePredictor
from helpers.data_sources import StockDataCollector
from helpers.plot_predictions import plot_model_predictions

# Initialize models
classical_lstm = ClassicalLSTM()
qlstm = QLSTM()

# Load and preprocess data
predictor = StockPricePredictor()
X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data()

# Train and evaluate
classical_losses = predictor.train_model(classical_lstm, X_train, y_train)
qlstm_losses = predictor.train_model(qlstm, X_train, y_train)

# Plot predictions
predictor.plot_predictions(classical_lstm, X_test, y_test, "Classical LSTM", predictor.scaler)
predictor.plot_predictions(qlstm, X_test, y_test, "QLSTM", predictor.scaler)
```

## Data Sources

### Available Stock Datasets
- **AAPL**: Apple Inc. (2022-2023)
- **MSFT**: Microsoft Corporation (2022-2023)
- **GOOGL**: Alphabet Inc. (Google) (2022-2023)
- **TSLA**: Tesla Inc. (2022-2023)
- **AMZN**: Amazon.com Inc. (2022-2023)

### Data Collection
```python
from helpers.data_sources import StockDataCollector

collector = StockDataCollector()

# Download real stock data
data = collector.get_yahoo_finance_data("AAPL", "2022-01-01", "2023-01-01")

# Use existing CSV files
data = collector.get_csv_data("data/AAPL_2022-01-01_2023-01-01.csv")
```

## Output

The implementation provides:
1. **Training Progress**: Real-time progress bars with loss tracking
2. **Performance Metrics**: RMSE and Accuracy comparisons
3. **Prediction Plots**: Actual vs predicted stock prices in dollar amounts
4. **Training Loss Plots**: Learning curves for both models
5. **Parameter Analysis**: Model complexity comparison

## Dependencies

- **PyTorch**: Deep learning framework
- **PennyLane**: Quantum machine learning library
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting and visualization
- **Scikit-learn**: Data preprocessing and metrics
- **yfinance**: Yahoo Finance data collection
- **tqdm**: Progress bars for training
- **requests**: API data collection

## Key Implementation Details

### Quantum Circuit Design
- **Angle Encoding**: Uses arctan function for data encoding
- **Entanglement**: CNOT gates between adjacent qubits
- **Measurement**: Pauli Z expectation values for classical output
- **Progress Tracking**: Real-time training progress with tqdm

### Training Parameters
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: Configurable (3-50 for demos, 50 for full training)

### Evaluation Metrics
- **RMSE**: Root Mean Square Error (in dollar amounts for real data)
- **Accuracy**: 1 - mean absolute error
- **Visualization**: Professional prediction plots with performance metrics

## Research Context

This implementation follows the paper "A Hybrid Quantum-Classical Model for Stock Price Prediction" which demonstrates:
- Superior performance of quantum-enhanced models
- Parameter efficiency with fewer parameters
- Robust performance even with quantum noise
- Applicability to real-world financial time series

## Advanced Features

### Progress Tracking
- **Training Progress**: Real-time progress bars for both models
- **Quantum Processing**: Progress tracking for quantum circuit operations
- **Performance Monitoring**: Live loss updates during training

### Data Handling
- **Multiple Sources**: Yahoo Finance, CSV files, synthetic data
- **Automatic Preprocessing**: Normalization and sequence creation
- **Real-time Download**: Live stock data collection
- **Flexible Formats**: Support for various data file structures

### Visualization
- **Prediction Plots**: Actual vs predicted with dollar amounts
- **Training Curves**: Learning progress visualization
- **Performance Metrics**: RMSE and accuracy display
- **Combined Comparisons**: Side-by-side model analysis

## Notes

- **Real Data Support**: Includes actual stock data from major companies
- **Progress Bars**: Training progress visible with tqdm integration
- **Dollar Values**: Predictions shown in actual stock prices ($)
- **Quantum Simulation**: Uses PennyLane for quantum circuit simulation
- **Results Variation**: May vary due to quantum circuit randomness 