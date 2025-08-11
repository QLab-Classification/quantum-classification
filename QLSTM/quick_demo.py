#!/usr/bin/env python3
"""
Quick Demo of QLSTM with Progress Bars
This script runs a faster version for demonstration purposes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers.draft import QLSTM, ClassicalLSTM, StockPricePredictor
from helpers.data_sources import StockDataCollector
from tqdm import tqdm

def quick_demo():
    """Run a quick demo with progress bars"""
    print("⚡ Quick QLSTM Demo with Progress Bars")
    print("=" * 50)
    
    # Generate sample data quickly
    print("\n📊 Generating sample data...")
    collector = StockDataCollector()
    data = collector.get_sample_data()
    data_file = "data/sample_stock_data.csv"
    
    # Initialize predictor
    print(f"\n🧠 Initializing models...")
    predictor = StockPricePredictor(sequence_length=5)  # Shorter sequence for speed
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data(data_file)
    
    print(f"✅ Data loaded: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    
    # Initialize models
    classical_lstm = ClassicalLSTM(input_size=1, hidden_size=7, output_size=1)
    qlstm = QLSTM(input_size=1, hidden_size=7, output_size=1, n_qubits=7)
    
    # Count parameters
    classical_params = sum(p.numel() for p in classical_lstm.parameters())
    qlstm_params = sum(p.numel() for p in qlstm.parameters())
    
    print(f"\n📊 Model Parameters:")
    print(f"   Classical LSTM: {classical_params}")
    print(f"   QLSTM: {qlstm_params}")
    print(f"   Reduction: {((classical_params - qlstm_params) / classical_params) * 100:.1f}%")
    
    # Train Classical LSTM (quick)
    print(f"\n🎯 Training Classical LSTM (5 epochs)...")
    classical_losses = predictor.train_model(classical_lstm, X_train, y_train, epochs=5, lr=0.01)
    
    # Train QLSTM (quick)
    print(f"\n⚛️  Training QLSTM (5 epochs)...")
    qlstm_losses = predictor.train_model(qlstm, X_train, y_train, epochs=5, lr=0.01)
    
    # Evaluate models
    print(f"\n📈 Evaluating models...")
    classical_rmse, classical_accuracy = predictor.evaluate_model(classical_lstm, X_test, y_test)
    qlstm_rmse, qlstm_accuracy = predictor.evaluate_model(qlstm, X_test, y_test)
    
    # Print results
    print(f"\n🏆 Quick Results:")
    print("-" * 30)
    print(f"Classical LSTM:")
    print(f"  RMSE: {classical_rmse:.4f}")
    print(f"  Accuracy: {classical_accuracy:.4f}")
    print(f"\nQLSTM:")
    print(f"  RMSE: {qlstm_rmse:.4f}")
    print(f"  Accuracy: {qlstm_accuracy:.4f}")
    
    # Calculate improvements
    rmse_improvement = ((classical_rmse - qlstm_rmse) / classical_rmse) * 100
    accuracy_improvement = ((qlstm_accuracy - classical_accuracy) / classical_accuracy) * 100
    
    print(f"\n📊 Performance:")
    print(f"  RMSE Improvement: {rmse_improvement:.2f}%")
    print(f"  Accuracy Improvement: {accuracy_improvement:.2f}%")
    
    # Plot training losses
    print(f"\n📊 Plotting training losses...")
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(classical_losses, 'b-', label='Classical LSTM')
    plt.title('Classical LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(qlstm_losses, 'r-', label='QLSTM')
    plt.title('QLSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot combined predictions comparison
    print(f"\n📊 Generating combined prediction comparison plot...")
    from helpers.plot_predictions import plot_combined_predictions
    plot_combined_predictions(classical_lstm, qlstm, X_test, y_test, predictor.scaler)
    
    print(f"\n✅ Quick demo completed!")
    print(f"\n💡 For full training:")
    print(f"   - Increase epochs to 50")
    print(f"   - Use longer sequence length")
    print(f"   - Run with real data from Yahoo Finance")

if __name__ == "__main__":
    quick_demo() 