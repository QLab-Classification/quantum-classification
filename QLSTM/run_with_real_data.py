#!/usr/bin/env python3
"""
Run QLSTM with Real Stock Data
This script demonstrates how to use real stock price data with the QLSTM implementation
"""

import os
import matplotlib.pyplot as plt
from helpers.draft import QLSTM, ClassicalLSTM, StockPricePredictor
from helpers.data_sources import StockDataCollector

def run_with_real_data():
    """Run QLSTM with real stock data"""
    print("🚀 QLSTM with Real Stock Data")
    print("=" * 40)
    
    # # Initialize data collector
    # collector = StockDataCollector()
    
    # # Option 1: Download real data from Yahoo Finance
    # print("\n📊 Option 1: Download real data from Yahoo Finance")
    # print("   This will download Apple stock data for 2022-2023")
    
    # # Uncomment the line below to download real data
    # # data = collector.get_yahoo_finance_data("AAPL", "2022-01-01", "2023-01-01")
    # # data_file = "data/AAPL_2022-01-01_2023-01-01.csv"
    
    # # Option 2: Use sample data (for demonstration)
    # print("\n📊 Option 2: Using sample data (for demonstration)")
    # data = collector.get_sample_data()
    # data_file = "data/sample_stock_data.csv"
    
    # Option 3: Use your own CSV file
    data_file = "/Users/kanjonavosabud/Documents/quantum/quantum-classification/QLSTM/data/AAPL_2022-01-01_2023-01-01.csv"
    
    # Initialize predictor with real data
    print(f"\n🧠 Initializing QLSTM with data from: {data_file}")
    predictor = StockPricePredictor(sequence_length=10)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data(data_file)
    
    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Sequence length: {X_train.shape[1]}")
    
    # Initialize models
    print("\n⚛️  Initializing models...")
    classical_lstm = ClassicalLSTM(input_size=1, hidden_size=7, output_size=1)
    qlstm = QLSTM(input_size=1, hidden_size=7, output_size=1, n_qubits=7)
    
    # Count parameters
    classical_params = sum(p.numel() for p in classical_lstm.parameters())
    qlstm_params = sum(p.numel() for p in qlstm.parameters())
    
    print(f"   Classical LSTM parameters: {classical_params}")
    print(f"   QLSTM parameters: {qlstm_params}")
    print(f"   Parameter reduction: {((classical_params - qlstm_params) / classical_params) * 100:.1f}%")
    
    # Train models (shorter training for demo)
    print("\n🎯 Training Classical LSTM...")
    classical_losses = predictor.train_model(classical_lstm, X_train, y_train, epochs=3, lr=0.01)
    
    print("\n⚛️  Training QLSTM...")
    qlstm_losses = predictor.train_model(qlstm, X_train, y_train, epochs=3, lr=0.01)
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    classical_rmse, classical_accuracy = predictor.evaluate_model(classical_lstm, X_test, y_test)
    qlstm_rmse, qlstm_accuracy = predictor.evaluate_model(qlstm, X_test, y_test)
    
    # Print results
    print("\n🏆 Results with Real Data:")
    print("-" * 40)
    print(f"Classical LSTM:")
    print(f"  RMSE: {classical_rmse:.4f}")
    print(f"  Accuracy: {classical_accuracy:.4f}")
    print(f"\nQLSTM:")
    print(f"  RMSE: {qlstm_rmse:.4f}")
    print(f"  Accuracy: {qlstm_accuracy:.4f}")
    
    # Calculate improvements
    rmse_improvement = ((classical_rmse - qlstm_rmse) / classical_rmse) * 100
    accuracy_improvement = ((qlstm_accuracy - classical_accuracy) / classical_accuracy) * 100

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

    # Plot combined predictions comparison
    print(f"\n📊 Generating combined prediction comparison plot...")
    from helpers.plot_predictions import plot_combined_predictions
    plot_combined_predictions(classical_lstm, qlstm, X_test, y_test, predictor.scaler)

    print(f"\n📊 Performance Improvements:")
    print(f"  RMSE Improvement: {rmse_improvement:.2f}%")
    print(f"  Accuracy Improvement: {accuracy_improvement:.2f}%")
    
    print("\n✅ Real data experiment completed!")
    print("\n💡 To use different data sources:")
    print("   1. Yahoo Finance: Uncomment line in data_sources.py")
    print("   2. Alpha Vantage: Get API key and use get_alpha_vantage_data()")
    print("   3. Your own CSV: Update data_file path above")

def download_real_data():
    """Download real stock data for use"""
    print("📊 Downloading Real Stock Data")
    print("=" * 40)
    
    collector = StockDataCollector()
    
    # Popular stocks to download
    stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
    
    for symbol in stocks:
        print(f"\n📈 Downloading {symbol} data...")
        data = collector.get_yahoo_finance_data(symbol, "2022-01-01", "2023-01-01")
        if data is not None:
            print(f"✅ {symbol} data downloaded successfully!")
            print(f"   Shape: {data.shape}")
            print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        else:
            print(f"❌ Failed to download {symbol} data")
    
    print("\n✅ Data download complete!")
    print("💡 You can now use these files with run_with_real_data.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_real_data()
    else:
        run_with_real_data() 