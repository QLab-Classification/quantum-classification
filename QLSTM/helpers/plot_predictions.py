#!/usr/bin/env python3
"""
Prediction Plotting for QLSTM Models
This script plots actual vs predicted stock prices for trained models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers.draft import QLSTM, ClassicalLSTM, StockPricePredictor
from helpers.data_sources import StockDataCollector
from sklearn.metrics import mean_squared_error

def plot_model_predictions():
    """Plot predictions for both Classical LSTM and QLSTM models"""
    print("📊 QLSTM Prediction Visualization")
    print("=" * 40)
    
    # Initialize data collector and predictor
    collector = StockDataCollector()
    predictor = StockPricePredictor(sequence_length=10)
    
    # Load data
    print("📈 Loading data...")
    data = collector.get_sample_data()
    data_file = "data/sample_stock_data.csv"
    
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data(data_file)
    print(f"✅ Data loaded: {X_test.shape[0]} test samples")
    
    # Initialize models
    print("\n🧠 Initializing models...")
    classical_lstm = ClassicalLSTM(input_size=1, hidden_size=7, output_size=1)
    qlstm = QLSTM(input_size=1, hidden_size=7, output_size=1, n_qubits=7)
    
    # Quick training for demonstration (fewer epochs)
    print("\n🎯 Training Classical LSTM (3 epochs)...")
    classical_losses = predictor.train_model(classical_lstm, X_train, y_train, epochs=3, lr=0.01)
    
    print("\n⚛️  Training QLSTM (3 epochs)...")
    qlstm_losses = predictor.train_model(qlstm, X_train, y_train, epochs=3, lr=0.01)
    
    # Plot predictions
    print("\n📊 Generating prediction plots...")
    
    # Classical LSTM predictions
    predictor.plot_predictions(classical_lstm, X_test, y_test, "Classical LSTM", predictor.scaler)
    
    # QLSTM predictions
    predictor.plot_predictions(qlstm, X_test, y_test, "QLSTM", predictor.scaler)
    
    # Combined comparison plot
    print("\n📊 Generating combined comparison plot...")
    plot_combined_predictions(classical_lstm, qlstm, X_test, y_test, predictor.scaler)
    
    print("\n✅ Prediction visualization completed!")

def plot_combined_predictions(classical_model, qlstm_model, X_test, y_test, scaler):
    """Create a combined plot showing both models' predictions"""
    
    # Get predictions from both models
    classical_model.eval()
    qlstm_model.eval()
    
    with torch.no_grad():
        classical_pred = classical_model(X_test)
        qlstm_pred = qlstm_model(X_test)
        
        # Handle output shapes
        if len(classical_pred.shape) > 2:
            classical_pred = classical_pred[:, -1, :]
        if len(qlstm_pred.shape) > 2:
            qlstm_pred = qlstm_pred[:, -1, :]
        
        # Ensure shapes match
        if classical_pred.shape != y_test.shape:
            classical_pred = classical_pred[:, :y_test.shape[1]]
        if qlstm_pred.shape != y_test.shape:
            qlstm_pred = qlstm_pred[:, :y_test.shape[1]]
        
        # Convert to numpy and denormalize
        y_true_np = y_test.numpy().flatten()
        classical_pred_np = classical_pred.numpy().flatten()
        qlstm_pred_np = qlstm_pred.numpy().flatten()
        
        if scaler is not None:
            y_true_np = scaler.inverse_transform(y_true_np.reshape(-1, 1)).flatten()
            classical_pred_np = scaler.inverse_transform(classical_pred_np.reshape(-1, 1)).flatten()
            qlstm_pred_np = scaler.inverse_transform(qlstm_pred_np.reshape(-1, 1)).flatten()
            y_label = "Stock Price ($)"
        else:
            y_label = "Normalized Price"
        
        # Create combined plot
        plt.figure(figsize=(14, 8))
        
        # Main prediction plot
        plt.subplot(2, 1, 1)
        plt.plot(y_true_np, label='Actual', color='black', linewidth=3, alpha=0.9)
        plt.plot(classical_pred_np, label='Classical LSTM', color='blue', linewidth=2, alpha=0.8)
        plt.plot(qlstm_pred_np, label='QLSTM', color='red', linewidth=2, alpha=0.8)
        
        plt.title('Stock Price Prediction Comparison: Classical LSTM vs QLSTM')
        plt.xlabel('Time Steps')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add performance metrics
        classical_rmse = np.sqrt(mean_squared_error(y_true_np, classical_pred_np))
        qlstm_rmse = np.sqrt(mean_squared_error(y_true_np, qlstm_pred_np))
        
        plt.text(0.02, 0.98, f'Classical LSTM RMSE: {classical_rmse:.4f}\nQLSTM RMSE: {qlstm_rmse:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Error comparison plot
        plt.subplot(2, 1, 2)
        classical_error = np.abs(y_true_np - classical_pred_np)
        qlstm_error = np.abs(y_true_np - qlstm_pred_np)
        
        plt.plot(classical_error, label='Classical LSTM Error', color='blue', alpha=0.7)
        plt.plot(qlstm_error, label='QLSTM Error', color='red', alpha=0.7)
        
        plt.title('Prediction Error Comparison')
        plt.xlabel('Time Steps')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add error statistics
        avg_classical_error = np.mean(classical_error)
        avg_qlstm_error = np.mean(qlstm_error)
        improvement = ((avg_classical_error - avg_qlstm_error) / avg_classical_error) * 100
        
        plt.text(0.02, 0.98, f'Avg Classical Error: {avg_classical_error:.4f}\nAvg QLSTM Error: {avg_qlstm_error:.4f}\nImprovement: {improvement:.1f}%', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n📊 Combined Performance Summary:")
        print(f"   Classical LSTM RMSE: {classical_rmse:.4f}")
        print(f"   QLSTM RMSE: {qlstm_rmse:.4f}")
        print(f"   RMSE Improvement: {((classical_rmse - qlstm_rmse) / classical_rmse) * 100:.1f}%")
        print(f"   Average Error Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    plot_model_predictions()
