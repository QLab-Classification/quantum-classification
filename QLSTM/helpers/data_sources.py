#!/usr/bin/env python3
"""
Data Sources for QLSTM Stock Price Prediction
This module provides various ways to obtain real stock price data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os

class StockDataCollector:
    """
    Collects real stock price data from various sources
    """
    
    def __init__(self):
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_yahoo_finance_data(self, symbol="AAPL", start_date="2022-01-01", end_date="2023-01-01"):
        """
        Get stock data from Yahoo Finance (free, reliable)
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        
        Returns:
            pd.DataFrame: Stock data with Date, Open, High, Low, Close, Volume
        """
        try:
            print(f"📊 Downloading {symbol} data from Yahoo Finance...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Save to CSV
            filename = f"{self.data_dir}/{symbol}_{start_date}_{end_date}.csv"
            data.to_csv(filename, index=False)
            print(f"✅ Data saved to {filename}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def get_alpha_vantage_data(self, symbol="AAPL", api_key=None):
        """
        Get stock data from Alpha Vantage (requires free API key)
        
        Args:
            symbol (str): Stock symbol
            api_key (str): Alpha Vantage API key (get from https://www.alphavantage.co/)
        
        Returns:
            pd.DataFrame: Stock data
        """
        if not api_key:
            print("⚠️  Alpha Vantage requires an API key. Get one from https://www.alphavantage.co/")
            return None
        
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
            response = requests.get(url)
            data = response.json()
            
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                df.index = pd.to_datetime(df.index)
                df.columns = ["Open", "High", "Low", "Close", "Volume"]
                df = df.astype(float)
                df = df.reset_index()
                df.columns = ["Date"] + list(df.columns[1:])
                
                filename = f"{self.data_dir}/{symbol}_alphavantage.csv"
                df.to_csv(filename, index=False)
                print(f"✅ Data saved to {filename}")
                return df
            else:
                print(f"❌ Error: {data.get('Note', 'Unknown error')}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            return None
    
    def get_csv_data(self, filepath):
        """
        Load data from a local CSV file
        
        Args:
            filepath (str): Path to CSV file
        
        Returns:
            pd.DataFrame: Stock data
        """
        try:
            data = pd.read_csv(filepath)
            print(f"✅ Loaded data from {filepath}")
            return data
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None
    
    def get_sample_data(self):
        """
        Get sample data for testing (if no real data available)
        """
        print("📊 Generating sample stock data for testing...")
        
        # Generate realistic stock data
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2023-01-01', freq='D')
        dates = dates[dates.weekday < 5]  # Weekdays only
        
        # Generate realistic stock price movements
        initial_price = 150.0
        prices = [initial_price]
        for i in range(len(dates) - 1):
            # Random walk with trend and volatility
            change = np.random.normal(0, 2) + 0.1  # Small upward trend
            new_price = prices[-1] + change
            prices.append(max(new_price, 50))  # Minimum price floor
        
        # Create DataFrame
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p + np.random.uniform(0, 3) for p in prices],
            'Low': [p - np.random.uniform(0, 3) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(prices))
        })
        
        # Ensure High >= Open, Close and Low <= Open, Close
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        filename = f"{self.data_dir}/sample_stock_data.csv"
        data.to_csv(filename, index=False)
        print(f"✅ Sample data saved to {filename}")
        
        return data

def main():
    """Demonstrate different data collection methods"""
    collector = StockDataCollector()
    
    print("🚀 Stock Data Collection for QLSTM")
    print("=" * 50)
    
    # Method 1: Yahoo Finance (Recommended - Free, No API key needed)
    print("\n1️⃣ Yahoo Finance (Recommended)")
    print("   - Free, no API key required")
    print("   - Reliable and comprehensive data")
    print("   - Easy to use")
    
    # Uncomment to download real data
    # data = collector.get_yahoo_finance_data("AAPL", "2022-01-01", "2023-01-01")
    
    # Method 2: Alpha Vantage
    print("\n2️⃣ Alpha Vantage")
    print("   - Requires free API key")
    print("   - Get key from: https://www.alphavantage.co/")
    print("   - More detailed data available")
    
    # Uncomment and add your API key
    # data = collector.get_alpha_vantage_data("AAPL", "YOUR_API_KEY_HERE")
    
    # Method 3: Local CSV file
    print("\n3️⃣ Local CSV file")
    print("   - Use your own data file")
    print("   - Format: Date, Open, High, Low, Close, Volume")
    
    # Method 4: Sample data
    print("\n4️⃣ Sample data (for testing)")
    data = collector.get_sample_data()
    
    print(f"\n📈 Data shape: {data.shape}")
    print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"💰 Price range: ${data['Close'].min():.2f} to ${data['Close'].max():.2f}")
    
    print("\n✅ Data collection complete!")
    print("\n💡 To use real data in QLSTM:")
    print("   1. Download data using one of the methods above")
    print("   2. Update the file path in draft.py")
    print("   3. Run the QLSTM implementation")

if __name__ == "__main__":
    main() 