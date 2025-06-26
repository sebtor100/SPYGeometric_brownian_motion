# AI-Powered Stock Analysis & Forecast

This project presents a sophisticated framework for stock market analysis and forecasting, combining deep learning with classical financial modeling. It leverages a hybrid **LSTM-Attention** model to predict not just the future price of a stock, but also the underlying parameters of its stochastic process—**drift** and **volatility**.

These AI-predicted parameters are then fed into a **Geometric Brownian Motion (GBM) Monte Carlo simulation** to generate thousands of possible future price paths. This provides a rich, probabilistic forecast that is more insightful than a single point prediction.

The framework is designed to be a powerful tool for quantitative analysts, traders, and data scientists interested in applying modern machine learning to financial markets.


*Example output showing an AI-enhanced vs. a traditional forecast for SPY.*

## 🚀 Features

- **Hybrid Deep Learning Model:** An `EnhancedStockPredictor` class built with PyTorch, featuring:
  - **LSTM Layers:** To capture long-term temporal dependencies in time-series data.
  - **Multi-Head Attention:** To allow the model to focus on the most relevant historical data points when making a prediction.
  - **Multiple Prediction Heads:** The model simultaneously predicts:
    1.  **Next-Day Price:** For direct forecasting.
    2.  **Stochastic Drift (μ):** The expected rate of return.
    3.  **Stochastic Volatility (σ):** The measure of price fluctuation.

- **Comprehensive Feature Engineering:** Automatically generates a rich set of features from raw price data, including:
  - Returns and Log Returns
  - Moving Averages (and price-to-MA ratios)
  - Volatility indicators (rolling standard deviation)
  - **RSI**, **MACD**, and **Bollinger Bands**
  - Volume-based indicators
  - Lag features

- **Monte Carlo Simulation:** Uses the AI-predicted drift and volatility to run an "enhanced" GBM simulation, providing a probability distribution of future prices.

- **Comparative Analysis:** The AI-enhanced forecast is always plotted against a **traditional forecast** that uses long-term historical averages. This clearly demonstrates the value and unique insights provided by the AI model.

- **Advanced Visualization:** Generates a comprehensive dashboard to visualize:
  - Model training performance and evaluation metrics.
  - A comparison of AI-driven vs. traditional price path forecasts.
  - The probability distribution of final prices.
  - A side-by-side comparison of risk percentiles (e.g., 5th, 25th, 50th, 75th, 95th).

## 🛠️ Technology Stack

- **Python 3.x**
- **PyTorch:** For building and training the deep learning model.
- **yfinance:** To download historical stock data from Yahoo Finance.
- **scikit-learn:** For data preprocessing (StandardScaler) and evaluation metrics.
- **pandas & NumPy:** For data manipulation and numerical operations.
- **Matplotlib:** For creating detailed visualizations.

## ⚙️ Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-stock-forecast.git
    cd ai-stock-forecast
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install torch yfinance scikit-learn pandas numpy matplotlib
    ```

## 📈 How to Run

The main script is designed to be run directly from the command line. You can easily change the stock ticker and other parameters within the `if __name__ == "__main__":` block.

1.  Open the Python script (e.g., `main.py` or the name you've given it).
2.  Locate the main execution block at the bottom of the file.
3.  Change the `ticker` variable to the stock symbol you want to analyze (e.g., "AAPL", "GOOGL", "MSFT", "TSLA").

    ```python
    # MAIN EXECUTION BLOCK
    if __name__ == "__main__":
        # 1. Configuration: Change the ticker for the desired stock/ETF
        ticker = "SPY"  # <-- CHANGE THIS TO ANY TICKER
        
        try:
            # 2. Train the model
            model, scaler_X, scaler_y, enhanced_data, feature_columns = train_enhanced_model(
                ticker=ticker, 
                sequence_length=60, 
                epochs=50
            )
            
            # 3. Run analysis and visualization
            enhanced_analysis_and_visualization(
                ticker=ticker, 
                model=model, 
                # ... other parameters ...
                forecast_months=6 # <-- You can also change the forecast horizon
            )
            
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
    ```

4.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```

The script will first download the necessary data, then train the model (showing progress), and finally display the evaluation and forecast plots.

## 📊 Interpreting the Results

The script generates two main plot windows:

### 1. Model Evaluation Dashboard
This shows how well the model was trained.
- **Training Loss:** Should decrease over epochs, indicating the model is learning.
- **Actual vs. Predicted Prices:** Points should be close to the diagonal line, showing prediction accuracy on unseen data.
- **Distribution of Errors:** Should be centered around zero, indicating unbiased predictions.
- **Predicted Parameters:** Shows the dynamic drift and volatility the model learned to recognize from market data.

### 2. Forecast Analysis Dashboard
This is the main output for decision-making.
- **Forecast Comparison:** The **Red Cone (AI-Enhanced)** shows the forecast based on recent market conditions. The **Blue Cone (Traditional)** shows the forecast based on long-term historical averages. Differences between them highlight the AI's unique, adaptive perspective.
- **Final Price Distribution:** Compares the probability of where the price might end up. Is the AI model more optimistic, pessimistic, or does it see more/less risk (a wider or narrower distribution) than the traditional model?
- **Risk Percentile Comparison:** This gives you a direct, quantitative look at risk. For example, you can compare the "worst-case 5% scenario" (5th percentile) predicted by the AI vs. the traditional model.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute.
