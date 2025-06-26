# AI-Powered Stock Analysis & Forecast

This project presents a sophisticated framework for stock market analysis and forecasting, combining deep learning with classical financial modeling. Inspired by professional-grade financial tools like [PRO Investing](https://www.proinvesting.co), this project aims to bring similar AI-powered insights to the open-source community.

It leverages a hybrid **LSTM-Attention** model to predict not just the future price of a stock, but also the underlying parameters of its stochastic process—**drift** and **volatility**. These parameters are then fed into a **Geometric Brownian Motion (GBM) Monte Carlo simulation** to generate thousands of possible future price paths, providing a rich, probabilistic forecast.


*Example output showing an AI-enhanced vs. a traditional forecast for SPY.*

## 🚀 Features

- **Hybrid Deep Learning Model:** An `EnhancedStockPredictor` class built with PyTorch, featuring:
  - **LSTM Layers:** To capture long-term temporal dependencies in time-series data.
  - **Multi-Head Attention:** To allow the model to focus on the most relevant historical data points.
  - **Multiple Prediction Heads:** The model simultaneously predicts **Price**, **Stochastic Drift (μ)**, and **Stochastic Volatility (σ)**.

- **Comprehensive Feature Engineering:** Automatically generates a rich set of features, including **RSI**, **MACD**, **Bollinger Bands**, moving averages, and more.

- **Monte Carlo Simulation:** Uses the AI-predicted parameters to run an "enhanced" GBM simulation, providing a probability distribution of future prices.

- **Comparative Analysis:** The AI-enhanced forecast is always plotted against a **traditional forecast** that uses long-term historical averages. This clearly demonstrates the value of the AI model's adaptive insights.

- **Advanced Visualization:** Generates a comprehensive dashboard to visualize model performance, forecast comparisons, price distributions, and risk percentiles.

## 🛠️ Technology Stack

- **Python 3.x**
- **PyTorch:** For building and training the deep learning model.
- **yfinance:** To download historical stock data from Yahoo Finance.
- **scikit-learn:** For data preprocessing and evaluation metrics.
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

The main script is designed to be run directly. Simply modify the `ticker` variable in the `if __name__ == "__main__":` block to analyze your desired stock.

1.  Open the Python script (e.g., `main.py`).
2.  Locate the main execution block at the bottom.
3.  Change the `ticker` variable.

    ```python
    # MAIN EXECUTION BLOCK
    if __name__ == "__main__":
        # Configuration: Change the ticker for the desired stock/ETF
        ticker = "SPY"  # <-- CHANGE THIS TO ANY TICKER
        
        try:
            # ... training and analysis code ...
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
    ```

4.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```

## 📊 Interpreting the Results

The script generates two main plot windows: a **Model Evaluation Dashboard** and a **Forecast Analysis Dashboard**. The forecast dashboard is key for insights, comparing the dynamic **AI-Enhanced Forecast (Red Cone)** against the static **Traditional Forecast (Blue Cone)**. This comparison highlights how the AI's perception of current market risk and momentum differs from the long-term historical average.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
