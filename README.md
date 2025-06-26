# AI-Powered Stock Forecast with Mean Reversion

This project presents an advanced framework for stock market analysis that goes beyond simple price prediction. Inspired by the sophisticated tools available to professional investors, such as those found on [PRO Investing](https://www.proinvesting.co), this repository aims to provide an open-source implementation of a truly insightful, AI-powered forecasting model.

The framework is built on a hybrid **LSTM-Attention** model in PyTorch, which learns the underlying stochastic parameters of a stock's movement—**drift (μ)** and **volatility (σ)**.

The key innovation is the creation of a **blended, mean-reverting forecast**. The model uses its AI-driven perception of short-term market momentum to adjust a stable, long-term historical baseline. The forecast then intelligently decays this short-term momentum over the forecast horizon, creating a realistic, curved projection that is more financially sound than a simple linear trend.

This repository is a comprehensive, end-to-end example of applying modern machine learning to quantitative finance in a thoughtful and robust way.


*Final output showing a mean-reverting AI forecast vs. a traditional forecast for SPY.*

## 🚀 Core Features

- **Hybrid Deep Learning Model:** A sophisticated `EnhancedStockPredictor` class featuring:
  - **LSTM Layers:** To capture long-term temporal dependencies.
  - **Multi-Head Attention:** To focus on the most relevant historical data points.
  - **Multi-Task Learning:** The model is explicitly trained to simultaneously predict three targets: next-day price, historical volatility, and historical drift, leading to a much more stable and accurate model.

- **Advanced Financial Concepts:**
  - **Stochastic Parameter Prediction:** The AI learns to predict the `drift` and `volatility` that drive a stock's price movements.
  - **Mean Reversion:** The final forecast incorporates a time-varying drift path. It starts with a high, AI-driven momentum signal which gradually decays towards the long-term historical average, modeling a key behavior of financial markets.
  - **Hybrid Blending:** The forecast intelligently blends the AI's short-term signal with the stability of a long-term traditional model, controlled by a simple `alpha` parameter.

- **Monte Carlo Simulation:** Uses the predicted parameters to run a **Geometric Brownian Motion (GBM)** simulation, generating thousands of possible future price paths to produce a full probability distribution.

- **Comprehensive Feature Engineering:** Automatically generates a rich set of over a dozen technical indicators, including RSI, MACD, Bollinger Bands, and various moving average ratios.

- **Insightful, Comparative Visualization:** The final output is a dashboard that clearly compares the **Mean-Reverting AI Forecast** against a **Traditional Historical Forecast**, visualizing differences in:
  - The expected price path and 90% confidence intervals.
  - The final probability distribution of outcomes.
  - A side-by-side bar chart of key risk percentiles (5th, 25th, 50th, 75th, 95th).

## 🛠️ Technology Stack

- **Python 3.x**
- **PyTorch:** For building and training the deep learning model.
- **yfinance:** To download historical stock data from Yahoo Finance.
- **scikit-learn:** For robust data preprocessing (`MinMaxScaler`).
- **pandas & NumPy:** For data manipulation and numerical operations.
- **Matplotlib:** For creating detailed and publication-quality visualizations.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-stock-forecast.git
    cd ai-stock-forecast
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install torch yfinance scikit-learn pandas numpy matplotlib
    ```

## 📈 How to Run

The script is designed for easy execution. You can configure the main parameters directly in the `if __name__ == "__main__":` block at the end of the file.

1.  Open the Python script (e.g., `main.py`).
2.  Modify the parameters in the main execution block:

    ```python
    # MAIN EXECUTION BLOCK
    if __name__ == "__main__":
        ticker = "SPY"  # Change to any stock ticker (e.g., "AAPL", "NVDA")
        
        try:
            model, scaler_X, scaler_y, enhanced_data, feature_columns = train_enhanced_model(
                ticker=ticker, 
                epochs=100  # 100 epochs is a good starting point
            )
            print("\n✅ Model training completed successfully!")
            
            enhanced_analysis_and_visualization(
                ticker=ticker, 
                model=model, 
                # ... other parameters ...
                forecast_months=6,  # Set the forecast horizon
                alpha=0.25  # Tune AI influence (0.0 = pure trad, 1.0 = full AI)
            )
            
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
    ```

3.  Execute the script from your terminal:
    ```bash
    python your_script_name.py
    ```

The script will handle data downloading, feature engineering, model training, and the final visualization automatically.

## 📊 Interpreting the Final Forecast

The final forecast dashboard is the key output. The **red forecast cone** represents the sophisticated **Mean-Reverting AI model**.

- **The Curve:** Look for the characteristic "bend" in the red line. It should start steep (reflecting current momentum) and gradually flatten as it reverts towards the long-term trend.
- **The Console Output:** Check the `Blended Drift` value. This shows the initial annualized return the model is using, which will be a blend of the AI's high-frequency signal and the long-term historical average.
- **Risk Comparison:** The bar chart on the bottom-right gives the most direct comparison. It shows how the AI's perception of risk and reward (red bars) differs from the simple historical model (blue bars) across various scenarios.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
