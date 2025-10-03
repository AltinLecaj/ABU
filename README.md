Market Sentiment Analysis: Real-time Examination of Financial News Dynamics in Equity Markets
A Python-based trading strategy framework that leverages natural language processing (NLP) and sentiment analysis of financial news to generate buy/sell signals and backtest investment strategies.

Overview
This project analyzes financial news sentiment using transformer-based models to evaluate market sentiment around specific equities. It generates trading signals based on sentiment thresholds and provides comprehensive backtesting capabilities to compare traditional Dollar-Cost Averaging (DCA) against sentiment-modified investment strategies.

Key Features

-Automated News Collection: Fetches financial news from EODHD API for specified tickers and date ranges
-Sentiment Analysis: Uses DistilRoBERTa fine-tuned on financial news for accurate sentiment classification
-Signal Generation: Generates BUY/SELL signals based on configurable sentiment thresholds
-Multiple Backtesting Strategies:
    Traditional DCA vs. Modified DCA with sentiment multipliers
    Sell signal-based investment strategy across multiple tickers
-Data Visualization: Interactive price charts with sentiment-colored signals
-GPU Acceleration: Supports DirectML for AMD GPU acceleration

Requirements
  pandas
  transformers
  torch-directml
  matplotlib
  mplcursors
  eodhd

Setup
  1. Create a creds.py file in your project directory with your EODHD API key:
      pythonapi_key = "your_api_key_here"
  2. Ensure you have the required Python packages installed
  3. Configure GPU acceleration if using AMD hardware (DirectML is configured by default)

Core Functions

Data Collection
  create_csv(ticker, start_date, end_date, limit)
    -Fetches financial news for a specific ticker and date range, filtering articles to include only those primarily focused on the target ticker.
    
  create_complete_csv(ticker)
    -Aggregates news data across multiple weeks (default: March 1, 2025 to April 15, 2025) for comprehensive analysis.
    
  create_final_csv(ticker)
    -Processes collected news through the sentiment analysis pipeline and exports results to CSV.
    
Signal Generation
  buy_or_sell(csv_file, week_increment)
    -Generates trading signals based on sentiment ratios:

      STRONG BUY: >70% positive sentiment
      BUY: >50% positive sentiment
      STRONG SELL: >70% negative sentiment
      SELL: >50% negative sentiment
      NEUTRAL: All other cases

Backtesting
  backtest_dca_multiplier(signal_csv_file, ticker, investment_per_period, week_increment)
    -Compares traditional DCA against a modified strategy that:

      -Invests 50% of the standard amount weekly
      -Banks the remaining 50%
      -Deploys 50% of banked funds on SELL signals
      -Reports total returns and percentage gains for both strategies

  backtest_sell_strategy_lists(tickers_list, price_csv_list, signal_data_list, initial_investment)
    -Tests a contrarian strategy that invests a fixed amount on every SELL/STRONG SELL signal across multiple stocks, with visualizations showing:

      -Individual stock returns
      -Overall portfolio performance

Visualization
  get_eod_prices(SYMBOL_NAME, period_day_week_month)
    -Retrieves historical adjusted close prices from EODHD.
  plot_all_data(df1, df2)
    -Creates interactive plots showing price movements with color-coded sentiment signals (green for buy, red for sell, grey for neutral).
    
Usage Example
python# 1. Collect and analyze news data
  df = create_final_csv('AAPL')

# 2. Generate trading signals
  signals = buy_or_sell('AAPL_data1.csv', week_increment=1)

# 3. Backtest the strategy
  backtest_dca_multiplier('AAPL_data1.csv', 'AAPL', investment_per_period=100)

# 4. Visualize results
  prices = get_eod_prices('AAPL', 'w')
  plot = plot_all_data(signals, prices)
  plot.show()

# 5. Multi-ticker backtest
  tickers = ['AAPL', 'MSFT', 'GOOGL']
  price_files = ['AAPL_prices.csv', 'MSFT_prices.csv', 'GOOGL_prices.csv']
  signal_data = [signals_aapl, signals_msft, signals_googl]
  results, overall_return, invested, final = backtest_sell_strategy_lists(
      tickers, price_files, signal_data, initial_investment=100
  )
Model Information
  This project uses the mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis model from Hugging Face, which is 
  specifically trained on financial news for improved accuracy in market sentiment classification.

Data Sources

  -News Data: EODHD Financial News API
  -Price Data: EODHD End-of-Day Prices API

Output Files
  -The system generates CSV files in the format {TICKER}_data1.csv containing:

    -date: Publication date of the news article
    -content: Cleaned article text
    -sentiment: Classification result with label and confidence score

Performance Considerations

  -Sentiment analysis processes 100 articles at a time with progress logging
  -GPU acceleration significantly improves processing speed
  -Weekly data aggregation reduces API calls and processing time
  -News filtering removes articles with mixed ticker focus to improve signal quality


EODHD for financial data API
Hugging Face for the sentiment analysis model
The transformers library team
