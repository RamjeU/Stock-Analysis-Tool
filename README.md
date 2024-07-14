# Stock Analysis Tool with Tkinter and yFinance

This project is a comprehensive stock analysis tool built using Python, Tkinter, and yFinance. It allows users to fetch historical stock data, analyze it using various methods, and visualize the results. The tool includes features for summary statistics, plotting stock closing prices, calculating moving averages, and predicting future stock prices using linear regression.

## Features

- **Autocomplete Combobox**: Easily select stock tickers from an extensive list with autocomplete functionality.
- **Fetch Historical Data**: Retrieve historical stock data from Yahoo Finance.
- **Data Analysis**: Perform various analyses, including summary statistics, plotting, moving averages, and price predictions.
- **Visualization**: Display plots and summary statistics in separate windows.
- **Error Handling**: Ensure all input fields are filled correctly before proceeding with the analysis.

## Code Overview

### `AutoCompleteCombobox` Class

A custom combobox widget that provides autocomplete functionality for stock tickers.

### `get_stock_data(ticker, start_date, end_date)`

Fetches historical stock data from Yahoo Finance.

### `analyze_data(data, analysis_type, window=None)`

Performs various analyses on the stock data:
- Summary statistics
- Plotting closing prices
- Calculating moving averages
- Predicting future prices

### `predict_stock_prices(data)`

Predicts future stock prices using linear regression and displays the results.

### `on_submit()`

Handles the submission of user inputs and displays the appropriate analysis results.

### `display_summary(summary)`

Displays summary statistics in a new window.

### `display_plot(fig)`

Displays plots in a new window.

