# 📊 AI Stock Screener & Quantitative Market Analysis

An AI/ML-powered Python application for real-time NSE stock screening, technical analysis, quantitative feature engineering, and SMMA crossover profitability prediction.

## 🚀 Overview

This project was developed as an AI/ML-based quantitative stock market screening and analysis system.

The application integrates real-time market data from FYERS and combines technical indicators, market depth, LTQ/ETQ analysis, and machine learning to identify potentially profitable SMMA crossover signals and filter out signals with a higher probability of failure.

## ✨ Key Features

- 📈 Real-time NSE stock screening
- 💰 LTP filtering between ₹30 and ₹500
- 💧 Liquidity filtering using Bid/Ask Quantity
- 📊 SMMA (20) and SMMA (120)
- 🔄 Automatic SMMA crossover detection
- 📦 Last Traded Quantity (LTQ) analysis
- ⏱️ 5-minute, 20-minute and 60-minute ETQ analysis
- 💹 20-minute and 60-minute average LTP
- 📚 Real-time market depth
- 🤖 Machine Learning-based trade prediction
- 🎯 Profitability probability/confidence
- 🚫 Identification of potentially avoidable signals
- 💵 Automated trade entry, exit and P/L analysis
- 📊 Real-time Flask dashboard
- 🔬 Quantitative feature engineering
- 📉 Historical backtesting
- 🔄 Walk-forward validation

## 🧠 Machine Learning

The ML component uses market and LTQ-based features to determine whether an SMMA crossover should be accepted or avoided.

Example quantitative features include:

- LTQ
- Average LTQ over 2 minutes
- Average LTQ over 5 minutes
- LTQ ratio
- LTQ z-score
- LTQ spike detection
- Price movement
- SMMA relationship
- Market liquidity
- Trade-related statistics

The objective is to reduce losing trades by filtering SMMA signals using quantitative market conditions.

## 📈 Trading Logic

### Buy Signal

SMMA (20) crosses above SMMA (120).

```text
SMMA 20 ↑
      crosses
SMMA 120
      ↓
BUY
