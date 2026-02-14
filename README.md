# Motorcycle Deal Finder

An automated system that scrapes used motorcycle listings from car.gr, predicts fair market prices using machine learning, and sends real-time notifications when underpriced bikes are found.

## Overview

This project combines web scraping, machine learning, and automated monitoring to identify profitable motorcycle deals. It trains a Random Forest model on historical listings and continuously monitors new postings, alerting you via Pushover when a bike is listed significantly below its predicted value.

## Features

- **Automated Scraping**: Collects motorcycle listings from car.gr with brand, model, year, mileage, CC, and HP
- **ML Price Prediction**: Trains multiple regression models (Random Forest performs best) to predict fair prices
- **Real-time Monitoring**: Continuously checks for new listings every 30 seconds
- **Smart Notifications**: Sends Pushover alerts only for deals with 1000€+ profit margin
- **Duplicate Prevention**: Tracks seen listings to avoid repeat notifications

## Project Structure

```
├── project.ipynb          # End-to-end ML pipeline (scraping, training, evaluation)
├── monitoring.py          # Live monitoring script with notifications
├── models/                # Saved models and encoders
│   ├── model_random_forest.pkl
│   └── label_encoders.pkl
└── seen_deals.txt         # Persistent log of notified listings
```

## ⚙️ Configuration

1. **Get Pushover credentials** (for notifications):
   - Sign up at [pushover.net](https://pushover.net)
   - Update `USER_KEY` and `APP_TOKEN` in `monitoring.py`

2. **Train the model** (or use pre-trained):
   - Run all cells in `project.ipynb` to scrape data and train models
   - Models are saved automatically to `models/` directory

## 🏃 Usage

### Train Models
```python
# Run project.ipynb to:
# 1. Scrape listings from car.gr
# 2. Clean and preprocess data
# 3. Train multiple ML models
# 4. Evaluate and save best model
```

### Start Monitoring
```bash
python monitoring.py
```

The script will:
- Load trained models
- Check car.gr every 30 seconds
- Compare listing prices to predictions
- Send notifications for deals with 1000€+ profit margin

## Model Performance

The Random Forest model achieves the best results:
- **R² Score**: ~0.95
- **MAE**: ~500€
- **Key Features**: CC (51%), Year (21%), HP (13%)

## Notification Example

```
Deal Found!
Yamaha Tracer 900 (2019)
Price: 8500€
Profit: 1250€
```

## How It Works

1. **Scraper** extracts bike features from listings using regex patterns
2. **Model** predicts fair price based on brand, model, year, mileage, CC, and HP
3. **Monitor** compares actual vs predicted prices
4. **Alert** triggers when `predicted_price - actual_price > 1000€`
5. **Deduplication** prevents repeat notifications using unique signatures

## Customization

- **Deal threshold**: Adjust `diff > 1000` in `monitoring.py`
- **Scan frequency**: Change `time.sleep(30)` for faster/slower checks
- **Target category**: Modify URL category parameter (currently set to touring bikes)

## Requirements

- Python 3.7+
- Active Pushover account
- Internet connection
- Chrome/Chromium browser (for DrissionPage)
