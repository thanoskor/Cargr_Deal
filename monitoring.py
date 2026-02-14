import os
import re
import time
import joblib
import pandas as pd
import requests
import warnings

from DrissionPage._configs.chromium_options import ChromiumOptions
from DrissionPage._pages.chromium_page import ChromiumPage

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
USER_KEY = ""
APP_TOKEN = ""
USER_DATA_PATH = r'/browser_user_data'
SEEN_FILE = 'seen_deals.txt'  # File to store notified bikes


def load_seen_deals():
    """Loads the set of previously notified bikes from a file."""
    if not os.path.exists(SEEN_FILE):
        return set()
    try:
        with open(SEEN_FILE, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)
    except Exception:
        return set()


def save_seen_deal(signature):
    """Appends a new bike signature to the file."""
    try:
        with open(SEEN_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{signature}\n")
    except Exception as e:
        print(f"Error saving deal: {e}")


def create_signature(bike_data, price):
    """Creates a unique string to identify a specific listing."""
    # Example: Yamaha_Tracer 900_2019_15000_8500
    # This combination is almost guaranteed to be unique for used bikes
    return f"{bike_data['Brand']}_{bike_data['Model']}_{bike_data['Year']}_{bike_data['Kilometers']}_{price}"


def send_notification(message, title="Bike Deal Alert"):
    url = "https://api.pushover.net/1/messages.json"
    data = {
        "token": APP_TOKEN,
        "user": USER_KEY,
        "title": title,
        "message": message,
    }
    try:
        r = requests.post(url, data=data)
        r.raise_for_status()
        print(f"Notification sent.")
    except Exception as e:
        print(f"Failed to send notification: {e}")


def get_bike(bike_text):
    """Parses raw text from the listing to extract bike features."""
    year_pattern = r'(19\d{2}|20\d{2})\s\n'
    year_match = re.search(year_pattern, bike_text)
    year = year_match.group(1) if year_match else "0"

    full_title = "Unknown"
    if year != "0":
        lines = bike_text.splitlines()
        for line in lines:
            if year in line:
                clean_line = re.sub(r'(\d+ / \d+)|Προωθημένη|Με ζημιά', '', line)
                full_title = clean_line.replace(year, '').strip()
                break

    price_match = re.search(r'([\d.]+)\s*€', bike_text)
    price = int(price_match.group(1).replace('.', '')) if price_match else 0

    km_match = re.search(r'([\d.]+)\s*Km', bike_text)
    km = int(km_match.group(1).replace('.', '')) if km_match else 0

    cc_match = re.search(r'([\d.]+)\s*cc', bike_text)
    cc = int(cc_match.group(1).replace('.', '')) if cc_match else 0

    hp_match = re.search(r'(\d+)\s*hp', bike_text)
    hp = int(hp_match.group(1)) if hp_match else 0

    if full_title != "Unknown" and price > 0:
        parts = full_title.split()
        brand = parts[0] if len(parts) > 0 else "Unknown"
        model_name = " ".join(parts[1:]) if len(parts) > 1 else "Unknown"

        bike_data = {
            'Year': int(year),
            'Kilometers': km,
            'CC': cc,
            'HP': hp,
            'Brand': brand,
            'Model': model_name
        }
        return bike_data, price

    return None, 0


def make_prediction(bike_data, model, label_encoders):
    new_df = pd.DataFrame([bike_data])

    # Handle categorical encoding safely
    for col in ['Brand', 'Model']:
        if col in new_df.columns and col in label_encoders:
            le = label_encoders[col]
            # Strip whitespace and ensure string format
            val = str(new_df.iloc[0][col]).strip()

            # Use specific known value or fallback to 0 if unknown
            if val in le.classes_:
                new_df[col] = le.transform([val])
            else:
                new_df[col] = 0

    if hasattr(model, 'feature_names_in_'):
        new_df = new_df[model.feature_names_in_]

    try:
        predicted_price = model.predict(new_df)[0]
        return predicted_price
    except Exception as e:
        print(f"Prediction Error: {e}")
        return 0


def main():
    # --- Path Setup ---
    base_dir = os.getcwd()
    if os.path.basename(base_dir) == 'results':
        model_dir = os.path.join(os.path.dirname(base_dir), 'models')
    elif os.path.exists('models'):
        model_dir = 'models'
    else:
        print("Error: 'models' directory not found.")
        return

    print(f"Loading models from: {model_dir}")

    try:
        label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        model = joblib.load(os.path.join(model_dir, 'model_random_forest.pkl'))
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        return

    # --- Load History ---
    seen_deals = load_seen_deals()
    print(f"Loaded {len(seen_deals)} previously seen deals.")

    # --- Browser Setup ---
    options = ChromiumOptions()
    if os.path.exists(USER_DATA_PATH):
        options.set_paths(user_data_path=USER_DATA_PATH)

    page = ChromiumPage(options)
    url = "https://www.car.gr/classifieds/bikes/?category=15002&condition=used&pg=1"


    send_notification("Cargr monitoring started")
    print("Starting monitoring loop...")
    print("-" * 60)

    try:
        while True:
            try:
                page.get(url)
                time.sleep(3)

                listings = page.eles('tag:li')
                current_time = time.strftime("%H:%M:%S")
                print(f"[{current_time}] Scanned {len(listings)} listings...")

                for bike_elem in listings:
                    bike_data, price = get_bike(bike_elem.text)

                    if not bike_data or price == 0:
                        continue

                        # 1. Create a unique ID for this bike
                    signature = create_signature(bike_data, price)

                    # 2. Skip if we've already seen this exact deal
                    if signature in seen_deals:
                        continue

                    predicted_price = make_prediction(bike_data, model, label_encoders)
                    diff = predicted_price - price
                    is_deal = diff > 1000  # Threshold of 1000 Euro profit

                    # Log checking
                    log_symbol = "★ DEAL!" if is_deal else ""
                    print(
                        f"  {bike_data['Brand']:<10} {bike_data['Model']:<20} | {price:>5}€ (Pred: {int(predicted_price)}) {log_symbol}")

                    if is_deal:
                        # 3. Save to memory and file immediately
                        seen_deals.add(signature)
                        save_seen_deal(signature)

                        msg = (f"Deal Found!\n"
                               f"{bike_data['Brand']} {bike_data['Model']} ({bike_data['Year']})\n"
                               f"Price: {price}€\n"
                               f"Profit: {int(diff)}€")
                        send_notification(msg)

                print("-" * 60)
                time.sleep(30)

            except Exception as e:
                print(f"Loop Error: {e}")
                time.sleep(5)
                continue

    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        page.quit()


if __name__ == "__main__":
    main()