import csv
import datetime
import time
import logging
import signal
import sys
from huggingface_hub import HfApi
import requests
import schedule

# -------------- PARAMETERS TO CHANGE -------------- #
LIMIT = 1000      # <--- Change me if you want fewer or more models
SORT_BY = "downloads"   # <--- Possible options: "downloads", "trending", "likes", "lastModified", etc.
DIRECTION = -1      # <--- -1 => descending, 1 => ascending
FULL = True         # <--- Whether to retrieve extended metadata (True) or basic info (False)
RETRY_DELAY = 60    # <--- Wait time (in seconds) before retrying after rate limit
# -------------------------------------------------- #

# Logging Configuration
logging.basicConfig(filename="model_fetcher.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_rate_limits(response):
    # Rate Limit Information.
    if response and response.headers.get("X-RateLimit-Limit"):
        logging.info(f"Rate Limit: {response.headers['X-RateLimit-Limit']}")
        logging.info(f"Rate Limit Remaining: {response.headers['X-RateLimit-Remaining']}")
    else:
        logging.info("No rate limit information available in headers.")

def fetch_top_models():
    api = HfApi()

    try:
        logging.info(f"Fetching top {LIMIT} models...")
        models = api.list_models(
            limit=LIMIT,
            sort=SORT_BY,
            direction=DIRECTION,
            full=FULL
        )
        total_models = len(models)
        logging.info(f"Total models retrieved: {total_models}")

        top_1pct_count = max(1, int(total_models * 0.01))
        top_1pct_models = models[:top_1pct_count]

        # Output File Name
        timestamp = datetime.datetime.utcnow().isoformat(timespec='seconds').replace(':', '-')
        filename = f"top_1pct_models_{timestamp}.csv"

        # Writing Data to The CSV file
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["modelId", "downloads", "likes", "lastModified", "tags"])
            # Rows
            for m in top_1pct_models:
                writer.writerow([
                    m.modelId,
                    m.downloads,
                    m.likes,
                    m.lastModified,
                    ",".join(m.tags) if m.tags else ""
                ])

        logging.info(f"Top 1% downloaded models saved to: {filename}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error occurred: {e}")
        check_rate_limits(e.response)
        logging.info(f"Retrying in {RETRY_DELAY} seconds due to potential rate limit...")
        time.sleep(RETRY_DELAY)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

# Schedule The Task To Run Daily at Midnight
schedule.every().day.at("00:00").do(fetch_top_models)

# Graceful Shutdown handler
def signal_handler(sig, frame):
    logging.info("Received termination signal. Shutting down gracefully...")
    sys.exit(0)

# Registering signal handler for SIGINT: Ctrl+C and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Keep The Script Running and Checking For Scheduled Tasks
def run_scheduled_tasks():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    logging.info("Script started.")
    fetch_top_models()
    run_scheduled_tasks()
