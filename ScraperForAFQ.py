import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import token  # Ensure your token is configured

# ------------------------------------------------------------------------------
# 1) Prepare data folder
# ------------------------------------------------------------------------------
data_folder = '/Users/turanyalincelik/HF-Analysis/Data'
os.makedirs(data_folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 2) Calculate top 1% count and retrieve models sorted by downloads
# ------------------------------------------------------------------------------
estimated_total_models = 1531678
top_1pct_count = max(1, int(estimated_total_models * 0.01))
print(f"Retrieving top {top_1pct_count} models sorted by downloads...")

api = HfApi(token=token)
models_info = api.list_models(sort="downloads", direction=-1, full=True, limit=top_1pct_count)
model_ids = [model.modelId for model in models_info]

# Build a dictionary to map modelId -> downloads count
downloads_dict = {model.modelId: model.downloads for model in models_info}

# ------------------------------------------------------------------------------
# 3) Define a function to scrape model tree data from each model's page
# ------------------------------------------------------------------------------
def get_model_tree_data(model_id):
    url = f"https://huggingface.co/{model_id}"
    headers = {"User-Agent": "Mozilla/5.0"}  # Common User-Agent
    try:
        response = requests.get(url, headers=headers)
    except Exception as e:
        print(f"Error fetching {model_id}: {e}")
        return None

    if response.status_code != 200:
        print(f"Error fetching {model_id}: status code {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    # Convert the full page text into a single string
    page_text = soup.get_text(separator=" ", strip=True)
    data = {}
    # Use regex that matches both "model" and "models" (the 's' is optional)
    for category in ["Adapters", "Finetunes", "Quantizations"]:
        pattern = rf'{category}\s+(\d+)\s+models?'
        match = re.search(pattern, page_text, re.IGNORECASE)
        if match:
            data[category] = int(match.group(1))
        else:
            data[category] = None
    return {"modelId": model_id, **data}

# ------------------------------------------------------------------------------
# 4) Process models in parallel to extract model tree data
# ------------------------------------------------------------------------------
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_model = {executor.submit(get_model_tree_data, model_id): model_id for model_id in model_ids}
    for future in as_completed(future_to_model):
        result = future.result()
        if result:
            # Append the downloads count from the API data
            result["downloads"] = downloads_dict.get(result["modelId"], None)
            results.append(result)
            print(f"Processed {result['modelId']}: {result}")

# ------------------------------------------------------------------------------
# 5) Save results to CSV, sorted by downloads (most to least downloaded)
# ------------------------------------------------------------------------------
df = pd.DataFrame(results)
df.sort_values(by="downloads", ascending=False, inplace=True)
csv_path = os.path.join(data_folder, "hf_top1pct_model_tree_data.csv")
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")