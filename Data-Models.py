import os
import pandas as pd
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor
from config import token  # Ensure your token is configured

# ------------------------------------------------------------------------------
# Helper function to categorize training data based on tags
# ------------------------------------------------------------------------------
def categorize_model(tags):
    """
    Categorizes the model based on its tags.
    Returns a string representing the training data category.
    """
    if not tags:
        return "Unknown"
    tags_str = " ".join(tag.lower() for tag in tags)
    if "math" in tags_str or "mathematics" in tags_str:
        return "Mathematics"
    elif "image" in tags_str or "vision" in tags_str:
        return "Computer Vision"
    elif "speech" in tags_str or "audio" in tags_str or "tts" in tags_str:
        return "Speech"
    elif "text" in tags_str:
        return "Natural Language Processing"
    else:
        return "Other"

# ------------------------------------------------------------------------------
# Function to process metadata for a single model (excluding license and private fields)
# ------------------------------------------------------------------------------
def process_model_metadata(m):
    owner = m.id.split('/')[0] if '/' in m.id else None
    return {
        'modelId': m.id,
        'owner': owner,
        'downloads': m.downloads if m.downloads is not None else 0,
        'likes': m.likes,
        'tags': m.tags,
        'lastModified': m.lastModified,
        'pipeline_tag': getattr(m, 'pipeline_tag', None),
        'training_data_category': categorize_model(m.tags)
    }

# ------------------------------------------------------------------------------
# 1) Prepare data folder
# ------------------------------------------------------------------------------
data_folder = '/Users/turanyalincelik/HF-Analysis/Data'
os.makedirs(data_folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 2) Retrieve Top 1% Models (by downloads) from the HF Hub
# ------------------------------------------------------------------------------
# Total number of models on Hugging Face Hub
estimated_total_models = 1531678
# Calculate top 1% count (~15,316 models)
top_1pct_count = max(1, int(estimated_total_models * 0.01))
print(f'Retrieving top {top_1pct_count} models sorted by downloads...')

api = HfApi(token=token)
# Use numeric direction (-1 for descending order)
models_info = api.list_models(sort="downloads", direction=-1, full=True, limit=top_1pct_count)

# ------------------------------------------------------------------------------
# 3) Process Model Metadata in Parallel
# ------------------------------------------------------------------------------
with ThreadPoolExecutor() as executor:
    models_data = list(executor.map(process_model_metadata, models_info))

# Create DataFrame and sort by downloads (optional, as API returns sorted list)
df = pd.DataFrame(models_data)
df.sort_values(by='downloads', ascending=False, inplace=True, na_position='last')

# ------------------------------------------------------------------------------
# 4) Save the Top Models to CSV
# ------------------------------------------------------------------------------
csv_path = os.path.join(data_folder, 'hf-models-top1pct.csv')
df.to_csv(csv_path, index=False)
print(f"Saved top 1% models data (total: {len(df)}) to: {csv_path}")
