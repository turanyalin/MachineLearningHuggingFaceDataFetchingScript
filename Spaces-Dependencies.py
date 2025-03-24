import pandas as pd
from huggingface_hub import HfApi
import os
import time
from config import token

# ------------------------------------------------------------------------------
# 1) Prepare Data Folder
# ------------------------------------------------------------------------------
data_folder = '/Users/turanyalincelik/HF-Analysis/Data'
os.makedirs(data_folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 2) Fetch and Limit to Top 100 Models
# ------------------------------------------------------------------------------
print('1/4: Retrieving top 100 models from HF model hub.')
api = HfApi(token=token)

# Fetch 500 models and sort to ensure the best 100 are selected
all_models = api.list_models(full=True, limit=500, sort="downloads")

# Sort by number of downloads (descending) and pick the top 100
sorted_models = sorted(all_models, key=lambda m: m.downloads if m.downloads else 0, reverse=True)
top_models = sorted_models[:100]

# ------------------------------------------------------------------------------
# 3) Fetch Model Info & Dependencies
# ------------------------------------------------------------------------------
print(f'2/4: Fetching info for {len(top_models)} public repositories')

models = []
for model in top_models:
    try:
        model_info = api.model_info(repo_id=model.modelId)
        models.append(model_info)
        time.sleep(0.5)  # Avoid rate limiting
    except Exception as e:
        print(f'Error with model {model.modelId}: {str(e)}')
        continue

# ------------------------------------------------------------------------------
# 4) Categorize Spaces & Create DataFrame
# ------------------------------------------------------------------------------
def categorize_space(space_name, tags):
    """ Categorize a Hugging Face Space based on name and tags. """
    categories = {
        "Text-based": ["chatbot", "text", "NLP", "question-answering", "translation", "summarization"],
        "Image-based": ["image", "diffusion", "segmentation", "object-detection", "classification"],
        "Multimodal": ["multimodal", "vision-language", "text-to-image", "image-to-text"],
        "Audio-based": ["speech", "audio", "speech-to-text", "voice", "tts"],
        "Scientific & Research": ["medical", "finance", "legal", "biomed", "climate"],
    }

    # Convert to lowercase for matching
    space_name_lower = space_name.lower()
    tags_lower = [t.lower() for t in tags]

    # Check for matches in space name or tags
    for category, keywords in categories.items():
        if any(keyword in space_name_lower for keyword in keywords) or any(keyword in tags_lower for keyword in keywords):
            return category

    return "Other"

print('3/4: Creating categorized dataframe for space dependencies...')
edges_spaces = []
for model in models:
    for space in model.spaces:
        category = categorize_space(space, model.tags)  # Categorize each Space
        edges_spaces.append((space, model.modelId, category))

# Convert to DataFrame
if edges_spaces:
    edgelist_spaces = pd.DataFrame(edges_spaces, columns=['source', 'target', 'category'])
    output_file = os.path.join(data_folder, 'hf-edgelist-spaces-dependencies.csv')
    edgelist_spaces.to_csv(output_file, index=False)
    print(f'4/4: Saved CSV of categorized space dependencies to {output_file}')
else:
    print("No space dependencies found for the top 100 models.")
