import pandas as pd
import re
from huggingface_hub import HfApi
import os
from collections import defaultdict
from config import token

# ------------------------------------------------------------------------------
# 1) Prepare Data Folder
# ------------------------------------------------------------------------------
data_folder = '/Users/turanyalincelik/HF-Analysis/Data'
os.makedirs(data_folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 2) Retrieve First 100 Models from Hugging Face
# ------------------------------------------------------------------------------
print('1/3: Retrieving the first 100 models from the HF Hub...')
api = HfApi(token=token)

# Get the top 100 models sorted by downloads
top_models = api.list_models(full=True, limit=100, sort="downloads")

# ------------------------------------------------------------------------------
# 3) Convert Model Info to a DataFrame
# ------------------------------------------------------------------------------
models_data = []
for m in top_models:
    models_data.append({
        'modelId': m.modelId,
        'sha': m.sha,
        'lastModified': m.lastModified,
        'private': m.private,
        'downloads': m.downloads,
        'likes': m.likes,
        'tags': m.tags,
        'pipeline_tag': m.pipeline_tag
    })

df = pd.DataFrame(models_data)

# Save the first 100 models
csv_path = os.path.join(data_folder, 'hf-models-first100.csv')
# df.to_csv(csv_path, index=False)  # Optional: Save model data
print(f'   - Saved first 100 models to: {csv_path}')

# ------------------------------------------------------------------------------
# 4) Gather Commit Data for Each Model
# ------------------------------------------------------------------------------
model_ids = df['modelId'].tolist()
print(f'2/3: Retrieving commit data for {len(model_ids)} models...')

# Dictionary to store contributions: { modelId -> { contributor -> commit_count } }
contributor_data = defaultdict(lambda: defaultdict(int))

for model_id in model_ids:
    try:
        # Get commit history for the model
        commits = api.list_repo_commits(repo_id=model_id, token=token)
    except Exception as e:
        print(f'Error fetching commits for {model_id}: {str(e)}')
        continue

    # Count commits per contributor for this model
    for commit in commits:
        for author in commit.authors:
            contributor_data[model_id][author] += 1

# ------------------------------------------------------------------------------
# 5) Identify Contributor Affiliation (Company, Institution, Individual)
# ------------------------------------------------------------------------------
INSTITUTION_DOMAINS = ["mit.edu", "stanford.edu", "ox.ac.uk", "harvard.edu", "berkeley.edu"]
COMPANY_DOMAINS = ["google.com", "microsoft.com", "meta.com", "apple.com", "amazon.com"]


def classify_affiliation_by_email(email):
    """Determine if a contributor belongs to a company, institution, or is independent using email."""
    domain_match = re.search(r'@([\w.-]+)', email)
    if domain_match:
        domain = domain_match.group(1).lower()
        if domain in INSTITUTION_DOMAINS:
            return "Institution"
        elif domain in COMPANY_DOMAINS:
            return "Company"
    return "Individual"


def classify_affiliation_by_profile(username):
    """Check Hugging Face profile metadata to determine organization affiliation."""
    try:
        user_info = api.user_info(username)
        if user_info.full_name and "org_" in user_info.full_name.lower():
            return "Company" if "company" in user_info.full_name.lower() else "Institution"
        if user_info.email:
            return classify_affiliation_by_email(user_info.email)
    except Exception:
        return "Individual"
    return "Individual"


# ------------------------------------------------------------------------------
# 6) Convert Contributor Data to DataFrame & Save
# ------------------------------------------------------------------------------
contributor_list = []

for model_id, contributors in contributor_data.items():
    for contributor, commit_count in contributors.items():
        # First try email-based classification, if not found use Hugging Face profile
        affiliation = classify_affiliation_by_email(contributor)
        if affiliation == "Individual":
            affiliation = classify_affiliation_by_profile(contributor)

        contributor_list.append({
            'modelId': model_id,
            'contributor': contributor,
            'commit_count': commit_count,
            'affiliation': affiliation
        })

contributor_df = pd.DataFrame(contributor_list)

# Save contributor data to CSV
contributor_csv = os.path.join(data_folder, 'hf-contributors-first100.csv')
contributor_df.to_csv(contributor_csv, index=False)

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
print(f'3/3: Saved contributor commit data to: {contributor_csv}')
print(f'   - Total Contributors Recorded: {len(contributor_df["contributor"].unique())}')
print(f'   - Total Model Contributions Recorded: {len(contributor_df)}')
