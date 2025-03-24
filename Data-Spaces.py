import json
import pandas as pd
from huggingface_hub import HfApi, get_repo_discussions
import numpy as np
import os
import time
from config import token

# ------------------------------------------------------------------------------
# 1) Prepare Data Folder
# ------------------------------------------------------------------------------
data_folder = "/Users/turanyalincelik/HF-Analysis/Data"
os.makedirs(data_folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 2) Fetch and Limit to Top 100 Spaces
# ------------------------------------------------------------------------------
print('1/3: Fetching Spaces from HF Hub...')
api = HfApi(token=token)

# Fetch more spaces to ensure filtering (fetch 500, then select the top 100)
all_spaces = list(api.list_spaces(full=True, limit=500))

# Sort by 'likes' (or choose another metric like 'lastModified' or 'commits')
sorted_spaces = sorted(all_spaces, key=lambda s: s.likes if s.likes else 0, reverse=True)

# Keep only the top 100 spaces
top_spaces = sorted_spaces[:100]

# ------------------------------------------------------------------------------
# 3) Process Space Data
# ------------------------------------------------------------------------------
print(f'2/3: Processing data for {len(top_spaces)} spaces...')

space_data_rows = []
for space in top_spaces:
    if '/' in space.id:
        organisation, space_name = space.id.split('/', 1)
    else:
        organisation = np.nan
        space_name = space.id

    license = next((tag.split(":")[1] for tag in space.tags if "license:" in tag), np.nan)

    # Retrieve commit data
    try:
        commits = list(api.list_repo_commits(repo_id=space.id, repo_type="space"))
        num_commits = len(commits)
        commit_contributors = set(author for commit in commits for author in commit.authors)
    except Exception as e:
        print(f"Error retrieving commits for {space.id}: {str(e)}")
        num_commits = np.nan
        commit_contributors = set()

    # Retrieve discussions data
    try:
        discussions = list(get_repo_discussions(repo_id=space.id, repo_type="space"))
        num_discussions = len(discussions)
        discussions_contributors = set(d.author for d in discussions)
    except Exception as e:
        if "403 Forbidden" in str(e):  # Suppress expected errors
            print(f"Skipping discussions for {space.id} (discussions disabled).")
        else:
            print(f"Error retrieving discussions for {space.id}: {str(e)}")
        num_discussions = np.nan
        discussions_contributors = set()

    # Store space data
    space_data_rows.append({
        'owner': organisation,
        'space': space_name,
        'id': space.id,
        'lastModified': space.lastModified if space.lastModified else np.nan,
        'license': license,
        'tags': ', '.join(space.tags if space.tags else []),
        'likes': space.likes if space.likes else np.nan,
        'commits': num_commits,
        'commits_contributors': len(commit_contributors),
        'discussions': num_discussions,
        'discussions_contributors': len(discussions_contributors)
    })

    time.sleep(1)  # Avoid rate-limiting

# Convert list to DataFrame
df = pd.DataFrame(space_data_rows)

# Convert numeric columns to int
for column in ['likes', 'commits', 'discussions']:
    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)

# Convert 'lastModified' to YYYY-MM-DD
if 'lastModified' in df.columns:
    df['lastModified'] = pd.to_datetime(df['lastModified'], errors='coerce').dt.strftime('%Y-%m-%d')

# ------------------------------------------------------------------------------
# 4) Save Data to Excel
# ------------------------------------------------------------------------------
output_file = os.path.join(data_folder, 'hf-top100-spaces.xlsx')
df.to_excel(output_file, index=False)

print(f'3/3: Saved top 100 spaces data to {output_file}')
