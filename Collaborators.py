import os
import json
import pandas as pd
from huggingface_hub import HfApi
from config import token
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo import MongoClient
import certifi

# ------------------------------------------------------------------------------
# 1) Prepare Data Folder
# ------------------------------------------------------------------------------
data_folder = '/Users/turanyalincelik/HF-Analysis/Data'
os.makedirs(data_folder, exist_ok=True)

# ------------------------------------------------------------------------------
# 2) Retrieve Top 1% Models
# ------------------------------------------------------------------------------
estimated_total_models = 1531678
top_1pct_count = max(1, int(estimated_total_models * 0.01))
print(f'Retrieving top {top_1pct_count} models sorted by downloads...')

api = HfApi(token=token)
models_info = api.list_models(sort="downloads", direction=-1, full=True, limit=top_1pct_count)
model_ids = [model.modelId for model in models_info]


# ------------------------------------------------------------------------------
# 3) Helper Function for Parallel Processing
# ------------------------------------------------------------------------------
def process_model(model_id):
    local_api = HfApi(token=token)
    try:
        commits = local_api.list_repo_commits(repo_id=model_id, token=token)
    except Exception as e:
        return {"model_id": model_id, "error": str(e)}

    edges = []
    # Create a set of unique authors from all commits
    unique_authors = list(set(author for commit in commits for author in commit.authors))

    for commit in commits:
        source_author = commit.authors[0]
        for target_author in unique_authors:
            if source_author != target_author:
                edges.append((source_author, target_author, model_id))
        # Handle the case where only one contributor exists
        if len(unique_authors) == 1:
            edges.append((source_author, source_author, model_id))

    return {
        "model_id": model_id,
        "edges": edges,
        "has_commits": len(commits) > 0,
        "only_one_contributor": len(unique_authors) == 1,
        "error": None
    }


# ------------------------------------------------------------------------------
# 4) Retrieve Commit Data in Parallel
# ------------------------------------------------------------------------------
edgelist = []
public_repositories = 0
repositories_with_commits = 0
repositories_only_one_contributor = 0

print(f'Processing commit data for {len(model_ids)} top models in parallel...')
results = []

with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_model = {executor.submit(process_model, model_id): model_id for model_id in model_ids}
    for future in as_completed(future_to_model):
        result = future.result()
        results.append(result)

# Aggregate results and build model-level edges
records = []
for result in results:
    if result.get("error") is None:
        public_repositories += 1
        if result["has_commits"]:
            repositories_with_commits += 1
        if result["only_one_contributor"]:
            repositories_only_one_contributor += 1
        for (src, tgt, model_id) in result["edges"]:
            owner, model_name = model_id.split("/", 1)
            records.append({
                "source": src,
                "target": tgt,
                "model_id": model_id,
                "owner": owner,
                "model_name": model_name
            })
    else:
        print(f"Error with {result['model_id']}: {result['error']}")

# ------------------------------------------------------------------------------
# 5) Save and Summarize the Data
# ------------------------------------------------------------------------------
edgelist_df = pd.DataFrame(records)
edgelist_df = edgelist_df.groupby(['source', 'target']).agg(
    freq=('model_id', 'count'),
    models_contributed_to=('model_id', lambda x: list(set(x))),
    owners=('owner', lambda x: list(set(x))),
    model_names=('model_name', lambda x: list(set(x)))
).reset_index()

# Save the enriched CSV (convert list columns to JSON strings for CSV storage)
output_csv = os.path.join(data_folder, 'hf-edgelist-contributors-with-models.csv')
edgelist_df["models_contributed_to"] = edgelist_df["models_contributed_to"].apply(json.dumps)
edgelist_df["owners"] = edgelist_df["owners"].apply(json.dumps)
edgelist_df["model_names"] = edgelist_df["model_names"].apply(json.dumps)
edgelist_df.to_csv(output_csv, index=False)

print(f'Saved enriched commit data to {output_csv}')
print('\nKey Facts:')
print(f' - Public model repositories processed: {public_repositories}')
print(
    f' - Model repositories with commits: {repositories_with_commits} ({repositories_with_commits / public_repositories:.2%})')
print(
    f' - Model repositories with zero commits: {public_repositories - repositories_with_commits} ({(public_repositories - repositories_with_commits) / public_repositories:.2%})')
print(
    f' - Model repositories with only one contributor: {repositories_only_one_contributor} ({repositories_only_one_contributor / (repositories_with_commits or 1):.2%})')

# ------------------------------------------------------------------------------
# 6) Save the Data to MongoDB
# ------------------------------------------------------------------------------
mongo_uri = "mongodb+srv://turanyalin:TuranYalin123456789@huggingfacemetadata.qwvgw.mongodb.net/?retryWrites=true&w=majority&appName=HuggingFaceMetaData"
client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
db = client['HuggingFaceMetaData']
collection = db['edgelist_contributors']

# Convert DataFrame to a list of dictionaries for insertion
documents = edgelist_df.to_dict(orient='records')
result = collection.insert_many(documents)
print(f"Inserted {len(result.inserted_ids)} documents into MongoDB collection 'Contributors'.")

client.close()
