from huggingface_hub import snapshot_download

repos = [
    "Project-AgML/pomegranate_disease_classification",
    "VedicSeeker/pomegranateDiseaseClassification",
    "Ibrahim125/Pomegranate-Fruit-Diseases-Dataset-for-Deep-Learning-Models"
]

for repo in repos:
    print(f"Downloading {repo}...")
    try:
        path = snapshot_download(repo_id=repo, repo_type="dataset")
        print(f" -> Done: {path}")
    except Exception as e:
        print(f" -> Failed {repo}: {e}")
