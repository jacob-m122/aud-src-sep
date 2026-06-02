import kagglehub

# Download latest version
path = kagglehub.dataset_download("quanglvitlm/musdb18-hq")

print("Path to dataset files:", path)