import os
import subprocess
import sys

def download_musdb18():
    target_dir = "./data/musdb18_hq"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading MUSDB18-HQ dataset to {target_dir}")


    subprocess.run([
        "kaggle", "datasets", "download", 
        "-d", "quanglvitlm/musdb18-hq", 
        "-p", target_dir, 
        "--unzip"
    ], check=True)

    print("Finished downloading.")

if __name__ == "__main__":
    download_musdb18()