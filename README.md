/// Audio Separation Project ///

This project implements a neural network-based architecture for source separation, specifically targeting audio bleed in multi-track recordings (Currently work in progress)

src/: Core logic including preprocess.py (STFT/Spectrogram generation) and dataloader.py (Dataset indexing and batching).

data/: Sample test files (Violet Wave) used for pipeline validation.

notebooks/: Demo notebook showing the data pipeline in action.

Installation
You can install this repository as a package in your local environment:

Bash
pip install .