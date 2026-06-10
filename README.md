# Audio Stem Source Separation via Cross-Track Attention by Jacob Mitani

## Project Purpose
In multi-track audio engineering, acoustic "bleed" (e.g., a vocal microphone picking up the snare drum) is a common issue faced by recording and mixing engineers. Traditional gating creates choppy audio, and standard EQ damages the primary signal's frequency spectrum. This issue is particularly interesting to me primarily because I have encountered it naturally by learning to record songs in a home studio set-up (various instruments with bleed potential e.g. amplified electric guitar, bass guitar, drums, and vocals).

This project implements a neural network architecture for source separation by means of a Cross-Track Attention Mechanism. The model is fed an artifacted target track and an isolated reference track (the bleed source), and the model computes a time-frequency attention matrix. This maps the structural image of the bleed onto the target track, and the network is thus able to subtract the artifact without degrading the primary instrument.

## Dataset and Augmentation
This project utilizes the **MUSDB18-HQ** dataset, which contains 150 uncompressed (.wav) multi-track studio recordings at 44.1k 16-bit.

MUSDB18-HQ contains already-isolated stems, thus realistic acoustic bleed was synthesized during training via the `MusdbDataset` class (in `src/dataloader.py`) which takes a random 3-second segment of a target stem and applies stochastic augmentation:

* **Random Mic Bleed:** 5% to 40% gain of the reference track.
* **Random Phase Flip:** Prevents the network from memorizing wave shapes.
* **Stochastic Room Reflections:** Random delay times (10ms–1000ms) and decay variables.
* **Analog Noise:** Gaussian noise floor injection.

This data augmentation is implemented to force the model to learn the abstract acoustic properties of bleed, and prevents overfitting via a static formula, which resulted in severe overfitting on the first training run.

## Training Instructions
1. **Installation:** Clone the repository and install it as a local package by running `pip install -e .` in the root directory.
2. **Training Execution:** The model was trained using a Google Colab instance with an NVIDIA A100 GPU. For training, open and run `notebooks/run_training.ipynb`, which executes the logic located in `src/train_model.py`.

## Results and Metrics
The model was optimized using L1 Loss and an Adam optimizer. Performance was measured using **SDR (Signal-to-Distortion Ratio)**, **SI-SDR (Scale-Invariant SDR)**, and **Mean LSD (Log-Spectral Distance)**. 

Training was completed through three iterations:

**Log 1** (10kHz Max Frequency, Batch Size 32):
Validation Loss: 0.1264
Validation SDR: 63.55 dB
Validation SI-SDR: 52.64 dB
Validation Mean LSD: 37.3217

Log 1 achieved incredibly high validation metrics, but evaluation revealed the model had overfit to early static augmentation (a fixed 650ms delay). Furthermore, cutting frequencies above 10kHz resulted in noticeably muffled audio.

**Log 2** - (10kHz Max Frequency, Batch Size 32):
Validation Loss: 0.2318
Validation SDR: 8.11 dB
Validation SI-SDR: 3.78 dB
Validation Mean LSD: 40.1371

**Log 3** - Final (16kHz Max Frequency, Batch Size 16): * Metrics: Val SI-SDR: 3.78 dB | Mean LSD: 40.13
Validation Loss: 0.2148
Validation SDR: 8.18 dB
Validation SI-SDR: 4.16 dB
Validation Mean LSD: 37.3269

Reducing the batch size allowed the GPU to process 16kHz frequencies, while the synthetic SI-SDR metric appears lower due to the introduction of extreme stochastic augmentation, the model demonstrated significantly better qualitative strength when applied to naturally artifacted stems.


## Visualizations
Model predictions are visualized in `eval.ipynb`. This notebook processes four examples taken from a song of mine (located in `eval_data/`).

Running the notebook generates side-by-side before & after Spectrograms for each example. The visual comparisons clearly display the model masking out vertical transient bleed streaks. The notebook additionally exports the cleanly separated `.wav` files back into the `eval_data/` subdirectories for qualitative listening.

## Limitations and Future Work
1. **Memory Complexity:** The Cross-Track Attention matrix scales quadratically. Processing the full 22.05kHz Nyquist limit requires significant VRAM, so it us inveitably difficult to train without fairly aggressive frequency capping and or extreme gradient accumulation. One of the main issues I faced was VRAM max-out when testing higher max_frequency values (12kHz, 16kHz, etc.)
2. **Generalization on Wild Data:** Stochastic augmentation improved the model's generalization, although there still remain outlier audio tracks with highly complex phase interference which can still result in the model outputting "soft masks" which allow some artifacts to pass through. Future iterations could possibly involve sparse attention mechanisms to handle longer contexts and wider frequency bands.

## Paths to Data and Weights
**Data Paths:**
Because the full dataset is too large, the MUSDB18-HQ dataset is not hosted in the main repository. 
* **Local/Kaggle:** The data can be downloaded locally via the Kaggle API (`quanglvitlm/musdb18-hq`) using 'download_data.py'.

* **Talapas Cluster:** If evaluating on the Talapas cluster, update the path variables at the top of `src/train_model.py` to point to the shared MUSDB directory:

`TRAIN_PATH = "/projects/path/to/shared/musdb18_hq/train"`
`VAL_PATH = "/projects/path/to/shared/musdb18_hq/test"`

**Model Weights:**
The final, fully trained model weights from the third final 16kHz/16-batch training run are stored locally in the repository and are automatically loaded by `eval.ipynb`.
* **Path:** `./weights/train_weights3.pth`

