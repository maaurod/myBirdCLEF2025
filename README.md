# BirdCLEF 2025: Avian Vocalization Classification

This repository contains a multi-stage machine learning pipeline for the BirdCLEF 2025 Kaggle competition. The goal is to identify bird species in soundscape recordings using log-mel spectrograms and deep learning models.

## Pipeline Overview

The project is structured into several modular Jupyter notebooks, following a logical progression from data exploration to submission:

1.  **Exploratory Data Analysis (EDA)**: [`eda-birdclef2025.ipynb`](./eda-birdclef2025.ipynb)
    - Initial data inspection, taxonomy analysis, and recording location visualization.
    - Audio sample loading and basic spectrogram plotting.

2.  **Data Preprocessing**: [`precomputing-spectrograms.ipynb`](./precomputing-spectrograms.ipynb)
    - Converts raw `.ogg` audio files into fixed-length (5s) log-mel spectrogram chunks.
    - Implements parallel processing and Voice Activity Detection (VAD) to filter silent or human-voice-contaminated segments.
    - Saves chunks as `.npy` files for efficient loading during training.

3.  **Model Training**: [`training-birdclef2025.ipynb`](./training-birdclef2025.ipynb)
    - Sets up the core training environment and `CFG` parameters.
    - Defines the dataset, augmentations (Time/Freq masking, Mixup), and model architecture (EfficientNet-based SED).
    - Implements the training loop with BCE/Focal loss and cosine annealing scheduler.

4.  **Pseudo-Labeling**: [`pseudo-labeling-birdclef2025.ipynb`](./pseudo-labeling-birdclef2025.ipynb)
    - Uses a trained model to generate "soft" labels for unlabeled soundscape data.
    - Enables semi-supervised learning by expanding the training pool.

5.  **Curriculum Training & HNM**: [`curriculum-training-hnm-birdclef2025.ipynb`](./curriculum-training-hnm-birdclef2025.ipynb)
    - Advanced training stage using a curriculum strategy (gradually introducing pseudo-labels).
    - Implements Hard Negative Mining (HNM) to specifically target and correct model mistakes.

6.  **Inference & Submission**: [`inference-birdclef2025.ipynb`](./inference-birdclef2025.ipynb)
    - End-to-end inference pipeline on test soundscapes.
    - Employs an ensemble of model weights for robust predictions.
    - Formats the final results for Kaggle submission.

## Requirements

The project relies on standard deep learning and audio processing libraries:
- `torch`, `torchaudio`, `timm`, `transformers`
- `pandas`, `numpy`, `scikit-learn`
- `librosa`, `joblib`, `tqdm`

## Future Local Adaptation

To run this pipeline locally (outside of Kaggle):
1.  Establish a data directory: `./input/birdclef-2025/`.
2.  Update the `CFG` classes in each notebook to resolve paths relative to the project root.
3.  Ensure GPU support via CUDA for efficient training and inference.