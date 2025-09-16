# IMDB-Sentiment-Pulse

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/SaifSiddique009/DiaPredictML/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SaifSiddique009/DiaPredictML/pulls)

A sentiment analysis project on IMDB movie reviews, demonstrating a junior ML engineer's workflow from data exploration to model deployment. This project uses classical machine learning baselines, advanced ensemble models, and deep learning architectures (RNN, LSTM, GRU) to classify reviews as positive or negative. It emphasizes best practices like cross-validation, hyperparameter tuning with Optuna, and modular code structure.

## Project Overview

This project analyzes the IMDB movie reviews dataset to perform binary sentiment classification (positive/negative). The goal is to showcase a structured ML pipeline, including enhanced exploratory data analysis (EDA) to inform model selection, baseline comparisons, advanced ML models, and sequential deep learning models. Key emphases include handling text data with TF-IDF and embeddings, robust evaluation via 5-fold cross-validation, and optimization using Optuna's Bayesian tuner.

The dataset consists of 50K comma-separated reviews (columns: Review, Sentiment) encoded as 0 (negative) or 1 (positive). The project evolves from simple ML baselines to sophisticated RNN variants, highlighting performance improvements.

## Key Features

- **Enhanced EDA**: Includes review length distributions, sentiment balances, word clouds, and n-grams to justify model choices (e.g., TF-IDF for ML, sequences for RNNs).
- **Baseline Models**: Logistic Regression and Naive Bayes for quick benchmarks.
- **Advanced ML Models**: Random Forest and XGBoost with Optuna hyperparameter tuning.
- **Deep Learning Models**: Simple RNN, LSTM, and GRU with embeddings, dropout, and early stopping.
- **Evaluation**: 5-fold stratified cross-validation, metrics (Accuracy, F1-score), and visualization of training histories.
- **Modular Design**: Separated modules for data loading, preprocessing, models, evaluation, and utilities.
- **Reproducibility**: Logging, random seeds, and script orchestration via `main.py`.
- **Notebooks**: An exploratory EDA notebook in `notebooks/` for interactive analysis.

## Repository Structure

```
imdb-sentiment-pulse/
├── data/                       # Dataset files
│   └── IMDBSentimentData.csv   # IMDB reviews dataset (tab-separated)
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and splitting
│   ├── eda.py                  # Enhanced EDA with visualizations
│   ├── preprocessing.py        # Text vectorization (TF-IDF, tokenization)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ml_models.py        # ML baselines and advanced models
│   │   └── dl_models.py        # Deep learning models (RNN, LSTM, GRU)
│   ├── evaluation.py           # Cross-validation, Optuna tuning, metrics
│   ├── utils.py                # Plotting and helper functions
├── notebooks/                  # Jupyter notebooks for exploration
│   └── exploratory_eda.ipynb   # Interactive EDA based on src/eda.py
├── logs/                       # Generated log files (e.g., project.log)
├── results/                    # Output artifacts
│   ├── model_metrics.csv       # CV metrics for all models
│   ├── review_length_dist.png  # EDA plots (e.g., length distribution)
│   ├── ...                     # Word clouds, training curves, etc.
├── main.py                     # Main script to run the pipeline
├── pyproject.toml              # Dependencies managed via UV
├── requirements.txt            # Fallback dependencies for pip
├── README.md                   # This file
└── .gitignore                  # Git ignore patterns
```

## Prerequisites

- Python 3.10 or higher
- Access to a GPU (optional, for faster DL training; CPU suffices for small dataset)
- Dataset: Ensure `data/IMDBSentimentData.csv` is present. If missing, download from [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (use the smaller subset if needed, rename to match, and ensure tab-separated format).

## Installation

### Using UV (Recommended for Modern Project Management)

UV is a fast, lightweight tool for managing Python projects, handling virtual environments and dependencies.

1. Install UV:
   ```
   pip install uv
   ```

2. Clone the repository:
   ```
   git clone https://github.com/SaifSiddique009/IMDB-Sentiment-Pulse.git
   cd IMDB-Sentiment-Pulse
   ```

3. Sync dependencies (creates `.venv` automatically):
   ```
   uv sync
   ```

### Using PIP (Fallback)

If UV is unavailable:

1. Clone the repository (as above).

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies from `pyproject.toml` (or manually):
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Locally

1. Ensure the dataset is in `data/IMDBSentimentData.csv`.

2. Run the main pipeline:
   ```
   uv run python main.py  # Or python main.py if using PIP
   ```

   - This loads data, performs EDA, preprocesses, trains/evaluates models, and saves results/logs.
   - Outputs: Metrics in `results/model_metrics.csv`, plots in `results/`, logs in `logs/`.

3. Explore interactively: Open `notebooks/exploratory_eda.ipynb` in Jupyter for EDA visualizations.

### Running on Google Colab
You can run this project in Google Colab for easy demonstration (uses pip).

1. Open a new notebook: [Google Colab](https://colab.research.google.com).
2. Clone and set up:
   ```bash
   !git clone https://github.com/SaifSiddique009/IMDB-Sentiment-Pulse.git
   %cd IMDB-Sentiment-Pulse
   !pip install -r requirements.txt
   ```
3. Run the main script:
   ```python
   !python main.py
   ```

   Note: DL models may run slower without GPU; enable runtime type "GPU" in Colab settings.

## Results

After running, check `results/` for:

- **model_metrics.csv**: Table with 5-fold CV Accuracy and F1-scores, e.g.:

  | Model              | CV_Acc | CV_F1 |
  |--------------------|--------|-------|
  | LogisticRegression | 0.85   | 0.84  |
  | NaiveBayes         | 0.82   | 0.81  |
  | RandomForest       | 0.88   | 0.87  |
  | XGBoost            | 0.89   | 0.88  |
  | RNN                | 0.75   | 0.74  |
  | LSTM               | 0.86   | 0.85  |
  | GRU                | 0.87   | 0.86  |

- **Plots**: Training loss/accuracy curves (e.g., `LSTM_acc.png`), EDA visuals (word clouds, distributions).

Overall, advanced ML (XGBoost) and DL (GRU/LSTM) outperform baselines, with embeddings improving sequence handling. Test accuracies typically range 80-90% on this dataset.

## Future Improvements

- **Data Augmentation**: Add techniques like synonym replacement or back-translation to handle small dataset size.
- **Advanced Preprocessing**: Incorporate stemming/lemmatization or pre-trained embeddings (e.g., GloVe).
- **Model Enhancements**: Experiment with transformers (e.g., BERT) for state-of-the-art performance.
- **Deployment**: Containerize with Docker and deploy as a Streamlit app for inference.
- **CI/CD**: Add GitHub Actions for automated testing/linting.

## Contributing

Contributions welcome! Fork the repo, create a branch, and submit a pull request. Follow PEP8 style and add tests where possible.
