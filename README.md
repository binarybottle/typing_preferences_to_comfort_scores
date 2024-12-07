# Engram 3
Author: Arno Klein (binarybottle.com)
GitHub repository: binarybottle/engram3
License: MIT 

## Description
The purpose of the scripts in this repository is to take as input data 
ratings of how comfortable one bigram is to type vs. another,
estimate latent bigram typing comfort scores, and use these estimates
to design ergonomically optimized keyboard layouts.

### Input data
The input data comes from an online bigram comparison typing test 
(https://github.com/binarybottle/bigram-typing-comfort-experiment) 
crowdsourced to hundreds of participants.
Here, the data is split into training (80% participants) and test (20% participants). Different participants are in train/test splits and in cross-validation folds to avoid data leakage, as participants may have consistent typing patterns or preferences that could bias the model if split incorrectly.
- The full dataset (100%) is used for:
  - Feature space analysis (does not influence feature selection or model training)
  - Bigram pair recommendations (what additional data to collect)
- Training data (80%) is used for:
  - Model training
  - Model output (bigram comfort scores)
- Test data (20%) is used to:
  - Feature evaluation (to select features to train the model)
  - Final model performance evaluation

The software is used to conduct two stages: 
1. evaluate features for use in training a Bayesian model
   - raw_metrics.txt: Basic CV scores and metrics for each fold
   - feature_details.txt: Detailed analysis of each feature's behavior
   - feature_metrics.txt: Comprehensive metrics for each feature
   - redundancy_analysis.txt: Analysis of feature correlations and multicollinearity
   - recommendations.txt: Final actionable recommendations
2. train the Bayesian GLMM

## Directory tree
engram3/
│
├── config.yaml
├── main.py
├── bayesian_modeling.py
├── bigram_feature_definitions.py
├── bigram_feature_extraction.py 
├── bigram_pair_feature_evaluation.py
├── bigram_pair_recommendations.py
├── data_processing.py
│── README.md
│
│── data/
│   │
│   ├── input/
│   │   ├── output_all4studies_406participants/
│   │   │   └── tables/
│   │   │       ├── filtered_bigram_data.csv
│   │   │       └── filtered_consistent_choices.csv
│   │   └── output_all4studies_303of406participants_0improbable/
│   │       └── tables/
│   │           ├── filtered_bigram_data.csv  
│   │           └── filtered_consistent_choices.csv
│   └── splits/
│       └── train_test_indices.npz
│
└── output/
    │
    ├── logs/
    │   └── pipeline.log
    │
    ├── feature_evaluation/
    │   └── evaluation_metrics.csv
    │
    ├── feature_space/
    │   ├── analysis.txt
    │   ├── recommendations.txt 
    │   ├── pca.png
    │   ├── bigram_graph.png
    │   └── underrepresented.png
    │
    ├── model/
    │   ├── results/
    │   └── diagnostics_{inference_method}.png
    │
    └── scores/
        └── bigram_comfort_scores.csv

## Scripts
- **`main.py`**: Orchestrates the entire pipeline from data processing to modeling.
- **`bayesian_modeling.py`**: Implements Bayesian modeling for latent bigram typing comfort estimation.
- **`bigram_feature_extraction.py`**: Extracts features from bigrams for analysis.
- **`bigram_pair_feature_evaluation.py`**: Evaluates feature effectiveness using test data.
- **`bigram_pair_recommendations.py`**: Recommends additional bigram pairs for data collection.
- **`data_processing.py`**: Processes raw data into a format suitable for analysis.
- **`config.yaml`**: Configuration file for customizing paths, analysis settings, and model parameters.

## Installation
pip install -r requirements.txt

## Running the pipeline
python main.py --config config.yaml

## Configuration
Key settings can be customized in `config.yaml`:
- **Data settings**: Specify input data files and participant splits.
- **Analysis settings**: Enable or disable frequency, feature space, and evaluation analyses.
- **Model settings**: Configure training parameters like sampling method and evaluation.
- **Logging**: Adjust verbosity and log output location.

## Dependencies
The required dependencies are listed in `requirements.txt`.