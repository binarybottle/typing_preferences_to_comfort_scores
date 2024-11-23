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
Here, the data is split into training (80% participants) and test (20% participants). 
- The full dataset (100%) is used for:
  - Feature space analysis (does not influence feature selection or model training)
  - Bigram pair recommendations (what additional data to collect)
  - Timing-frequency analysis (timing information is not used to train the model)
- Training data (80%) is used for:
  - Model training
  - Model output (bigram comfort scores)
- Test data (20%) is used to:
  - Feature evaluation (to select features to train the model)
  - Final model performance evaluation

## Directory tree
engram3/
│
├── config.yaml
├── main.py
├── bayesian_modeling.py
├── bigram_feature_definitions.py
├── bigram_feature_extraction.py 
├── bigram_frequency_timing.py
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
    ├── frequency_timing/
    │   ├── analysis.txt
    │   ├── relationship.png
    │   └── groups/
    │
    ├── model/
    │   ├── results/
    │   └── diagnostics_{inference_method}.png
    │
    └── scores/
        └── bigram_comfort_scores.csv

# Installation
pip install -r requirements.txt

# Running the pipeline
python main.py --config config.yaml