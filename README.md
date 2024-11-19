# Engram 3
Author: Arno Klein (binarybottle.com)

GitHub repository: binarybottle/engram3

License: MIT 

The purpose of the scripts in this repository is to take as input data 
ratings of how comfortable one bigram is to type vs. another,
estimate latent bigram typing comfort scores, and use these estimates
to design ergonomically optimized keyboard layouts.

engram3/
├── config.yaml
├── requirements.txt
├── main.py
├── bayesian_modeling.py
├── data_preprocessing.py
├── feature_extraction.py
├── visualization.py
├── bigram_features.py
├── data/
│   └── filtered_bigram_data.csv
└── output/
    ├── analysis/
    ├── logs/
    ├── model/
    ├── scores/
    └── visualizations/

pip install -r requirements.txt

# Create necessary directories
mkdir -p output/{logs,model,scores,visualizations}

# Run the pipeline
python main.py --config config.yaml