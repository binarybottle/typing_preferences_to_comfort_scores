# Engram 3
Author: Arno Klein (binarybottle.com)

GitHub repository: binarybottle/engram3

License: MIT 

The purpose of the scripts in this repository is to take as input data 
ratings of how comfortable one bigram is to type vs. another,
estimate latent bigram typing comfort scores, and use these estimates
to design ergonomically optimized keyboard layouts.

project_root/
├── data/
│   ├── raw/                                # Original data files
│   ├── splits/
│   │   └── train_test_indices.npz          # Saved train/test split indices
│   └── processed/                          # Final processed data
│
├── output/
│   ├── logs/
│   │   └── pipeline.log
│   │
│   ├── feature_evaluation/                 # Feature evaluation results
│   │   ├── analysis/
│   │   │   ├── evaluation_results.csv      # Numerical results
│   │   │   ├── feature_importance.csv
│   │   │   └── feature_correlations.csv
│   │   └── plots/
│   │       ├── feature_importance.png
│   │       ├── feature_correlations.png
│   │       └── cv_scores.png
│   │
│   ├── feature_space/                      # Feature space analysis
│   │   ├── analysis.txt                    # Combined analysis results
│   │   ├── recommendations.txt
│   │   ├── pca.png
│   │   ├── bigram_graph.png
│   │   └── underrepresented.png
│   │
│   ├── timing_frequency/                   # Timing analysis
│   │   ├── analysis.txt
│   │   ├── relationship.png
│   │   ├── groups_boxplot.png
│   │   └── groups_violin.png
│   │
│   ├── model/                             # Model outputs
│   │   ├── results_mcmc_trace.nc
│   │   ├── results_mcmc_point_estimates.csv
│   │   ├── results_mcmc_model_info.json
│   │   ├── diagnostics_mcmc.png
│   │   ├── forest_mcmc.png
│   │   # (Similar files for variational inference with 'vi' instead of 'mcmc')
│   │
│   └── scores/
│       └── bigram_comfort_scores.csv
│
├── __init__.py
│── main.py
│── config.yaml
│── feature_evaluation.py
│── analysis_visualization.py
│── bayesian_modeling.py
│── data_preprocessing.py
│── bigram_feature_extraction.py
│── bigram_features.py
├── requirements.txt
├── setup.py
└── README.md

pip install -r requirements.txt

# Create necessary directories
mkdir -p output/{logs,model,scores,visualizations}

# Run the pipeline
python main.py --config config.yaml