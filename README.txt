# First, run feature selection
python main.py --config config.yaml --mode select_features --n_repetitions 10

# Second, generate recommendations
python main.py --config config.yaml --mode recommend_bigram_pairs

# Third, train the model using selected features
python main.py --config config.yaml --mode train_model

engram3/                        # Root project directory
├── pyproject.toml              # Python poetry settings
├── config.yaml                 # Configure settings
├── main.py                     # Main script
└── engram3/                    # Package directory
    ├── __init__.py
    ├── data.py                 # Dataset handling
    ├── model.py                # Preference model
    ├── utils.py                # Utility functions
    ├── features/               # Feature-related code
    │   ├── __init__.py
    │   ├── bigram_frequencies.py
    │   ├── keymaps.py
    │   ├── features.py
    │   ├── extraction.py
    │   └── recommendations.py  # Recommend bigram pairs
    └── tests/                  # Tests
        ├── __init__.py
        └── test_models.py

## Key measures and their purposes:

Model effect (mean ± std)
Mean: Average effect size of the feature across cross-validation folds
Std: Standard deviation of the effect, showing variability across folds
Purpose: Shows how strongly and consistently the feature influences typing preferences

Correlation
Direct correlation between feature differences and user preferences
Purpose: Shows simpler linear relationship between feature and preferences, without controlling for other features

Mutual information
Non-linear measure of dependency between feature and preferences
Purpose: Can capture more complex relationships that correlation might miss

Effect CV (Coefficient of Variation)
Calculated as std_effect / abs(mean_effect)
Purpose: Measures relative stability of feature effect across folds, normalized by effect size
Lower values indicate more stable effects

Relative range
Calculated as (max_effect - min_effect) / abs(mean_effect)
Purpose: Another stability measure showing the total spread of effects relative to mean effect size
Lower values indicate more consistent effects

Combined score
Weighted combination of model effect, correlation, and mutual information
Purpose: Single metric for overall feature importance, currently used for selection decisions
These metrics together help evaluate features on both:

Importance (how much they influence preferences)
Stability (how reliable/consistent their effects are)