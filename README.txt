# Engram3: Keyboard Layout Optimization via Preference Learning

A system for optimizing keyboard layouts using Bayesian preference learning and active sampling.

## Project Structure

engram3/                            # Root project directory
├── pyproject.toml                  # Project metadata and dependencies
├── config.yaml                     # Configure settings
├── main.py                         # Main script
└── engram3/                        # Package directory
    ├── features/
    │   ├── bigram_frequencies.py   # Bigrams and bigram frequencies in the English language
    │   ├── extraction.py           # Feature computation and interactions
    │   ├── features.py             # Individual feature calculations
    │   ├── keymaps.py              # Keyboard layout mappings
    │   └── visualization.py        # FeatureMetricsVisualizer for visualizing feature measures
    ├── data.py                     # PreferenceDataset class for data loading/handling 
    ├── model.py                    # PreferenceModel class for modeling/evaluation
    ├── recommendations.py          # BigramRecommender class for generating bigram pair recommendations
    └── config.yaml                 # Configuration settings

engram3/
├── utils/
│   ├── __init__.py
│   ├── visualization.py      # Add PlottingUtils class
│   └── logging_utils.py      # Add LoggingManager class
├── features/
│   ├── __init__.py
│   ├── feature_extraction.py # (Already created)
│   ├── importance.py        # Add FeatureImportanceCalculator class
│   └── ...

## Usage

The system operates in four main modes:

1. Select Features
```bash
python main.py --config config.yaml --mode select_features

Identifies important typing comfort features through cross-validation.

2. Generate Recommendations
```python main.py --config config.yaml --mode recommend_bigram_pairs```
Suggests new bigram pairs to evaluate, optimizing for information gain.

3. Train Model
```python main.py --config config.yaml --mode train_model```
Trains the final preference model using selected features.

4. Predict Bigram Scores
```python main.py --config config.yaml --mode predict_bigram_scores```
Calculates absolute comfort scores for all possible bigrams.

1. Select important features
2. Collect more data for those features
3. Train the final model
4. Use the model to score all possible bigrams

The output would be a comprehensive CSV containing comfort scores and uncertainties 
for every possible bigram, which could then be used for keyboard layout optimization.

## Configuration
See config.yaml for detailed settings including:
  - Data paths and splits
  - Feature selection parameters
  - Model hyperparameters
  - Recommendation criteria weights
  
## Model Details
Uses a hierarchical Bayesian preference learning model with:
  - Feature-based comfort scores
  - Participant-level random effects
  - Bradley-Terry preference structure
  - Full posterior uncertainty quantification

## Implementation
Built using Stan for robust Bayesian inference, providing:
  - Excellent hierarchical model support
  - Stable MCMC sampling
  - Comprehensive diagnostics

## Key Measures of importance (influence on preferences) and stability (consistency of effects)

  - Model Effect (mean ± std): Feature impact strength and consistency
    - Mean: Average effect size of the feature across MCMC samples
    - Std: Standard deviation of the effect
    - Purpose: Quantifies how strongly and consistently the feature influences typing preferences
  - Effect CV (Coefficient of Variation)
    - Calculation: std_effect / abs(mean_effect)
    - Purpose: Measures relative stability of feature effect, normalized by effect size
    - Interpretation: Lower values indicate more stable effects
  - Relative Range
    - Calculation: (max_effect - min_effect) / abs(mean_effect)
    - Purpose: Shows total spread of effects relative to mean effect size
    - Interpretation: Lower values indicate more consistent effects
  - Feature Stability: Reliability across cross-validation
  - Interaction Strength: Feature interaction importance
  - Coverage: Feature space exploration
  - Transitivity: Preference consistency validation
- Combined Score
  - Components:
    - Inclusion probability (probability that feature weight is meaningfully different from 0)
    - Effect magnitude (strength of feature's impact)
    - Effect consistency (stability of feature's direction)
    - Predictive impact (improvement in model predictions when feature is included)
  - Purpose: Single metric for overall feature importance, used for selection decisions



