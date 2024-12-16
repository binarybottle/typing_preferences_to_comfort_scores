Engram3: Keyboard Layout Optimization via Preference Learning
==========================================================

A system for optimizing keyboard layouts using Bayesian preference learning and active sampling.

Project Structure
----------------
engram3/                            # Root project directory
├── pyproject.toml                  # Project metadata and dependencies
├── config.yaml                     # Configuration settings
├── main.py                         # CLI and pipeline orchestration
└── engram3/                        # Package directory
    ├── features/                   # Feature computation and analysis
    │   ├── __init__.py
    │   ├── bigram_frequencies.py   # Bigram frequency data
    │   ├── extraction.py           # Feature extraction pipeline
    │   ├── features.py            # Core feature calculations
    │   ├── importance.py          # Feature importance analysis
    │   └── keymaps.py             # Keyboard layout definitions
    ├── utils/                      # Utility modules
    │   ├── __init__.py
    │   ├── config.py              # Configuration management
    │   ├── visualization.py       # Plotting utilities
    │   └── logging.py            # Logging system
    ├── data.py                    # Dataset management
    ├── model.py                   # Bayesian preference model
    └── recommendations.py         # Bigram pair recommendations

Core Components
--------------
1. Feature System
   - Extraction of physical and ergonomic typing features
   - Feature importance analysis and selection
   - Interaction detection and evaluation
   - Dynamic feature computation and caching

2. Preference Model
   - Hierarchical Bayesian model using Stan
   - Bradley-Terry preference structure
   - Participant-level random effects
   - Full posterior uncertainty quantification

3. Recommendation Engine
   - Information-theoretic pair selection
   - Multi-criteria scoring system
   - Feature space coverage optimization
   - Transitivity validation

Operation Modes
--------------
1. Feature Selection (select_features)
   $ python main.py --config config.yaml --mode select_features
   - Evaluates base features and interactions
   - Uses adaptive thresholding
   - Performs stability analysis
   - Outputs comprehensive metrics

2. Model Training (train_model)
   $ python main.py --config config.yaml --mode train_model
   - Trains on selected features
   - Handles participant grouping
   - Includes cross-validation
   - Provides detailed diagnostics

3. Bigram Recommendations (recommend_bigram_pairs)
   $ python main.py --config config.yaml --mode recommend_bigram_pairs
   - Generates candidate pairs
   - Optimizes information gain
   - Visualizes recommendations
   - Exports for data collection

4. Comfort Prediction (predict_bigram_scores)
   $ python main.py --config config.yaml --mode predict_bigram_scores
   - Scores all possible bigrams
   - Includes uncertainty estimates
   - Provides detailed metrics
   - Exports comprehensive results

Configuration
------------
The config.yaml file controls all aspects of the system:

1. Data Management
   - Input/output paths
   - Train/test splits
   - Data validation rules

2. Feature Selection
   - Base features list
   - Interaction parameters
   - Selection thresholds
   - Stability criteria

3. Model Parameters
   - MCMC settings
   - Prior specifications
   - Diagnostic thresholds

4. Recommendation Settings
   - Scoring weights
   - Candidate generation
   - Visualization options

Implementation Details
--------------------
1. Feature Selection Process
   - Iterative selection with interaction consideration
   - Adaptive thresholding using distribution statistics
   - Multiple testing correction
   - Stability assessment

2. Model Architecture
   - Stan-based implementation
   - Efficient MCMC sampling
   - Comprehensive diagnostics
   - Uncertainty propagation

3. Output Metrics
   - Feature importance scores
   - Model effect sizes
   - Stability measures
   - Statistical significance

Workflow
--------
1. Select important features
2. Collect targeted preference data
3. Train final preference model
4. Generate bigram comfort scores
5. Use scores for layout optimization

The system produces detailed CSVs and visualizations at each stage to guide the 
optimization process.

Requirements
-----------
- Python 3.8+
- Stan 2.26+
- NumPy
- Pandas
- Matplotlib
- PyYAML
- Scikit-learn

Installation
-----------
1. Clone the repository:
   git clone https://github.com/yourname/engram3.git

2. Install dependencies:
   pip install -r requirements.txt

3. Install Stan:
   Follow instructions at https://mc-stan.org/users/interfaces/cmdstan

4. Configure:
   Copy config.yaml.example to config.yaml and adjust settings

5. Run:
   python main.py --config config.yaml --mode select_features

License
-------
MIT License. See LICENSE file for details.




## Feature selection process
One feature is selected at each iteration if it meets ALL these criteria:
  1. highest importance_score among remaining features
  2. importance_score >= adaptive threshold

  One feature is selected due to interaction effects 
  (adding one can change the importance of others, etc.).

The adaptive threshold is calculated as:
  - threshold = mean + importance_threshold * std
  - threshold = np.clip(threshold, min_threshold, max_threshold)
where:
  - importance_threshold set in config.yaml (feature_selection:threshold:importance)
  - threshold range is fixed between min_threshold and max_threshold (25th and 75th percentiles), 
    so that it stays within reasonable bounds based on the distribution of the importance scores

## Feature selection output
  - feature_name: Name of the feature or interaction feature being evaluated
  - n_components: Number of components in the feature (1 for base features, 2+ for interaction features)
  - selected: Binary indicator (1 or 0) showing if the feature was selected in the final model
  - importance_score: Combined score from the prediction and variation metrics below, indicating overall feature importance
  - Prediction/Effect measures: 
    - model_effect: Absolute size of the feature's effect in the model
    - predictive_power: Model performance improvement from feature
  - Variation/Stability measures: 
    - effect_consistency: How consistent the feature's effect is across cross-validation splits
      - (1 - coefficient of variation: 1 - std_effect / abs(mean_effect)
  - Resulting model weights: 
    - weight: Mean weight/coefficient of the feature in the model
    - weight_std: Standard deviation of the feature's weight (uncertainty measure)

For prediction, only the trained model weights (mean and std) are needed.

For recommendation, both model weights and the feature selection metrics are used to help identify informative pairs:
  - prediction uncertainty
  - comfort score uncertainty
  - feature space coverage
  - MI differences
  - stability
  - interaction effects
  - transitivity

