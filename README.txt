Engram3: Keyboard Layout Optimization via Preference Learning
=============================================================
A system for optimizing keyboard layouts using Bayesian preference learning and active sampling.
https://github.com/yourname/engram3.git

Author: Arno Klein (binarybottle.com)

Project Structure
----------------
engram3/                           
├── config.yaml                    # Main configuration file
├── main.py                        # CLI and pipeline orchestration
└── engram3/                       
    ├── features/                  
        ├── feature_extraction.py  # Core feature computation
        ├── feature_importance.py  # Feature evaluation metrics
        ├── keymaps.py             # Keyboard layout mappings
    ├── utils/                     
        ├── config.py              # Configuration handling
        ├── logging.py             # Logging system
        └── caching.py             # Feature and prediction caching
    ├── data.py                    # Dataset and preference handling
    ├── model.py                   # Bayesian preference model
    └── recommendations.py         # Bigram pair recommendations

Core Components
--------------

3. Recommendation Engine
   - Multi-criteria pair scoring
   - Configurable scoring weights
   - Feature space coverage analysis
   - Transitivity validation
   - Recommendation visualization

Operation Modes
--------------
1. Feature Selection (select_features)
   $ python main.py --config config.yaml --mode select_features
   - Evaluates features using three metrics
   - Handles feature interactions
   - Performs cross-validation stability analysis
   - Generates comprehensive metrics report

2. Model Training (train_model)
   $ python main.py --config config.yaml --mode train_model
   - Trains using selected features
   - Maintains participant separation in splits
   - Performs cross-validation
   - Saves model state for reuse

3. Bigram Recommendations (recommend_bigram_pairs)
   $ python main.py --config config.yaml --mode recommend_bigram_pairs
   - Scores candidate pairs using multiple criteria
   - Visualizes recommendations in feature space
   - Validates transitivity
   - Exports recommendations for data collection

4. Comfort Prediction (predict_bigram_scores)
   $ python main.py --config config.yaml --mode predict_bigram_scores
   - Predicts comfort scores for all bigrams
   - Includes uncertainty quantification
   - Exports detailed scoring results

Configuration
------------
The config.yaml file controls all aspects of the system:

1. Data Configuration
   - Input file paths
   - Train/test split ratio
   - Layout character definitions

2. Feature Selection
   - Metric weights for selection
   - Base features and interactions
   - Output file paths

3. Model Parameters
   - MCMC settings (fast/slow options)
   - Chains and samples configuration
   - Adaptation parameters

4. Recommendation Settings
   - Scoring weights for different criteria
   - Number of recommendations
   - Maximum candidate pairs
   - Visualization parameters

5. Logging Configuration
   - Console and file logging levels
   - Output directory structure
   - Logging format

Workflow
--------
1. Select important features
2. Collect targeted preference data
3. Train final preference model
4. Generate bigram comfort scores
5. Use scores for layout optimization

License
-------
MIT License. See LICENSE file for details.
