
# bigram_typing_preferences_to_comfort_scores: 
Convert typing preferences into comfort scores for bigrams 
and individual keys via Bayesian Preference Learning
================================================================================

https://github.com/binarybottle/bigram_typing_preferences_to_comfort_scores.git

Author: Arno Klein (arnoklein.info)

MIT License. See LICENSE file for details.

## Description
This code converts data like data from the Bigram Typing Preference Study 
(https://github.com/binarybottle/bigram_typing_preference_study) into:
- Comfort scores for bigrams (pairs of keys)
- Comfort scores for individual keys
These scores are used elsewhere to optimize keyboard layouts 
(https://github.com/binarybottle/optimize_layouts).

## Workflow
1. Select important features
2. Train final preference model
3. Generate comfort scores for bigrams and individual keys

## Project Structure
bigram_typing_preferences_to_comfort_scores/                           
- README                    # This file
- config.yaml               # Main configuration file
- main.py                   # Pipeline implementation
- bigram_typing_preferences_to_comfort_scores/                       
   - features/                  
       - analyze_features.py    # Analyze feature metrics to determine importance threshold
       - bigram_frequencies.py  # English language bigrams and bigram frequencies
       - feature_extraction.py  # Core feature computation
       - features.py            # Individual feature calculations
       - keymaps.py             # Keyboard layout definitions
   - models/                  
       - preference_model.stan  # Stan MCMC model file
   - utils/                     
       - config.py              # Configuration validation
       - logging.py             # Logging system
       - visualization.py       # Visualization functions
   - data.py                  # Data processing (participant-aware splitting, feature extraction, caching)
   - model.py                 # Bayesian preference model (participant effects, cv feature selection, uncertainty)
   - recommendations.py       # Bigram pair recommendations (PCA-based feature space analysis & visualization)
- extra/                    # Optional scripts
  - analyze_single_feature.py # Run main.py analyze_feature mode for 1 feature (avoid memory faults)
  - analyze_single_feature.sh # Loop through features with the above script
  - memory-estimator.py       # Estimate memory requirements   
  - pair_recommender.py       # Recommend bigram pairs for further data collection

## Operation Modes
1. Analyze Features
   ```bash
   python main.py --config config.yaml --mode analyze_features [--no-split]```

  - Generates feature importance metrics
  - Evaluates feature interactions
  - Setting the --no-split option uses all participants' data
  - If selecting among features, analyze_features.py can help generate thresholds:

```bash
   python features/analyze_features.py --metrics ../output/data/feature_metrics.csv```

2. Select Features
   ```bash
   python main.py --config config.yaml --mode select_features [--no-split]```

  - Selects optimal feature combinations
  - Participant-aware cross-validation
  - Reports metrics

3. Train Model
   ```bash
   python main.py --config config.yaml --mode train_model```

  - Trains model on selected features (can set selected features in feature_metrics.csv from step 1)
  - Participant-aware validation

4. Recommend Bigram Pairs (if collecting more data)
   ```bash
   python main.py --config config.yaml --mode recommend_bigram_pairs```

  - Generates diverse bigram pairs
  - Feature space coverage analysis
  - PCA-based visualization

5. Predict Bigram Scores
   ```bash
   python main.py --config config.yaml --mode predict_bigram_scores```

  - Predicts comfort scores and uncertainty estimates for bigrams
  - Exports detailed results for optimization

6. Predict Key Scores
   ```bash
   python main.py --config config.yaml --mode predict_key_scores```

  - Predicts individual key comfort scores based on the trained model
  - Derives comfort metrics from bigram model predictions

7. Compute Key Scores from Same-Key Bigrams
   ```bash
   python main.py --config config.yaml --mode compute_key_scores```

  - Analyzes same-key bigram preferences (e.g., "aa" vs "bb")
  - Uses Bradley-Terry model to derive direct key comfort metrics
  - Provides separate rankings from empirical data and model predictions
  - Generates integrated rankings prioritizing empirical same-key data
  
## Configuration
config.yaml controls:
  - Dataset parameters and splits
  - Feature selection settings
  - Stan MCMC parameters
  - Recommendation criteria
  - Logging configuration

## Requirements
  - Python packages: pyyaml, numpy, pandas, scikit-learn, matplotlib, cmdstanpy, adjustText, pydantic, tenacity
  - CmdStan v2.36.0 (special setup for M1/M2 Macs -- see below)

## Notes on Stan installation
Install Stan on Macos M1/M2 (or through Rosetta):
1. Create directory
   mkdir -p ~/.cmdstan
   cd ~/.cmdstan
2. Clone CmdStan
   git clone --depth 1 --branch v2.36.0 https://github.com/stan-dev/cmdstan.git cmdstan-2.36.0
   cd cmdstan-2.36.0
3. Initialize submodules
   git submodule update --init --recursive
4. Create make/local with very specific Apple Silicon settings
   cat > make/local << EOL
   STAN_OPENCL=false
   CC=/usr/bin/clang
   CXX=/usr/bin/clang++
   CFLAGS=-arch arm64 -target arm64-apple-darwin -isysroot $(xcrun --show-sdk-path)
   CXXFLAGS=-arch arm64 -target arm64-apple-darwin -isysroot $(xcrun --show-sdk-path)
   LDFLAGS=-arch arm64 -target arm64-apple-darwin
   STAN_FLAG_MARCH="-arch arm64"
   GTEST_CXXFLAGS="-arch arm64"
   O=3
   BUILD_ARCH=arm64
   OS=Darwin
   TBB_INTERFACE_NEW=TRUE
   EOL
5. Clean and build
   make clean-all
   STAN_CPU_ARCH=arm64 make build -j4

Note: Variations in Stan results are due to:
- MCMC sampling being inherently probabilistic
- Random initialization of Stan's MCMC chains
- Random train/test data splits using numpy's random seed
While the seed is set for reproducibility at each run, Stan chains operate independently 
and can explore the parameter space differently each time. This leads to slightly different 
posterior distributions and therefore different metrics.
These variations are expected and normal in Bayesian inference.
