# First, run feature selection
python main.py --config config.yaml --mode select_features --n_repetitions 10

# Second, generate recommendations
python main.py --config config.yaml --mode recommend_bigram_pairs

# Third, train the model using selected features
python main.py --config config.yaml --mode train_model

engram3/                        # Root project directory
├── pyproject.toml
├── config.yaml
├── main.py
└── engram3/                    # Package directory
    ├── __init__.py
    ├── data.py                 # Dataset handling
    ├── utils.py
    ├── features/               # Feature-related code
    │   ├── __init__.py
    │   ├── definitions.py
    │   ├── extraction.py
    │   ├── recommendations.py  # Recommend bigram pairs
    │   └── selection.py
    ├── models/                 # Model-related code
    │   ├── __init__.py
    │   └── bayesian.py
    └── tests/                  # Tests
        ├── __init__.py
        └── test_models.py