# First, run feature selection
python main.py --config config.yaml --mode select_features --n_repetitions 10

# Second, generate recommendations
python main.py --config config.yaml --generate_recommendations

# Third, train the model using selected features
python main.py --config config.yaml --mode train_model

engram3/                        # Root project directory
├── pyproject.toml
├── config.yaml
├── main.py
└── engram3/                    # Package directory
    ├── __init__.py
    ├── data.py                 # Dataset handling
    ├── analysis.py             # General analysis functions
    ├── utils.py
    ├── features/               # Feature-related code
    │   ├── __init__.py
    │   ├── definitions.py
    │   ├── extraction.py
    │   └── recommendations.py  # Recommend bigram pairs
    ├── models/                 # Model-related code
    │   ├── __init__.py
    │   ├── base.py
    │   ├── simple.py           # Mock model
    │   ├── bayesian.py
    │   └── utils.py
    └── tests/                  # Tests
        ├── __init__.py
        └── test_models.py