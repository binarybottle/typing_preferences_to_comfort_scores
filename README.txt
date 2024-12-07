engram3/                    # Root project directory
├── pyproject.toml
├── config.yaml
├── main.py
└── engram3/               # Package directory
    ├── __init__.py
    ├── data.py
    ├── analysis.py
    ├── utils.py
    ├── features/         # Feature-related code
    │   ├── __init__.py
    │   ├── definitions.py
    │   └── extraction.py
    ├── models/          # Model-related code
    │   ├── __init__.py
    │   ├── base.py
    │   ├── simple.py    # Add this for our mock model
    │   ├── bayesian.py
    │   └── utils.py
    └── tests/          # Tests directory should be inside package
        ├── __init__.py
        └── test_models.py