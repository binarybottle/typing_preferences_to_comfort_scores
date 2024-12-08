# scripts/train_model.py
from pathlib import Path
import logging
import json
from typing import List, Tuple
import yaml

from engram3.models.bayesian import BayesianPreferenceModel
from engram3.data import PreferenceDataset
from engram3.utils import setup_logging
def train_final_model(
    data_path: str,
    config_path: str,
    selected_features: List[str]
) -> BayesianPreferenceModel:
    """
    Train final model using only selected features.
    """
    dataset = PreferenceDataset(data_path)
    config = load_config(config_path)
    
    # Filter dataset to only use selected features
    filtered_dataset = dataset.filter_features(selected_features)
    
    # Split into train/test
    train_data, test_data = filtered_dataset.split_by_participants(
        test_fraction=config['data']['splits']['test_ratio']
    )
    
    # Train model
    model = BayesianPreferenceModel()
    model.fit(train_data)
    
    return model, test_data