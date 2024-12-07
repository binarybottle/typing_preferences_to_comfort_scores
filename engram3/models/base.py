# engram3/models/base.py
from typing import Dict

from engram3.data import PreferenceDataset

class PreferenceModel:
    def fit(self, dataset: PreferenceDataset) -> None:
        """Train the model on preference data."""
        raise NotImplementedError()
    
    def predict_preference(self, bigram1: str, bigram2: str) -> float:
        """Predict probability of preferring bigram1 over bigram2."""
        raise NotImplementedError()
        
    def get_comfort_score(self, bigram: str) -> float:
        """Get underlying comfort score for a bigram."""
        raise NotImplementedError()
        
    def get_feature_weights(self) -> Dict[str, float]:
        """Get learned feature importance weights."""
        raise NotImplementedError()