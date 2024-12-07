from typing import Dict

from engram3.data import PreferenceDataset

class PreferenceModel:
    def fit(self, dataset: PreferenceDataset) -> None:
        raise NotImplementedError()
    
    def predict_preference(self, bigram1: str, bigram2: str) -> float:
        raise NotImplementedError()