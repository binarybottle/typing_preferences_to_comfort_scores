from typing import Dict

from engram3.data import PreferenceDataset
from engram3.models.base import PreferenceModel

class MockPreferenceModel(PreferenceModel):
    def fit(self, dataset: PreferenceDataset) -> None:
        self.preferences = {
            pref.bigram1: pref.typing_time1
            for pref in dataset.preferences
        }
    
    def predict_preference(self, bigram1: str, bigram2: str) -> float:
        return float(self.preferences.get(bigram1, 0) < self.preferences.get(bigram2, 0))