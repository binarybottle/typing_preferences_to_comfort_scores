from pathlib import Path
import pytest

from engram3.data import PreferenceDataset
from engram3.models.simple import MockPreferenceModel

def test_model_pipeline():
    # Use a small test dataset
    test_file = Path("tests/data/test_preferences.csv")  # You'll need to create this
    dataset = PreferenceDataset(test_file)
    model = MockPreferenceModel()
    model.fit(dataset)
    pred = model.predict_preference("th", "he")
    assert isinstance(pred, float)
    assert 0 <= pred <= 1