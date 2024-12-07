# engram3/models/bayesian.py
from typing import Dict, List
import numpy as np
from scipy import stats
import pymc as pm

from engram3.data import PreferenceDataset
from engram3.models.base import PreferenceModel

class BayesianPreferenceModel(PreferenceModel):
    """
    Bayesian preference learning model that:
    - Models latent comfort scores through features
    - Maintains transitivity by construction
    - Can include participant effects
    - Learns interpretable feature weights
    """
    
    def __init__(self, include_participant_effects: bool = True):
        self.include_participant_effects = include_participant_effects
        self.feature_names: List[str] = []
        self.feature_weights = None
        self.participant_effects = None
        self.model = None
        self.trace = None
        
    def fit(self, dataset: PreferenceDataset) -> None:
        """
        Fit the model using MCMC.
        """
        self.feature_names = dataset.get_feature_names()
        n_features = len(self.feature_names)
        n_participants = len(dataset.participants)
        
        # Prepare data
        X1 = []  # Features for first bigram in each pair
        X2 = []  # Features for second bigram in each pair
        y = []   # 1 if first preferred, 0 if second preferred
        participant_idx = []  # Participant indices
        
        for pref in dataset.preferences:
            feat1 = [pref.features1[f] for f in self.feature_names]
            feat2 = [pref.features2[f] for f in self.feature_names]
            X1.append(feat1)
            X2.append(feat2)
            y.append(float(pref.preferred))
            participant_idx.append(list(dataset.participants).index(pref.participant_id))
            
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)
        participant_idx = np.array(participant_idx)
        
        # Build model
        with pm.Model() as self.model:
            # Feature weights (regularized)
            weights = pm.Normal('weights', mu=0, sigma=1, shape=n_features)
            
            # Optional participant effects
            if self.include_participant_effects:
                participant_effects = pm.Normal('participant_effects', 
                                             mu=0, sigma=0.5, 
                                             shape=n_participants)
            
            # Comfort scores for each bigram in pairs
            comfort1 = pm.Deterministic('comfort1', pm.math.dot(X1, weights))
            comfort2 = pm.Deterministic('comfort2', pm.math.dot(X2, weights))
            
            # Add participant effects if included
            if self.include_participant_effects:
                comfort1 = comfort1 + participant_effects[participant_idx]
                comfort2 = comfort2 + participant_effects[participant_idx]
            
            # Preference likelihood
            p = pm.math.sigmoid(comfort1 - comfort2)
            pm.Bernoulli('y', p=p, observed=y)
            
            # Sample from posterior
            self.trace = pm.sample(2000, tune=1000, return_inferencedata=False)
        
        # Store mean parameter values
        self.feature_weights = dict(zip(
            self.feature_names,
            np.mean(self.trace['weights'], axis=0)
        ))
        
        if self.include_participant_effects:
            self.participant_effects = dict(zip(
                list(dataset.participants),
                np.mean(self.trace['participant_effects'], axis=0)
            ))
    
    def predict_preference(self, bigram1: str, bigram2: str) -> float:
        """Predict probability of preferring bigram1 over bigram2."""
        score1 = self.get_comfort_score(bigram1)
        score2 = self.get_comfort_score(bigram2)
        return 1 / (1 + np.exp(-(score1 - score2)))
    
    def get_comfort_score(self, bigram: str) -> float:
        """Get underlying comfort score for a bigram."""
        if not hasattr(self, 'feature_weights'):
            raise RuntimeError("Model must be fit before getting comfort scores")
        # Note: This will need feature extraction for the bigram
        # For now returning placeholder
        return 0.0
        
    def get_feature_weights(self) -> Dict[str, float]:
        """Get learned feature importance weights."""
        if self.feature_weights is None:
            raise RuntimeError("Model must be fit before getting feature weights")
        return self.feature_weights.copy()