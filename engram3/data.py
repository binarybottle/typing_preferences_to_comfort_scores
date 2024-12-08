from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import pandas as pd
import numpy as np

from engram3.features.definitions import (
    column_map, row_map, finger_map, 
    engram_position_values, row_position_values
)
from engram3.features.extraction import extract_bigram_features

logger = logging.getLogger(__name__)

@dataclass
class Preference:
    """Single preference instance with all needed data."""
    bigram1: str
    bigram2: str
    participant_id: str
    preferred: bool
    features1: Dict[str, float]
    features2: Dict[str, float]
    confidence: Optional[float] = None
    typing_time1: Optional[float] = None
    typing_time2: Optional[float] = None

    def __str__(self) -> str:
        """Return human-readable preference."""
        preferred_bigram = self.bigram1 if self.preferred else self.bigram2
        other_bigram = self.bigram2 if self.preferred else self.bigram1
        return f"'{preferred_bigram}' preferred over '{other_bigram}'"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"Preference('{self.bigram1}' vs '{self.bigram2}', preferred: {self.bigram1 if self.preferred else self.bigram2})"
    
# data.py
class PreferenceDataset:

    REQUIRED_COLUMNS = {
        'bigram1': str,
        'bigram2': str,
        'user_id': str,
        'chosen_bigram': str,
        'bigram1_time': float,
        'bigram2_time': float,
        'chosen_bigram_correct': float,
        'unchosen_bigram_correct': float,
        'abs_sliderValue': float
    }

    def __init__(self, 
                 file_path: Union[str, Path],
                 precomputed_features: Dict[str, Any]):
        """Initialize dataset from CSV file with precomputed features."""
        self.file_path = Path(file_path)
        self.preferences: List[Preference] = []
        self.participants: Set[str] = set()
        
        # Store precomputed features
        self.all_bigrams = precomputed_features['all_bigrams']
        self.all_bigram_features = precomputed_features['all_bigram_features']
        self.feature_names = precomputed_features['feature_names']
        self.samekey_bigrams = precomputed_features['samekey_bigrams']
        self.samekey_bigram_features = precomputed_features['samekey_bigram_features']
        self.samekey_feature_names = precomputed_features['samekey_feature_names']
        
        # Load and process data
        print(f"Loading data from {self.file_path}")
        self._load_csv()    

    def _load_csv(self):
        """Load and validate preference data from CSV."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"\nLoaded CSV with {len(data)} rows")

            # Validate required columns
            missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            success_count = 0
            for idx, row in data.iterrows():
                try:
                    pref = self._create_preference(row)
                    self.preferences.append(pref)
                    self.participants.add(pref.participant_id)
                    success_count += 1
                    
                    if success_count == 1:
                        print("\nFirst successful preference creation:")
                        print(f"Bigram1: {pref.bigram1}")
                        print(f"Bigram2: {pref.bigram2}")
                        print(f"Participant: {pref.participant_id}")
                        print(f"Preferred: {pref.preferred}")
                        print(f"Features1: {pref.features1}")
                        print(f"Features2: {pref.features2}")
                except Exception as e:
                    print(f"\nError processing row {idx}:")
                    print(f"Error: {str(e)}")
                    continue

            print(f"\nSuccessfully processed {success_count} out of {len(data)} rows")

            if not self.preferences:
                raise ValueError("No valid preferences found in data")

            logger.info(f"Loaded {len(self.preferences)} preferences from "
                  f"{len(self.participants)} participants")

        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        if not self.preferences:
            return []
        return list(self.preferences[0].features1.keys())

    def _create_preference(self, row: pd.Series) -> Preference:
        """Create single Preference instance from data row."""
        # Use precomputed features instead of computing them
        bigram1 = (row['bigram1'][0], row['bigram1'][1])
        bigram2 = (row['bigram2'][0], row['bigram2'][1])
        
        features1 = self.all_bigram_features.get(bigram1, {})
        features2 = self.all_bigram_features.get(bigram2, {})
        
        # Add timing and any other non-precomputed features
        features1.update({
            'typing_time': float(row['bigram1_time'])
        })
        features2.update({
            'typing_time': float(row['bigram2_time'])
        })

        return Preference(
            bigram1=str(row['bigram1']),
            bigram2=str(row['bigram2']),
            participant_id=str(row['user_id']),
            preferred=(row['chosen_bigram'] == row['bigram1']),
            features1=features1,
            features2=features2,
            confidence=float(row['abs_sliderValue']),
            typing_time1=float(row['bigram1_time']),
            typing_time2=float(row['bigram2_time'])
        )

    def split_by_participants(self, test_fraction: float = 0.2) -> Tuple['PreferenceDataset', 'PreferenceDataset']:
        """Split into train/test keeping participants separate."""
        # Randomly select participants for test set
        n_test = max(1, int(len(self.participants) * test_fraction))
        test_participants = set(np.random.choice(
            list(self.participants), n_test, replace=False))

        # Split preferences
        train_prefs = []
        test_prefs = []
        
        for pref in self.preferences:
            if pref.participant_id in test_participants:
                test_prefs.append(pref)
            else:
                train_prefs.append(pref)

        # Create new datasets
        train_data = PreferenceDataset.__new__(PreferenceDataset)
        test_data = PreferenceDataset.__new__(PreferenceDataset)
        
        # Set attributes
        for data, prefs in [(train_data, train_prefs), 
                           (test_data, test_prefs)]:
            data.preferences = prefs
            data.participants = {p.participant_id for p in prefs}
            data.file_path = self.file_path

        return train_data, test_data
    
    def check_transitivity(self) -> Dict[str, float]:
        """
        Check for transitivity violations in preferences.
        
        Returns:
            Dict containing:
            - violations: Number of transitivity violations
            - total_triples: Total number of transitive triples checked
            - violation_rate: Proportion of triples that violate transitivity
        """
        # Build preference graph
        pref_graph = {}
        for pref in self.preferences:
            if pref.preferred:
                better, worse = pref.bigram1, pref.bigram2
            else:
                better, worse = pref.bigram2, pref.bigram1
                
            if better not in pref_graph:
                pref_graph[better] = set()
            pref_graph[better].add(worse)

        # Check all possible triples
        violations = 0
        triples = 0
        
        for a in pref_graph:
            for b in pref_graph.get(a, set()):
                for c in pref_graph.get(b, set()):
                    triples += 1
                    # If a > b and b > c, then we should have a > c
                    # If we find c > a, that's a violation
                    if c in pref_graph.get(a, set()):  # Transitive
                        continue
                    if a in pref_graph.get(c, set()):  # Violation
                        violations += 1

        violation_rate = violations / triples if triples > 0 else 0.0
        
        results = {
            'violations': violations,
            'total_triples': triples,
            'violation_rate': violation_rate
        }
        
        logger.info("\nTransitivity check results:")
        logger.info(f"Total transitive triples checked: {triples}")
        logger.info(f"Number of violations: {violations}")
        
        return results
    