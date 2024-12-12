# data.py
"""
Data loading and preprocessing functionality for typing preference data.
Implements PreferenceDataset class which handles:
  - Loading and validating raw preference data
  - Computing and storing bigram features
  - Managing train/test splits
  - Providing consistent data access for modeling
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import pandas as pd
import numpy as np

from engram3.features.keymaps import *
from engram3.features.extraction import extract_bigram_features, extract_same_letter_features

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
                 column_map: Dict[str, int],
                 row_map: Dict[str, int],
                 finger_map: Dict[str, int],
                 engram_position_values: Dict[str, float],
                 row_position_values: Dict[str, float],
                 precomputed_features: Optional[Dict] = None):
        """Initialize dataset from CSV file."""
        self.file_path = Path(file_path)
        self.preferences: List[Preference] = []
        self.participants: Set[str] = set()
        
        # Store layout maps
        self.column_map = column_map
        self.row_map = row_map
        self.finger_map = finger_map
        self.engram_position_values = engram_position_values
        self.row_position_values = row_position_values
        
        # Store precomputed features
        if precomputed_features:
            self.all_bigrams = precomputed_features['all_bigrams']
            self.all_bigram_features = precomputed_features['all_bigram_features']
            self.feature_names = precomputed_features['feature_names']
        
        # Load and process data
        print(f"Loading data from {self.file_path}")
        self._load_csv()  

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names including interactions."""
        if hasattr(self, 'feature_names'):
            return self.feature_names  # This should include interaction features from precompute
        elif self.preferences:
            return list(self.preferences[0].features1.keys())
        return []

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
    
    def _load_csv(self):
        """Load and validate preference data from CSV."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
        try:
            data = pd.read_csv(self.file_path)
            logger.info(f"\nLoaded CSV with {len(data)} rows")

            # Filter out same-letter bigrams
            data = data[
                (data['bigram1'].str[0] != data['bigram1'].str[1]) & 
                (data['bigram2'].str[0] != data['bigram2'].str[1])
            ]
            logger.info(f"Filtered to {len(data)} rows after removing same-letter bigrams")

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
                        print(f"Bigram 1: {pref.bigram1}")
                        print(f"Bigram 2: {pref.bigram2}")
                        print(f"Participant: {pref.participant_id}")
                        print(f"Preferred: {pref.preferred}")
                        print(f"Features Bigram 1: {pref.features1}")
                        print(f"Features Bigram 2: {pref.features2}")
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

    def _create_subset_dataset(self, indices: List[int]) -> 'PreferenceDataset':
        """Create a new dataset containing only the specified preferences."""
        try:
            # Add validation
            if len(indices) == 0:
                raise ValueError("Empty indices list provided")
                
            # Convert indices to numpy array if not already
            indices = np.array(indices)
                
            # Add debug logging
            logger.debug(f"Creating subset dataset:")
            logger.debug(f"Total preferences: {len(self.preferences)}")
            logger.debug(f"Number of indices: {len(indices)}")
            logger.debug(f"Max index: {indices.max()}")
            logger.debug(f"Min index: {indices.min()}")
            
            # Validate indices are within bounds
            if indices.max() >= len(self.preferences):
                raise ValueError(f"Index {indices.max()} out of range for preferences list of length {len(self.preferences)}")
            
            if indices.min() < 0:
                raise ValueError(f"Negative index {indices.min()} is invalid")

            subset = PreferenceDataset.__new__(PreferenceDataset)
            
            # Create subset preferences list with validation
            subset.preferences = [self.preferences[i] for i in indices]
            
            # Copy needed attributes
            subset.file_path = self.file_path
            subset.column_map = self.column_map
            subset.row_map = self.row_map
            subset.finger_map = self.finger_map
            subset.engram_position_values = self.engram_position_values
            subset.row_position_values = self.row_position_values
            subset.participants = {p.participant_id for p in subset.preferences}
            
            # Copy feature-related attributes if they exist
            if hasattr(self, 'all_bigrams'):
                subset.all_bigrams = self.all_bigrams
                subset.all_bigram_features = self.all_bigram_features
                subset.feature_names = self.feature_names
                
            return subset
            
        except Exception as e:
            logger.error(f"Error creating subset dataset: {str(e)}")
            logger.error(f"Preferences length: {len(self.preferences)}")
            logger.error(f"Indices length: {len(indices) if isinstance(indices, (list, np.ndarray)) else 'unknown'}")
            logger.error(f"Sample of indices: {indices[:10] if isinstance(indices, (list, np.ndarray)) else indices}")
            raise
        
    def _create_preference(self, row: pd.Series) -> Preference:
        """Create single Preference instance from data row."""
        bigram1 = (row['bigram1'][0], row['bigram1'][1])
        bigram2 = (row['bigram2'][0], row['bigram2'][1])

        try:
            # Get base features
            features1 = self.all_bigram_features[bigram1].copy()
            features2 = self.all_bigram_features[bigram2].copy()

            # Add timing information
            try:
                time1 = float(row['bigram1_time'])
                time2 = float(row['bigram2_time'])
                
                if np.isnan(time1) or np.isnan(time2):
                    logger.debug(f"NaN timing values for bigrams {bigram1}-{bigram2}")
                    time1 = None if np.isnan(time1) else time1
                    time2 = None if np.isnan(time2) else time2
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid timing values for bigrams {bigram1}-{bigram2}: {e}")
                time1 = None
                time2 = None

            # Add timing to features
            features1['typing_time'] = time1
            features2['typing_time'] = time2

            return Preference(
                bigram1=str(row['bigram1']),
                bigram2=str(row['bigram2']),
                participant_id=str(row['user_id']),
                preferred=(row['chosen_bigram'] == row['bigram1']),
                features1=features1,
                features2=features2,
                confidence=float(row['abs_sliderValue']),
                typing_time1=time1,
                typing_time2=time2
            )

        except Exception as e:
            logger.error(f"Error processing row {row.name}:")
            logger.error(f"Bigrams: {bigram1}-{bigram2}")
            logger.error(f"Error: {str(e)}")
            raise
        
