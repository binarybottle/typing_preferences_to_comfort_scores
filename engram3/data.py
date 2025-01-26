# data.py
"""
Data loading and preprocessing functionality for typing preference data.
Implements PreferenceDataset class which handles:
  - Loading and validating raw preference data
  - Computing and storing bigram features
  - Managing train/test splits
  - Providing consistent data access for modeling
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import pandas as pd
import numpy as np

from engram3.utils.config import Preference
from engram3.features.keymaps import *
from engram3.features.feature_extraction import FeatureExtractor, FeatureConfig
from engram3.features.bigram_frequencies import bigrams, bigram_frequencies_array
from engram3.utils.logging import LoggingManager
logger = logging.getLogger(__name__)
    
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
                feature_extractor: Optional['FeatureExtractor'] = None,
                config: Optional[Dict] = None,
                precomputed_features: Optional[Dict] = None):
        """
        Initialize dataset from CSV file.
        
        Args:
            file_path: Path to dataset CSV file
            feature_extractor: FeatureExtractor instance for computing features
            config: Optional configuration dictionary
            precomputed_features: Optional dict of precomputed features
        """
        self.file_path = Path(file_path)
        self.preferences: List[Preference] = []
        self.participants: Set[str] = set()
        
        # Store the config
        self.config = config
        # Initialize logging
        if config:
            self.logging_manager = LoggingManager(config)
            logger.info(f"Loading data from {self.file_path}")
            
            # Store control features configuration
            self.control_features = config.features.control_features
        else:
            self.control_features = []

        # Handle feature extraction setup
        if feature_extractor:

            # Use provided feature extractor
            self.feature_extractor = feature_extractor
            
            # Get maps from feature_extractor's config
            if feature_extractor.config:
                self.column_map = feature_extractor.config.column_map
                self.row_map = feature_extractor.config.row_map
                self.finger_map = feature_extractor.config.finger_map
                self.engram_position_values = feature_extractor.config.engram_position_values
                self.row_position_values = feature_extractor.config.row_position_values
                self.angles = feature_extractor.config.angles
                self.bigrams = feature_extractor.config.bigrams
                self.bigram_frequencies_array = feature_extractor.config.bigram_frequencies_array
        
        # Handle precomputed features
        if precomputed_features:
            self.all_bigrams = precomputed_features['all_bigrams']
            self.all_bigram_features = precomputed_features['all_bigram_features']
            
            # Initialize features including base, interaction, and control features
            if self.config:
                base_features = self.config.features.base_features
                interaction_features = self.config.features.get_all_interaction_names()
                self.feature_names = base_features + interaction_features + list(self.control_features)
            else:
                self.feature_names = precomputed_features['feature_names']
                
            # Debug before validation
            logger.debug(f"Precomputed feature names: {self.feature_names}")
            logger.debug(f"Control features to check: {self.config.features.control_features}")
                
            # Validate that control features are present in precomputed features
            missing_controls = [f for f in self.control_features 
                              if f not in self.feature_names]
            if missing_controls:
                raise ValueError(f"Control features missing from precomputed features: {missing_controls}")
            
            # Don't create new feature extractor if one was already provided
            if not feature_extractor and config:
                self.feature_config = FeatureConfig(
                    column_map=self.column_map,
                    row_map=self.row_map,
                    finger_map=self.finger_map,
                    engram_position_values=self.engram_position_values,
                    row_position_values=self.row_position_values,
                    angles=self.angles,
                    bigrams=self.bigrams,
                    bigram_frequencies_array=self.bigram_frequencies_array
                )
                self.feature_extractor = FeatureExtractor(self.feature_config)
        
        # Load and process data
        self._load_csv()

        # Filter out preferences with NaN typing times
        if hasattr(self, 'feature_names') and 'typing_time' in self.feature_names:
            original_count = len(self.preferences)
            valid_prefs = []
            for pref in self.preferences:
                if (pref.features1.get('typing_time') is not None and 
                    pref.features2.get('typing_time') is not None):
                    valid_prefs.append(pref)
            
            n_filtered = original_count - len(valid_prefs)
            logger.info(f"Filtered out {n_filtered} preferences with NaN typing times")
            self.preferences = valid_prefs
            self.participants = {p.participant_id for p in self.preferences}
            logger.info(f"Remaining preferences: {len(self.preferences)}")
            logger.info(f"Remaining participants: {len(self.participants)}")

    def get_feature_names(self, include_control: bool = True) -> List[str]:
        """Get list of all feature names, optionally including control features."""
        if hasattr(self, 'feature_names'):
            if include_control:
                return self.feature_names
            return [f for f in self.feature_names if f not in self.control_features]
        elif hasattr(self, 'config'):
            # Get base features and interactions from config
            base_features = self.config.features.base_features
            interaction_features = self.config.features.get_all_interaction_names()
            all_features = base_features + interaction_features
            if include_control:
                return all_features + list(self.config.features.control_features)
            return all_features
        elif self.preferences:
            # Fallback to preferences features only if no config
            features = list(self.preferences[0].features1.keys())
            if include_control:
                return features
            return [f for f in features if f not in self.control_features]
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
                
        # Create new datasets using __new__ to skip __init__
        train_data = PreferenceDataset.__new__(PreferenceDataset)
        test_data = PreferenceDataset.__new__(PreferenceDataset)
        
        # Set attributes for both datasets
        for data, prefs in [(train_data, train_prefs), (test_data, test_prefs)]:
            # Core attributes
            data.preferences = prefs
            data.participants = {p.participant_id for p in prefs}
            data.file_path = self.file_path
            data.config = self.config
            data.control_features = self.control_features
            
            # Feature extraction related
            data.feature_extractor = self.feature_extractor
            
            # Copy feature-related attributes
            if hasattr(self, 'feature_names'):
                data.feature_names = self.feature_names
            if hasattr(self, 'all_bigrams'):
                data.all_bigrams = self.all_bigrams
            if hasattr(self, 'all_bigram_features'):
                data.all_bigram_features = self.all_bigram_features
                
            # Copy maps and values
            for attr in ['column_map', 'row_map', 'finger_map', 
                        'engram_position_values', 'row_position_values']:
                if hasattr(self, attr):
                    setattr(data, attr, getattr(self, attr))
        
        logger.info(f"Split dataset:")
        logger.info(f"  Train: {len(train_prefs)} preferences, {len(train_data.participants)} participants")
        logger.info(f"  Test: {len(test_prefs)} preferences, {len(test_data.participants)} participants")
        
        return train_data, test_data
        
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

# In data.py

    def _create_subset_dataset(self, indices: Union[List[int], np.ndarray]) -> 'PreferenceDataset':
        """Create a new dataset containing only the specified preferences."""
        try:
            # Convert indices to numpy array if not already
            indices = np.array(indices)
            
            # Validate indices
            if len(indices) == 0:
                raise ValueError("Empty indices list provided")
                
            if len(set(indices)) != len(indices):
                logger.warning(f"Duplicate indices detected, will be deduplicated")
                indices = np.unique(indices)
                
            # Check for out of bounds indices
            if np.any(indices < 0) or np.any(indices >= len(self.preferences)):
                invalid_indices = indices[(indices < 0) | (indices >= len(self.preferences))]
                raise ValueError(f"Invalid indices detected: {invalid_indices[:5]}...")
            
            subset = PreferenceDataset.__new__(PreferenceDataset)
            
            # Create subset preferences list
            subset.preferences = [self.preferences[i] for i in indices]
            
            # Verify subset creation
            if len(subset.preferences) != len(indices):
                raise ValueError(f"Subset creation failed: size mismatch. Expected {len(indices)}, got {len(subset.preferences)}")
            
            # Copy needed attributes
            subset.file_path = self.file_path
            subset.config = self.config
            if subset.config:
                subset.control_features = subset.config.features.control_features
            else:
                subset.control_features = []
                
            # Copy maps and values
            for attr in ['column_map', 'row_map', 'finger_map', 
                        'engram_position_values', 'row_position_values']:
                if hasattr(self, attr):
                    setattr(subset, attr, getattr(self, attr))
                else:
                    logger.warning(f"Source dataset missing attribute: {attr}")
            
            # Set participants based on preferences
            subset.participants = {p.participant_id for p in subset.preferences}
            
            # Copy feature extraction related attributes
            subset.feature_extractor = self.feature_extractor
            
            # Copy feature-related attributes if they exist
            if hasattr(self, 'all_bigrams'):
                for attr in ['all_bigrams', 'all_bigram_features', 'feature_names']:
                    setattr(subset, attr, getattr(self, attr))
            
            # Validate final subset
            if not subset.preferences:
                raise ValueError("Created subset has no preferences")
            if not subset.participants:
                raise ValueError("Created subset has no participants")
                
            return subset
                
        except Exception as e:
            logger.error(f"Error creating subset dataset: {str(e)}")
            logger.error(f"Preferences length: {len(self.preferences)}")
            logger.error(f"Indices length: {len(indices)}")
            logger.error(f"Sample of indices: {indices[:10]}")
            raise
                            
    def _create_preference(self, row: pd.Series) -> Preference:
        """Create single Preference instance from data row."""
        bigram1 = (row['bigram1'][0], row['bigram1'][1])
        bigram2 = (row['bigram2'][0], row['bigram2'][1])

        try:
            # Get base features
            features1 = self.all_bigram_features[bigram1].copy()
            features2 = self.all_bigram_features[bigram2].copy()

            # Add interaction features if config exists
            if self.config and hasattr(self.config.features, 'interactions'):
                for interaction in self.config.features.interactions:
                    interaction_name = '_x_'.join(sorted(interaction))
                    feat1_interaction = 1.0
                    feat2_interaction = 1.0
                    for component in interaction:
                        feat1_interaction *= features1.get(component, 0.0)
                        feat2_interaction *= features2.get(component, 0.0)
                    features1[interaction_name] = feat1_interaction
                    features2[interaction_name] = feat2_interaction

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
        
