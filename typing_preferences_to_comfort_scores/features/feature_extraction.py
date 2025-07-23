# typing_preferences_to_comfort_scores/features/feature_extraction.py
"""
Feature extraction module for keyboard layout analysis and preference modeling.

Provides comprehensive extraction of bigram typing features through:
  - Physical layout characteristics (rows, columns, fingers)
  - Ergonomic metrics (angles, distances, position values)
  - Motion-based features (rolls, finger transitions)
  - Same-letter and different-letter handling
  - Interaction pattern detection
  - Efficient caching mechanisms
  - Configuration-driven extraction

Supports keyboard analysis through:
  - Centralized feature computation
  - Pre-computation for performance
  - Feature interaction loading
  - Proper error handling and logging
  - Extensible feature set
  - Metrics-based calculations
  - Bigram characteristic analysis
"""
from typing import Dict, List, Tuple
from pathlib import Path
import yaml

from typing_preferences_to_comfort_scores.features.features import (
    qwerty_bigram_frequency,
    same_finger,
    sum_finger_values,
    adj_finger_diff_row,
    rows_apart,
    angle_apart,
    outward_roll,
    middle_column,
    sum_engram_position_values,
    sum_row_position_values,
    same_key
)
from typing_preferences_to_comfort_scores.features.features import qwerty_bigram_frequency
from typing_preferences_to_comfort_scores.utils.config import FeatureConfig
from typing_preferences_to_comfort_scores.features.bigram_frequencies import bigrams, bigram_frequencies_array
from typing_preferences_to_comfort_scores.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class FeatureExtractor:
    """Centralized feature extraction functionality"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self._cache = {}  # Cache for computed features
        
    def precompute_all_features(self, layout_chars: List[str]) -> Tuple[List[Tuple[str, str]], Dict]:
        """
        Precompute features for all possible bigrams in layout.
        """
        all_bigrams = []
        bigram_features = {}

        # Get first different-letter bigram for debug logging
        first_char1 = layout_chars[0]
        first_char2 = layout_chars[1] if len(layout_chars) > 1 else layout_chars[0]
        if first_char1 != first_char2:
            first_features = self.extract_bigram_features(first_char1, first_char2)
                
        for char1 in layout_chars:
            for char2 in layout_chars:
                # Skip same-letter bigrams
                #if char1 == char2:
                #    continue
                    
                bigram = (char1, char2)
                all_bigrams.append(bigram)
                features = self.extract_bigram_features(char1, char2)
                if 'bigram_frequency' not in features:
                    logger.warning(f"bigram_frequency missing for bigram {bigram}")
                bigram_features[bigram] = features
        
        # Debug log final results
        logger.debug(f"Computed features for {len(all_bigrams)} bigrams")
        if all_bigrams:
            logger.debug(f"Feature keys in first bigram: {list(bigram_features[all_bigrams[0]].keys())}")
        
        return all_bigrams, bigram_features

    def extract_bigram_features(self, char1: str, char2: str) -> Dict[str, float]:
        """
        Extract all features for a bigram.
        
        Args:
            char1: First character of bigram
            char2: Second character of bigram
            
        Returns:
            Dictionary of feature names to values
        """
        # Check cache first
        bigram = (char1, char2)
        if bigram in self._cache:
            return self._cache[bigram].copy()
            
        # Extract features
        features = self._extract_features(char1, char2)
            
        # Cache results
        self._cache[bigram] = features.copy()
        return features

    def _extract_features(self, char1: str, char2: str) -> Dict[str, float]:
        """Extract features for different-letter bigrams"""
        try:
            features = {
                'typing_time': 0.0,  # Placeholder for actual timing data
                
                # Finger usage features
                'same_finger': same_finger(char1, char2, self.config.column_map, self.config.finger_map),
                'sum_finger_values': sum_finger_values(char1, char2, self.config.finger_map),
                'adj_finger_diff_row': adj_finger_diff_row(char1, char2, self.config.column_map, 
                                                        self.config.row_map, self.config.finger_map),
                
                # Spatial features
                'rows_apart': rows_apart(char1, char2, self.config.column_map, self.config.row_map),
                'angle_apart': angle_apart(char1, char2, self.config.column_map, self.config.angles),
                'outward_roll': outward_roll(char1, char2, self.config.column_map, self.config.finger_map),
                'middle_column': middle_column(char1, char2, self.config.column_map),
                
                # Position values
                'sum_engram_position_values': sum_engram_position_values(char1, char2, 
                                                                    self.config.column_map, 
                                                                    self.config.engram_position_values),
                'sum_row_position_values': sum_row_position_values(char1, char2, 
                                                                self.config.column_map, 
                                                                self.config.row_position_values),
                
                # Control feature
                'bigram_frequency': qwerty_bigram_frequency(char1, char2, 
                                                        self.config.bigrams,
                                                        self.config.bigram_frequencies_array)
            }
            return features
        except Exception as e:
            logger.error(f"Error computing features for bigram {char1}{char2}: {str(e)}")
            raise
        
    def _calc_same_finger(self, char1: str, char2: str) -> float:
        """Calculate if bigram uses same finger"""
        return same_finger(char1, char2, self.config.column_map, self.config.finger_map)

    def _calc_outward_roll(self, char1: str, char2: str) -> float:
        """Calculate if bigram involves outward rolling motion"""
        return outward_roll(char1, char2, self.config.column_map, self.config.finger_map)

    def _calc_rows_apart(self, char1: str, char2: str) -> float:
        """Calculate number of rows between keys"""
        return rows_apart(char1, char2, self.config.column_map, self.config.row_map)

    def _calc_angle_apart(self, char1: str, char2: str) -> float:
        """Calculate angular distance between keys"""
        return angle_apart(char1, char2, self.config.column_map, self.config.angles)

    def _calc_middle_column(self, char1: str, char2: str) -> float:
        """Calculate middle column usage"""
        return middle_column(char1, char2, self.config.column_map)

    def _calc_adj_finger_diff_row(self, char1: str, char2: str) -> float:
        """Calculate adjacent finger difference within row"""
        return adj_finger_diff_row(char1, char2, self.config.column_map, 
                                self.config.row_map, self.config.finger_map)

    def _calc_sum_finger_values(self, char1: str, char2: str) -> float:
        """Calculate sum of finger values"""
        return sum_finger_values(char1, char2, self.config.finger_map)

    def _calc_sum_engram_position_values(self, char1: str, char2: str) -> float:
        """Calculate sum of engram position values"""
        return sum_engram_position_values(char1, char2, self.config.column_map, 
                                        self.config.engram_position_values)

    def _calc_sum_row_position_values(self, char1: str, char2: str) -> float:
        """Calculate sum of row position values"""
        return sum_row_position_values(char1, char2, self.config.column_map, 
                                    self.config.row_position_values)
    