# engram3/features/feature_extraction.py
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
from dataclasses import dataclass
from pathlib import Path
import yaml

from engram3.utils.config import FeatureConfig
from engram3.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

class FeatureExtractor:
    """Centralized feature extraction functionality"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self._cache = {}  # Cache for computed features
        
    def precompute_all_features(self, layout_chars: List[str]) -> Tuple[List[Tuple[str, str]], Dict]:
        """
        Precompute features for all possible bigrams in layout.
        
        Args:
            layout_chars: List of characters in keyboard layout
            
        Returns:
            Tuple of (all_bigrams, bigram_features)
        """
        all_bigrams = []
        bigram_features = {}
        
        for char1 in layout_chars:
            for char2 in layout_chars:
                bigram = (char1, char2)
                all_bigrams.append(bigram)
                bigram_features[bigram] = self.extract_bigram_features(char1, char2)
                
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
            
        # Handle same-letter bigrams differently
        if char1 == char2:
            features = self._extract_same_letter_features(char1)
        else:
            features = self._extract_different_letter_features(char1, char2)
            
        # Cache results
        self._cache[bigram] = features.copy()
        return features

    def _extract_different_letter_features(self, char1: str, char2: str) -> Dict[str, float]:
        """Extract features for different-letter bigrams"""
        features = {
            'typing_time': 0.0,  # Placeholder for actual timing data
            
            # Finger usage features
            'same_finger': self._calc_same_finger(char1, char2),
            'sum_finger_values': self._calc_sum_finger_values(char1, char2),
            'adj_finger_diff_row': self._calc_adj_finger_diff_row(char1, char2),
            
            # Spatial features
            'rows_apart': self._calc_rows_apart(char1, char2),
            'angle_apart': self._calc_angle_apart(char1, char2),
            'outward_roll': self._calc_outward_roll(char1, char2),
            'middle_column': self._calc_middle_column(char1, char2),
            
            # Position values
            'sum_engram_position_values': self._calc_sum_engram_position_values(char1, char2),
            'sum_row_position_values': self._calc_sum_row_position_values(char1, char2)
        }
        return features

    def _extract_same_letter_features(self, char: str) -> Dict[str, float]:
        """Extract features for same-letter bigrams"""
        return {
            'typing_time': 0.0,
            'same_finger': 1.0,
            'sum_finger_values': self.config.finger_map.get(char, 0) * 2,
            'adj_finger_diff_row': 0.0,
            'rows_apart': 0,
            'angle_apart': 0.0,
            'outward_roll': 0.0,
            'middle_column': 1.0 if self._is_middle_column(char) else 0.0,
            'sum_engram_position_values': self.config.engram_position_values.get(char, 0) * 2,
            'sum_row_position_values': self.config.row_position_values.get(char, 0) * 2
        }

    def _calc_same_finger(self, char1: str, char2: str) -> float:
        """Calculate if bigram uses same finger"""
        return float(self.config.finger_map.get(char1) == self.config.finger_map.get(char2))

    def _calc_sum_finger_values(self, char1: str, char2: str) -> float:
        """Calculate sum of finger values"""
        return (self.config.finger_map.get(char1, 0) + 
                self.config.finger_map.get(char2, 0))

    def _calc_adj_finger_diff_row(self, char1: str, char2: str) -> float:
        """Calculate adjacent finger difference within row"""
        f1 = self.config.finger_map.get(char1)
        f2 = self.config.finger_map.get(char2)
        r1 = self.config.row_map.get(char1)
        r2 = self.config.row_map.get(char2)
        return float(abs(f1 - f2) == 1 and r1 == r2)

    def _calc_rows_apart(self, char1: str, char2: str) -> float:
        """Calculate number of rows between keys"""
        r1 = self.config.row_map.get(char1, 0)
        r2 = self.config.row_map.get(char2, 0)
        return float(abs(r1 - r2))

    def _calc_angle_apart(self, char1: str, char2: str) -> float:
        """Calculate angular distance between keys"""
        c1 = self.config.column_map.get(char1)
        c2 = self.config.column_map.get(char2)
        if c1 is None or c2 is None:
            return 0.0
        angles = self.config.angles
        return angles.get((char1, char2), 0.0)  # Use tuple key directly

    def _calc_outward_roll(self, char1: str, char2: str) -> float:
        """Calculate if bigram involves outward rolling motion"""
        c1 = self.config.column_map.get(char1)
        c2 = self.config.column_map.get(char2)
        f1 = self.config.finger_map.get(char1)
        f2 = self.config.finger_map.get(char2)
        if None in (c1, c2, f1, f2):
            return 0.0
        return float(c2 > c1 and f2 > f1)

    def _calc_middle_column(self, char1: str, char2: str) -> float:
        """Calculate middle column usage"""
        return float(self._is_middle_column(char1) or self._is_middle_column(char2))

    def _is_middle_column(self, char: str) -> bool:
        """Check if character is in middle column"""
        col = self.config.column_map.get(char, 0)
        return col in (5, 6)  # Assuming middle columns are 5 and 6

    def _calc_sum_engram_position_values(self, char1: str, char2: str) -> float:
        """Calculate sum of engram position values"""
        return (self.config.engram_position_values.get(char1, 0) + 
                self.config.engram_position_values.get(char2, 0))

    def _calc_sum_row_position_values(self, char1: str, char2: str) -> float:
        """Calculate sum of row position values"""
        return (self.config.row_position_values.get(char1, 0) + 
                self.config.row_position_values.get(char2, 0))



def load_interactions(filepath: str) -> List[List[str]]:
    """Load feature interactions from YAML file"""
    logger.debug(f"Loading interactions from {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            
        if not data or 'interactions' not in data:
            logger.warning(f"No interactions found in {filepath}")
            return []
            
        interactions = data['interactions']
        
        if not isinstance(interactions, list):
            logger.error("Interactions must be a list")
            return []
            
        valid_interactions = []
        for interaction in interactions:
            if isinstance(interaction, list) and all(isinstance(f, str) for f in interaction):
                valid_interactions.append(interaction)
            else:
                logger.warning(f"Skipping invalid interaction format: {interaction}")
                
        logger.info(f"Loaded {len(valid_interactions)} valid interactions")
        return valid_interactions
        
    except Exception as e:
        logger.error(f"Error loading interactions file: {str(e)}")
        return []