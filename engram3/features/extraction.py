"""
Feature extraction

These functions handle the extraction of all bigram features for keyboard layout analysis.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging

from engram3.utils import load_interactions
from engram3.features.keymaps import *
from engram3.features.features import *

logger = logging.getLogger(__name__)

def extract_bigram_features(char1: str, 
                            char2: str, 
                            column_map: Dict[str, int], 
                            row_map: Dict[str, int], 
                            finger_map: Dict[str, int], 
                            engram_position_values: Dict[str, float], 
                            row_position_values: Dict[str, float]) -> Dict[str, float]:

    """
    Extract features for a bigram (two-character combination).

    Args:
        char1: First character of the bigram
        char2: Second character of the bigram

    Returns:
        Dictionary mapping feature names to their computed values
    """
    features = {}
    
    # Same finger usage
    features['same_finger'] = same_finger(char1, char2, column_map, finger_map)
    
    # Finger values
    features['sum_finger_values'] = sum_finger_values(char1, char2, finger_map)
    
    # Adjacent finger difference within row
    features['adj_finger_diff_row'] = adj_finger_diff_row(char1, char2, column_map, row_map, finger_map)
    
    # Row positions
    features['rows_apart'] = rows_apart(char1, char2, column_map, row_map)
    
    # Angular distance between keys
    features['angle_apart'] = angle_apart(char1, char2, column_map, key_metrics)
    
    # Outward rolling motion
    features['outward_roll'] = outward_roll(char1, char2, column_map, finger_map)
    
    # Middle column usage
    features['middle_column'] = middle_column(char1, char2, column_map)
    
    # Position values
    features['sum_engram_position_values'] = sum_engram_position_values(char1, char2, column_map, engram_position_values)
    features['sum_row_position_values'] = sum_row_position_values(char1, char2, column_map, row_position_values)
    
    return features

def extract_same_letter_features(letter: str, 
        column_map, finger_map, engram_position_values, row_position_values) -> Dict[str, float]:
    """Get features for same-letter bigrams."""
    return {
        'same_finger': 1.0,  # Always same finger
        'sum_finger_values': finger_map.get(letter, 0) * 2,  # Double the finger value
        'adj_finger_diff_row': 0.0,  # No finger movement
        'rows_apart': 0,  # Same key
        'angle_apart': 0.0,  # No angle
        'outward_roll': 0.0,  # No roll
        'middle_column': 1.0 if column_map.get(letter, 0) == 5 or column_map.get(letter, 0) == 6 else 0.0,  # Middle column check
        'sum_engram_position_values': engram_position_values.get(letter, 0) * 2,  # Double position value
        'sum_row_position_values': row_position_values.get(letter, 0) * 2,  # Double row position value
    }
    
def precompute_all_bigram_features(
    layout_chars: List[str],
    column_map: Dict[str, int], 
    row_map: Dict[str, int], 
    finger_map: Dict[str, int], 
    engram_position_values: Dict[str, float], 
    row_position_values: Dict[str, float],
    config: Dict[str, Any]
) -> Tuple[List[Tuple[str, str]], Dict, List[str], List[Tuple[str, str]], Dict, List[str]]:
    """
    Precompute features for all possible 2-key bigrams in the layout.
    
    Args:
        layout_chars: List of characters in the keyboard layout
        config: Configuration dictionary

    Returns:
        Tuple containing:
        - List of all possible 2-key bigrams
        - Dictionary of bigram features
        - List of feature names
    """
    logger.info("Computing base features...")
    
    # Compute base features
    all_bigrams = []
    all_bigram_features = {}
    
    # Process all possible 2-key bigram combinations
    for char1 in layout_chars:
        for char2 in layout_chars:
            bigram = (char1, char2)
            all_bigrams.append(bigram)
            
            # Extract features differently for same-letter vs different-letter bigrams
            if char1 == char2:
                features = extract_same_letter_features(char1, column_map, finger_map, 
                                engram_position_values, row_position_values)  # Special features for same-letter
            else:
                features = extract_bigram_features(char1, char2, 
                                                   column_map, row_map, finger_map, 
                                                   engram_position_values, 
                                                   row_position_values)
            
            all_bigram_features[bigram] = features
    
    # Get feature names from first computed features
    feature_names = list(next(iter(all_bigram_features.values())).keys())
    
    # Compute interactions
    try:
        logger.info("Computing feature interactions...")
        
        # Get and load interactions
        interaction_file = config['features'].get('interactions_file')
        if interaction_file:
            interactions = load_interactions(interaction_file)
            logger.info(f"Loaded {len(interactions)} interactions from {interaction_file}")
        else:
            logger.info("No interactions file specified")
            interactions = []
        
        if interactions:
            for bigram in all_bigrams:
                base_features = all_bigram_features[bigram]
                
                for interaction in interactions:
                    # Skip if not all features available
                    if not all(f in base_features for f in interaction):
                        continue
                    
                    # Compute interaction
                    interaction_name = "_".join(interaction)
                    interaction_value = np.prod([base_features[f] for f in interaction])
                    all_bigram_features[bigram][interaction_name] = interaction_value
                    
                    # Add interaction name to feature list if new
                    if interaction_name not in feature_names:
                        feature_names.append(interaction_name)
        else:
            logger.info("No interactions loaded from file")
                    
    except Exception as e:
        logger.error(f"Error computing interactions: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        
    return all_bigrams, all_bigram_features, feature_names
