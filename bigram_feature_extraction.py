"""
Feature Extraction Module

This module handles the extraction and computation of features for keyboard layout analysis.
It provides functions for computing ergonomic metrics for single-key and multi-key combinations.

The module computes three types of features:
1. Regular bigram features (for different key combinations)
2. Same-key features (when a key is pressed twice)
3. Feature differences between bigram pairs (for comparison analysis)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging

from bigram_feature_definitions import (
    column_map, row_map, finger_map, engram_position_values,
    row_position_values, bigrams, bigram_frequencies_array,
    function_map
)

logger = logging.getLogger(__name__)

def extract_bigram_features(
    char1: str, 
    char2: str, 
    column_map: Dict[str, int], 
    row_map: Dict[str, int], 
    finger_map: Dict[str, int],
    engram_position_values: Dict[str, float],
    row_position_values: Dict[str, float]
) -> Dict[str, float]:
    """
    Extract features for a bigram (two-character combination).

    Args:
        char1: First character of the bigram
        char2: Second character of the bigram
        column_map: Mapping of characters to keyboard columns
        row_map: Mapping of characters to keyboard rows
        finger_map: Mapping of characters to fingers used
        engram_position_values: Comfort scores for key positions
        row_position_values: Comfort scores for row positions

    Returns:
        Dictionary mapping feature names to their computed values
    """
    features = {}
    
    # Same finger usage
    features['same_finger'] = 1.0 if finger_map[char1] == finger_map[char2] else 0.0
    
    # Finger values
    features['sum_finger_values'] = finger_map[char1] + finger_map[char2]
    
    # Adjacent finger difference within row
    features['adj_finger_diff_row'] = (
        1.0 if abs(finger_map[char1] - finger_map[char2]) == 1 and 
        row_map[char1] == row_map[char2] else 0.0
    )
    
    # Row positions
    features['rows_apart'] = abs(row_map[char1] - row_map[char2])
    
    # Angular distance between keys
    col_diff = column_map[char2] - column_map[char1]
    row_diff = row_map[char2] - row_map[char1]
    features['angle_apart'] = np.sqrt(col_diff**2 + row_diff**2)
    
    # Outward rolling motion
    features['outward_roll'] = (
        1.0 if (finger_map[char2] > finger_map[char1] and column_map[char2] > column_map[char1]) or
               (finger_map[char2] < finger_map[char1] and column_map[char2] < column_map[char1])
        else 0.0
    )
    
    # Middle column usage
    features['middle_column'] = (
        1.0 if column_map[char1] == 3 or column_map[char2] == 3 else 0.0
    )
    
    # Position values
    features['sum_engram_position_values'] = (
        engram_position_values[char1] + engram_position_values[char2]
    )
    features['sum_row_position_values'] = (
        row_position_values[char1] + row_position_values[char2]
    )
    
    return features

def compute_samekey_features(char: str, finger_map: Dict[str, int]) -> Dict[str, float]:
    """
    Compute features for same-key bigrams (double-taps).
    
    Args:
        char: The character being analyzed
        finger_map: Mapping of characters to fingers

    Returns:
        Dictionary of feature values for the same-key press
    """
    features = {}
    
    # Basic finger position
    finger_pos = finger_map[char]
    features['finger_position'] = float(finger_pos)
    
    # Finger grouping (index/middle vs ring/pinky)
    features['outer_finger'] = 1.0 if finger_pos in [1, 4] else 0.0
    
    return features

def precompute_all_bigram_features(
    layout_chars: List[str],
    column_map: Dict[str, int], 
    row_map: Dict[str, int], 
    finger_map: Dict[str, int],
    engram_position_values: Dict[str, float],
    row_position_values: Dict[str, float],
    bigrams: List[Tuple[str, str]],
    bigram_frequencies_array: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[List[Tuple[str, str]], Dict, List[str], List[Tuple[str, str]], Dict, List[str]]:
    """
    Precompute features for all possible bigrams in the layout.
    
    Args:
        layout_chars: List of characters in the keyboard layout
        column_map: Mapping of characters to keyboard columns
        row_map: Mapping of characters to keyboard rows
        finger_map: Mapping of characters to fingers
        engram_position_values: Comfort scores for key positions
        row_position_values: Comfort scores for row positions
        bigrams: List of all possible bigram combinations
        bigram_frequencies_array: Array of bigram frequency values
        config: Configuration dictionary

    Returns:
        Tuple containing:
        - List of all possible bigrams
        - Dictionary of features for regular bigrams
        - List of feature names
        - List of same-key bigrams
        - Dictionary of features for same-key bigrams
        - List of same-key feature names
    """
    logger.info("Computing base features")
    
    # Compute base features
    all_bigrams = []
    all_bigram_features = {}
    
    # Process all possible bigram combinations
    for char1 in layout_chars:
        for char2 in layout_chars:
            bigram = (char1, char2)
            all_bigrams.append(bigram)
            
            # Extract basic features
            features = extract_bigram_features(
                char1, char2, column_map, row_map, finger_map,
                engram_position_values, row_position_values
            )
            
            all_bigram_features[bigram] = features
    
    # Get feature names from first computed features
    feature_names = list(next(iter(all_bigram_features.values())).keys())
    
    # Compute interactions if enabled and model training is active
    if (config['model']['train'] and 
        config['model'].get('features', {}).get('interactions', {}).get('enabled', False)):
        logger.info("Computing feature interactions")
        try:
            interactions = config['model']['features']['interactions'].get('interactions', [])
            
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
                        
        except Exception as e:
            logger.warning(f"Error computing interactions: {str(e)}")
    
    # Compute same-key features
    logger.info("Computing same-key features")
    samekey_bigrams = []
    samekey_bigram_features = {}
    
    for char in layout_chars:
        bigram = (char, char)
        samekey_bigrams.append(bigram)
        samekey_bigram_features[bigram] = compute_samekey_features(char, finger_map)
    
    samekey_feature_names = list(next(iter(samekey_bigram_features.values())).keys())
    
    return (all_bigrams, all_bigram_features, feature_names,
            samekey_bigrams, samekey_bigram_features, samekey_feature_names)

def precompute_bigram_feature_differences(
    bigram_features: Dict[Tuple[str, str], Dict[str, float]]
) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], np.ndarray]:
    """
    Precompute feature differences between all possible bigram pairs.
    
    Args:
        bigram_features: Dictionary mapping bigram tuples to their feature dictionaries
            Example: {('a', 'b'): {'same_finger': 1.0, 'rows_apart': 2.0}}

    Returns:
        Dictionary mapping bigram pairs to their feature differences
    """
    logger.info("Computing bigram feature differences")
    bigram_feature_differences = {}
    bigrams_list = list(bigram_features.keys())
    
    # Get consistent feature ordering from first bigram
    first_bigram = bigrams_list[0]
    feature_names = list(bigram_features[first_bigram].keys())
    
    # Pre-log what we're computing
    logger.info(f"Computing differences for {len(feature_names)} features across {len(bigrams_list)} bigrams")
    
    # Compute differences for all pairs
    for i, bigram1 in enumerate(bigrams_list):
        for j, bigram2 in enumerate(bigrams_list):
            if i <= j:  # Only compute unique pairs
                # Convert feature dictionaries to arrays in consistent order
                features1 = np.array([bigram_features[bigram1][feat] for feat in feature_names])
                features2 = np.array([bigram_features[bigram2][feat] for feat in feature_names])
                
                abs_feature_diff = np.abs(features1 - features2)
                
                # Store both directions
                bigram_feature_differences[(bigram1, bigram2)] = abs_feature_diff
                bigram_feature_differences[(bigram2, bigram1)] = abs_feature_diff

    logger.info(f"Completed feature difference computation")
    return bigram_feature_differences
