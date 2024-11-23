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
from itertools import product
from typing import List, Tuple, Dict

from bigram_feature_definitions import (
    column_map, row_map, finger_map, engram_position_values,
    row_position_values, bigrams, bigram_frequencies_array,
    function_map
)

def extract_bigram_features(
    char1: str, 
    char2: str, 
    column_map: Dict[str, int], 
    row_map: Dict[str, int], 
    finger_map: Dict[str, int],
    engram_position_values: Dict[str, float],
    row_position_values: Dict[str, float],
    bigrams: List[Tuple[str, str]],
    bigram_frequencies_array: np.ndarray,
    config: Dict
) -> Tuple[Dict[str, float], List[str]]:
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
        bigrams: List of all possible bigram combinations
        bigram_frequencies_array: Array of bigram frequency values
        config: Configuration dictionary containing feature settings

    Returns:
        Tuple containing:
        - Dictionary mapping feature names to their computed values
        - List of computed feature names
    """
    features_functions = function_map(
        char1, char2, column_map, row_map, finger_map,
        engram_position_values, row_position_values,
        bigrams, bigram_frequencies_array
    )

    features = {}
    all_features = config['features']['all_features']
    
    for feature_name in all_features:
        if feature_name in features_functions:
            features[feature_name] = features_functions[feature_name]()
    
    feature_names = list(features.keys())
    
    # Add interaction features if enabled
    if config['features']['interactions']['enabled']:
        for pair in config['features']['interactions']['pairs']:
            interaction_name = f"{pair[0]}_{pair[1]}"
            features[interaction_name] = features[pair[0]] * features[pair[1]]
            feature_names.append(interaction_name)
    
    return features, feature_names

def extract_samekey_features(
    char: str, 
    finger_map: Dict[str, int]
) -> Tuple[Dict[str, int], List[str]]:
    """
    Extract features for same-key bigrams (double-taps).
    
    Args:
        char: The character being analyzed
        finger_map: Mapping of characters to fingers (1-4)

    Returns:
        Tuple containing:
        - Dictionary mapping finger positions to binary values (1 if used, 0 if not)
        - List of feature names ('finger1' through 'finger4')
    """
    features = {
        'finger1': int(finger_map[char] == 1),
        'finger2': int(finger_map[char] == 2),
        'finger3': int(finger_map[char] == 3),
        'finger4': int(finger_map[char] == 4)
    }
    feature_names = list(features.keys())
    
    return features, feature_names

def precompute_all_bigram_features(
    layout_chars: List[str], 
    column_map: Dict[str, int], 
    row_map: Dict[str, int], 
    finger_map: Dict[str, int], 
    engram_position_values: Dict[str, float],
    row_position_values: Dict[str, float],
    bigrams: List[Tuple[str, str]],
    bigram_frequencies_array: np.ndarray,
    config: Dict
) -> Tuple[List[Tuple[str, str]], pd.DataFrame, List[str],
           List[Tuple[str, str]], pd.DataFrame, List[str]]:
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
        - all_bigrams: List of all possible two-character combinations
        - features_df: DataFrame of features for regular bigrams
        - feature_names: List of regular bigram feature names
        - samekey_bigrams: List of same-key bigrams
        - samekey_features_df: DataFrame of features for same-key bigrams
        - samekey_feature_names: List of same-key feature names
    """
    # Generate all possible 2-key bigrams
    all_bigrams = [(x, y) for x, y in product(layout_chars, repeat=2) if x != y]
    # Generate same-key bigrams
    samekey_bigrams = [(char, char) for char in layout_chars]

    # Extract features for each type
    feature_vectors = []
    feature_names = None
    samekey_feature_vectors = []
    samekey_feature_names = None

    # Process regular bigrams
    for char1, char2 in all_bigrams:
        features, names = extract_bigram_features(
            char1, char2, column_map, row_map, 
            finger_map, engram_position_values, 
            row_position_values, bigrams, 
            bigram_frequencies_array,
            config
        )
        feature_vectors.append(list(features.values()))
        if feature_names is None:
            feature_names = names

    # Process same-key bigrams
    for char1, char2 in samekey_bigrams:
        samekey_features, names = extract_samekey_features(char1, finger_map)
        samekey_feature_vectors.append(list(samekey_features.values()))
        if samekey_feature_names is None:
            samekey_feature_names = names

    # Convert to DataFrames with MultiIndex
    features_df = pd.DataFrame(feature_vectors, columns=feature_names, index=all_bigrams)
    features_df.index = pd.MultiIndex.from_tuples(features_df.index)
    
    samekey_features_df = pd.DataFrame(
        samekey_feature_vectors, 
        columns=samekey_feature_names, 
        index=samekey_bigrams
    )
    samekey_features_df.index = pd.MultiIndex.from_tuples(samekey_features_df.index)

    return (all_bigrams, features_df, feature_names, 
            samekey_bigrams, samekey_features_df, samekey_feature_names)

def precompute_bigram_feature_differences(
    bigram_features: pd.DataFrame
) -> Dict[Tuple[Tuple[str, str], Tuple[str, str]], np.ndarray]:
    """
    Precompute feature differences between all possible bigram pairs.
    
    Args:
        bigram_features: DataFrame of precomputed features for each bigram

    Returns:
        Dictionary mapping bigram pairs to their feature differences.
        Keys are tuples of bigram tuples: ((char1, char2), (char3, char4))
        Values are numpy arrays of absolute feature differences.
    """
    bigram_feature_differences = {}
    bigrams_list = list(bigram_features.index)

    # Compute differences for all pairs
    for i, bigram1 in enumerate(bigrams_list):
        for j, bigram2 in enumerate(bigrams_list):
            if i <= j:  # Only compute unique pairs
                abs_feature_diff = np.abs(bigram_features.loc[bigram1].values - 
                                        bigram_features.loc[bigram2].values)
                # Store both directions
                bigram_feature_differences[(bigram1, bigram2)] = abs_feature_diff
                bigram_feature_differences[(bigram2, bigram1)] = abs_feature_diff

    return bigram_feature_differences