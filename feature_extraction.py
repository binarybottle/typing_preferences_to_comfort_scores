"""
Feature Extraction Module

This module handles the extraction and computation of features for keyboard layout analysis.
It provides functions for computing various metrics related to keyboard layout ergonomics.
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import List, Tuple, Dict, Any
from scipy.spatial.distance import cdist

from bigram_features import *

FEATURE_NAMES = ['freq', 'engram_sum', 'row_sum']

def extract_features(char1: str, char2: str, column_map: Dict, row_map: Dict, 
                     finger_map: Dict, engram_position_values: Dict,
                     row_position_values: Dict, bigrams: List, 
                     bigram_frequencies_array: np.ndarray) -> Tuple[Dict, List[str]]:
    """
    Extract features for a bigram pair.
    
    Args:
        char1: First character of the bigram
        char2: Second character of the bigram
        column_map: Mapping of characters to keyboard columns
        row_map: Mapping of characters to keyboard rows
        finger_map: Mapping of characters to fingers
        engram_position_values: Dictionary of engram position values
        row_position_values: Dictionary of row position values
        bigrams: List of bigrams ordered by frequency
        bigram_frequencies_array: Array of corresponding frequency values

    Returns:
        Tuple containing:
        - Dictionary of computed features
        - List of feature names
    """
    features = {
        'freq': qwerty_bigram_frequency(char1, char2, bigrams, bigram_frequencies_array),
        'engram_sum': sum_engram_position_values(char1, char2, column_map, engram_position_values),
        'row_sum': sum_row_position_values(char1, char2, column_map, row_position_values)
    }
    return features, FEATURE_NAMES

def extract_features_samekey(char: str, finger_map: Dict) -> Tuple[Dict, List]:
    """
    Extract features for same-key bigrams.
    
    Args:
        char: The character to analyze
        finger_map: Mapping of characters to fingers

    Returns:
        Tuple containing:
        - Dictionary of computed features
        - List of feature names
    """
    features = {
        'finger1': int(finger_map[char] == 1),
        'finger2': int(finger_map[char] == 2),
        'finger3': int(finger_map[char] == 3),
        'finger4': int(finger_map[char] == 4)
    }
    feature_names = list(features.keys())
    
    return features, feature_names

def precompute_all_bigram_features(layout_chars: List[str], column_map: Dict, 
                                   row_map: Dict, finger_map: Dict, 
                                   engram_position_values: Dict,
                                   row_position_values: Dict,
                                   bigrams: List,
                                   bigram_frequencies_array: np.ndarray) -> Tuple:
    """
    Precompute features for all possible bigrams based on the given layout characters.
    
    Args:
        layout_chars: List of all possible characters in the keyboard layout
        column_map: Mapping of characters to keyboard columns
        row_map: Mapping of characters to keyboard rows
        finger_map: Mapping of characters to fingers
        engram_position_values: Dictionary of engram position values
        row_position_values: Dictionary of row position values
        bigrams: List of bigrams ordered by frequency
        bigram_frequencies_array: Array of corresponding frequency values

    Returns:
        Tuple containing:
        - all_bigrams: All possible bigrams
        - features_df: DataFrame of all bigram features
        - feature_names: List of feature names
        - samekey_bigrams: All possible same-key bigrams
        - samekey_features_df: DataFrame of same-key bigram features
        - samekey_feature_names: List of same-key feature names
    """
    # Generate all possible 2-key bigrams (permutations of 2 unique characters)
    all_bigrams = [(x, y) for x, y in product(layout_chars, repeat=2) if x != y]
    # Generate all possible same-key bigrams
    samekey_bigrams = [(char, char) for char in layout_chars]

    # Extract features for each bigram
    feature_vectors = []
    feature_names = None
    samekey_feature_vectors = []
    samekey_feature_names = None

    # Extract features for the bigram pairs
    for char1, char2 in all_bigrams:
        features, feature_names = extract_features(char1, char2, column_map, row_map, 
                                                   finger_map, engram_position_values, 
                                                   row_position_values, bigrams, 
                                                   bigram_frequencies_array)
        feature_vectors.append(list(features.values()))

    # Extract features for same-key bigrams
    for char1, char2 in samekey_bigrams:
        samekey_features, samekey_feature_names = extract_features_samekey(char1, finger_map)
        samekey_feature_vectors.append(list(samekey_features.values()))

    # Convert to DataFrames
    features_df = pd.DataFrame(feature_vectors, columns=feature_names, index=all_bigrams)
    features_df.index = pd.MultiIndex.from_tuples(features_df.index)
    
    samekey_features_df = pd.DataFrame(samekey_feature_vectors, 
                                       columns=samekey_feature_names, 
                                       index=samekey_bigrams)
    samekey_features_df.index = pd.MultiIndex.from_tuples(samekey_features_df.index)

    return (all_bigrams, features_df, feature_names, 
            samekey_bigrams, samekey_features_df, samekey_feature_names)

def precompute_bigram_feature_differences(bigram_features: pd.DataFrame) -> Dict:
    """
    Precompute and store feature differences between bigram pairs.
    
    Args:
        bigram_features: DataFrame of precomputed features for each bigram

    Returns:
        Dictionary where each key is a tuple of bigrams (bigram1, bigram2),
        and the value is the precomputed feature differences
    """
    bigram_feature_differences = {}
    bigrams_list = list(bigram_features.index)

    # Loop over all pairs of bigrams
    for i, bigram1 in enumerate(bigrams_list):
        for j, bigram2 in enumerate(bigrams_list):
            if i <= j:  # Only compute differences for unique pairs
                abs_feature_diff = np.abs(bigram_features.loc[bigram1].values - 
                                        bigram_features.loc[bigram2].values)
                bigram_feature_differences[(bigram1, bigram2)] = abs_feature_diff
                # Store symmetric pair
                bigram_feature_differences[(bigram2, bigram1)] = abs_feature_diff

    return bigram_feature_differences
