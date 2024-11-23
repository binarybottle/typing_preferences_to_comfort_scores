"""
Feature Extraction Module

This module handles the extraction and computation of features for keyboard layout analysis.
It provides functions for computing various metrics related to keyboard layout ergonomics.
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import List, Tuple, Dict

from bigram_feature_definitions import *

def extract_bigram_features(char1: str, char2: str, column_map: Dict, row_map: Dict, 
                            finger_map: Dict, engram_position_values: Dict,
                            row_position_values: Dict, bigrams: List, 
                            bigram_frequencies_array: np.ndarray, 
                            config: Dict) -> Tuple[Dict, List[str]]:
   """Extract features for a bigram pair based on configuration."""
   
   features_functions = function_map(char1, char2, column_map, row_map, finger_map,
                                     engram_position_values, row_position_values,
                                     bigrams, bigram_frequencies_array)

   features = {}
   all_features = config['features']['all_features']
   
   for feature_name in all_features:
       if feature_name in features_functions:
           features[feature_name] = features_functions[feature_name]()
   
   feature_names = list(features.keys())
   
   if config['features']['interactions']['enabled']:
       for pair in config['features']['interactions']['pairs']:
           interaction_name = f"{pair[0]}_{pair[1]}"
           features[interaction_name] = features[pair[0]] * features[pair[1]]
           feature_names.append(interaction_name)
   
   return features, feature_names

def extract_samekey_features(char: str, finger_map: Dict) -> Tuple[Dict, List]:
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

def precompute_all_bigram_features(layout_chars: List[str], 
                                 column_map: Dict, 
                                 row_map: Dict, 
                                 finger_map: Dict, 
                                 engram_position_values: Dict,
                                 row_position_values: Dict,
                                 bigrams: List,
                                 bigram_frequencies_array: np.ndarray,
                                 config: Dict) -> Tuple:
    """
    Precompute features for all possible bigrams based on configuration.
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

    # Extract features for same-key bigrams
    for char1, char2 in samekey_bigrams:
        samekey_features, names = extract_samekey_features(char1, finger_map)
        samekey_feature_vectors.append(list(samekey_features.values()))
        if samekey_feature_names is None:
            samekey_feature_names = names

    # Convert to DataFrames
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
