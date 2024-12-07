from typing import Dict, List, Any
import numpy as np

from engram3.data import PreferenceDataset

def analyze_feature_importance(dataset: PreferenceDataset) -> Dict[str, Any]:
    """
    Compute comprehensive feature importance analysis.
    """
    results = {
        'correlations': {},
        'statistics': {},
        'sample_sizes': {}
    }
    
    for feature in dataset.get_feature_names():
        # Get feature differences and preference directions
        diffs = []
        prefs = []
        
        for pref in dataset.preferences:
            try:
                diff = pref.features1[feature] - pref.features2[feature]
                diffs.append(diff)
                prefs.append(1 if pref.preferred else -1)
            except KeyError:
                continue
                
        # Convert to numpy arrays
        diffs = np.array(diffs)
        prefs = np.array(prefs)
        
        # Skip if no valid samples
        if len(diffs) == 0:
            continue
            
        # Compute correlation with handling for constant features
        if np.std(diffs) == 0 or np.std(prefs) == 0:
            # If either variable is constant, correlation is undefined
            results['correlations'][feature] = 0.0
        else:
            # Suppress warnings and compute correlation
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.corrcoef(diffs, prefs)[0, 1]
                results['correlations'][feature] = 0.0 if np.isnan(corr) else corr
        
        # Compute basic statistics
        results['statistics'][feature] = {
            'mean': np.mean(diffs),
            'std': np.std(diffs),
            'min': np.min(diffs),
            'max': np.max(diffs)
        }
        
        # Store sample size
        results['sample_sizes'][feature] = len(diffs)
    
    return results

def find_sparse_regions(dataset: PreferenceDataset) -> List[Dict[str, float]]:
    """
    Identify sparse regions in feature space.
    
    Args:
        dataset: PreferenceDataset containing preferences
        
    Returns:
        List of dictionaries representing points in sparse regions
    """
    # Get all feature vectors
    vectors = []
    feature_names = dataset.get_feature_names()  # Call the method instead of accessing attribute
    
    for pref in dataset.preferences:
        for features in [pref.features1, pref.features2]:
            vector = [features[f] for f in feature_names]  # Use the retrieved feature names
            vectors.append(vector)
    
    vectors = np.array(vectors)
    
    # Find points with few neighbors
    sparse_points = []
    for i, v in enumerate(vectors):
        distances = np.linalg.norm(vectors - v, axis=1)
        n_neighbors = (distances < np.percentile(distances, 10)).sum()
        if n_neighbors < 5:  # Arbitrary threshold
            sparse_points.append(dict(zip(feature_names, v)))  # Use the retrieved feature names
            
    return sparse_points
