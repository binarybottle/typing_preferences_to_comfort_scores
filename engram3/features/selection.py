# engram3/features/selection.py
import yaml
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
import logging
from pathlib import Path
from collections import defaultdict
import json
import warnings
from engram3.data import PreferenceDataset
from engram3.models.bayesian import BayesianPreferenceModel

logger = logging.getLogger(__name__)

def load_interactions(filepath: str) -> List[List[str]]:
    """
    Load feature interactions from file.
    
    Args:
        filepath: Path to YAML file containing interaction definitions
        
    Returns:
        List of lists, where each inner list contains feature names to interact
        
    Example:
        interactions:
        - ['same_finger', 'sum_finger_values']
        - ['same_finger', 'rows_apart']
        - ['sum_finger_values', 'adj_finger_diff_row']
    """
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
            
        # Validate each interaction
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

class FeatureEvaluator:
    """Comprehensive feature evaluation for preference learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Get thresholds from config
        self.importance_threshold = config['feature_evaluation']['thresholds']['importance']
        self.stability_threshold = config['feature_evaluation']['thresholds']['stability']
        self.correlation_threshold = config['feature_evaluation']['thresholds']['correlation']
    
    def run_feature_selection(
        self,
        dataset: PreferenceDataset,
        output_dir: Path,
        feature_set_config: Dict
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Run feature selection multiple times."""
        
        # Get features configuration
        base_features = feature_set_config['base_features']
        interaction_file = feature_set_config.get('interactions_file')

        cv_config = self.config.get('model', {}).get('cross_validation', {})
        n_splits = cv_config.get('n_splits', 5)
        n_repetitions = cv_config.get('n_repetitions', 1)

        if interaction_file:
            interactions = load_interactions(interaction_file)
            logger.info(f"Loaded {len(interactions)} interactions from {interaction_file}")
        else:
            interactions = []

        # Prepare features including interactions
        X1, X2, y, all_features = self._prepare_feature_matrices(
            dataset, 
            base_features=base_features,
            interactions=interactions
        )
        
        # Validate all features (both base and interaction features)
        logger.info(f"\nStarting feature selection:")
        logger.info(f"Analyzing {len(all_features)} features "
                f"({len(base_features)} base + {len(interactions)} interactions)")
        
        # Validate features before processing
        valid_features = []
        problematic_features = set()
        
        # Initial feature validation
        for i, feature in enumerate(all_features):
            diff = X1[:, i] - X2[:, i]
            n_unique = len(np.unique(diff[~np.isnan(diff)]))
            n_nan = np.sum(np.isnan(diff))
            n_inf = np.sum(np.isinf(diff))
            
            if n_unique <= 1:
                logger.warning(f"Feature '{feature}' has constant differences - will be excluded")
                problematic_features.add(feature)
            elif n_nan > 0:
                logger.warning(f"Feature '{feature}' has {n_nan} NaN values - will be excluded")
                problematic_features.add(feature)
            elif n_inf > 0:
                logger.warning(f"Feature '{feature}' has {n_inf} infinite values - will be excluded")
                problematic_features.add(feature)
            else:
                valid_features.append(feature)
        
        logger.info(f"Using {len(valid_features)} valid features for selection")
        if problematic_features:
            logger.info("Excluded features:")
            for feat in sorted(problematic_features):
                if feat in base_features:
                    logger.info(f"  - {feat} (base feature)")
                else:
                    logger.info(f"  - {feat} (interaction feature)")
                    
        # Collect results and diagnostics across repetitions
        feature_names = dataset.get_feature_names()
        feature_results = defaultdict(list)
        all_diagnostics = {
            'importance_diagnostics': [],
            'correlation_diagnostics': [],
            'problematic_features': set(),
            'iteration_metrics': []
        }
                    
        for iter_idx in range(n_repetitions):
            logger.debug(f"Feature selection iteration {iter_idx+1}/{n_repetitions}")
            
            try:
                # Calculate correlations once
                correlations, correlation_diagnostics = self._calculate_feature_correlations(
                    X1, X2, feature_names)
                
                # Run cross-validation
                model = BayesianPreferenceModel()
                cv_results = model.cross_validate(dataset)
                
                if not cv_results.get('feature_effects'):
                    logger.warning("No feature effects obtained from cross-validation")
                    continue
                
                # Evaluate features using pre-calculated correlations
                eval_results = self.evaluate_features(
                    dataset, 
                    cv_results,
                    output_dir=output_dir,
                    feature_matrices=(X1, X2, y),
                    correlation_results=(correlations, correlation_diagnostics)
                )
                
                if not eval_results:
                    logger.warning(f"No evaluation results in iteration {iter_idx + 1}")
                    continue
                
                # Store basic results
                for feature in valid_features:  # Only store results for valid features
                    if feature in eval_results.get('importance', {}):
                        feature_results[feature].append({
                            'importance': eval_results['importance'][feature],
                            'stability': eval_results.get('stability', {}).get(feature, {})
                        })
                
                # Store diagnostics
                all_diagnostics['importance_diagnostics'].append(
                    eval_results.get('diagnostics', {}))
                all_diagnostics['correlation_diagnostics'].append(correlation_diagnostics)
                
                # Store iteration metrics
                all_diagnostics['iteration_metrics'].append({
                    'iteration': iter_idx + 1,
                    'n_features': len(valid_features),
                    'n_problematic': len(problematic_features),
                    'correlation_issues': len(correlation_diagnostics.get('correlation_issues', [])),
                    'highly_correlated_pairs': len(correlation_diagnostics.get('highly_correlated_pairs', []))
                })
                
            except Exception as e:
                logger.error(f"Error in iteration {iter_idx + 1}: {str(e)}")
                continue

        if not feature_results:
            logger.warning("No valid results collected across iterations")
            return [], all_diagnostics

        # Analyze results and get recommendations
        results = self._analyze_selection_results(feature_results, all_diagnostics)
        
        # Save detailed reports
        if output_dir:
            self._save_diagnostic_report(results, output_dir)
            self._save_detailed_report(results, output_dir)

        return results['selected_features'], results
    
    def evaluate_features(
        self,
        dataset: PreferenceDataset,
        model_results: Dict[str, Dict],
        output_dir: Optional[Path] = None,
        feature_matrices: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        correlation_results: Optional[Tuple[pd.DataFrame, Dict]] = None
    ) -> Dict[str, Dict]:
        """
        Comprehensive feature evaluation.
        """
        try:
            feature_names = dataset.get_feature_names()
            logger.debug(f"Feature names: {feature_names}")
            logger.debug(f"Model results type: {type(model_results)}")
            logger.debug(f"Model results keys: {model_results.keys()}")
            
            # Use provided matrices if available, otherwise calculate them
            if feature_matrices is not None:
                X1, X2, y = feature_matrices
                logger.debug(f"Using provided feature matrices: X1 shape {X1.shape}")
            else:
                X1, X2, y = self._prepare_feature_matrices(dataset)
            
            # Calculate feature importance
            logger.info("Calculating feature importance...")
            importance = self._calculate_feature_importance(X1, X2, y, feature_names, model_results)
            
            if not importance:
                logger.warning("No feature importance scores were calculated")
                return {}
            
            # Use provided correlation results or calculate new ones
            if correlation_results is not None:
                correlations, corr_diagnostics = correlation_results
            else:
                correlations, corr_diagnostics = self._calculate_feature_correlations(
                    X1, X2, feature_names)
                
            # Get stability metrics from model results
            stability = model_results.get('stability', {})
            if not stability:
                logger.warning("No stability metrics in model results")
                stability = {f: {'sign_consistency': 0.0} for f in feature_names}
                
            # Generate recommendations
            recommendations = self._generate_recommendations(
                importance, stability, correlations, corr_diagnostics)
                
            # Organize results
            results = {
                'importance': importance,
                'stability': stability,
                'correlations': correlations,
                'correlation_diagnostics': corr_diagnostics,
                'recommendations': recommendations,
                'selected_features': recommendations.get('strong_features', []),
                'rejected_features': recommendations.get('weak_features', []),
                'all_features': feature_names,
                'diagnostics': {
                    'n_samples': len(dataset.preferences),
                    'evaluation_metrics': model_results.get('metrics', {})
                }
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Feature evaluation failed: {str(e)}")
            logger.error(f"Traceback:", exc_info=True)  # Add full traceback
            return {}

    def _validate_features(self, feature_data: Dict[str, Any]) -> List[str]:
        """Validate and filter features."""
        valid_features = []
        for feature in feature_data:
            if feature == 'correct':
                continue  # Skip 'correct' as it's not a feature
                
            values = feature_data[feature]
            n_nan = np.sum(np.isnan(values))
            n_total = len(values)
            
            # Allow features with some NaN values if they have sufficient valid data
            if n_nan > 0:
                valid_ratio = (n_total - n_nan) / n_total
                if valid_ratio >= 0.9:  # Keep features with at least 90% valid data
                    logger.info(f"Feature '{feature}' has {n_nan} NaN values but sufficient valid data")
                    valid_features.append(feature)
                else:
                    logger.warning(f"Feature '{feature}' excluded: {n_nan}/{n_total} NaN values")
            else:
                valid_features.append(feature)
                
        return valid_features
                          
    def _analyze_selection_results(
        self,
        feature_results: Dict[str, List[Dict]],
        diagnostics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze feature selection results across iterations with diagnostics.
        """
        logger = logging.getLogger(__name__)
        aggregated_results = {}
        importance_metrics = {}  # Add this to store importance metrics
        
        # First pass: Calculate basic metrics for each feature
        for feature, iterations in feature_results.items():
            # Extract scores across iterations
            importance_scores = [it['importance'].get('combined_score', 0.0) for it in iterations]
            stability_scores = [it['stability'].get('sign_consistency', 0.0) for it in iterations]
            
            # Calculate basic metrics
            aggregated_results[feature] = {
                'mean_importance': np.mean(importance_scores),
                'std_importance': np.std(importance_scores),
                'mean_stability': np.mean(stability_scores),
                'std_stability': np.std(stability_scores),
                'selection_frequency': sum(
                    s >= self.stability_threshold and i >= self.importance_threshold
                    for s, i in zip(stability_scores, importance_scores)
                ) / len(iterations),
                'is_problematic': feature in diagnostics.get('problematic_features', set())
            }
            
            # Store importance metrics
            importance_metrics[feature] = {
                'combined_score': aggregated_results[feature]['mean_importance']
            }
            
            # Add correlation information
            high_correlations = []
            for corr_diag in diagnostics.get('correlation_diagnostics', []):
                for pair in corr_diag.get('highly_correlated_pairs', []):
                    if feature in (pair['feature1'], pair['feature2']):
                        other_feature = pair['feature2'] if feature == pair['feature1'] else pair['feature1']
                        high_correlations.append((other_feature, pair['correlation']))
            
            if high_correlations:
                aggregated_results[feature]['high_correlations'] = high_correlations

        # Second pass: Make selection decisions
        selected_features = []
        rejected_features = []
        selection_diagnostics = {
            'n_iterations': len(next(iter(feature_results.values()))),
            'n_total_features': len(feature_results),
            'rejected_features': []
        }

        for feature, metrics in aggregated_results.items():
            # Skip problematic features
            if metrics['is_problematic']:
                rejected_features.append((feature, 'problematic feature'))
                continue
            
            # Calculate selection criteria
            meets_importance = metrics['mean_importance'] >= self.importance_threshold
            meets_stability = metrics['mean_stability'] >= self.stability_threshold
            is_consistent = metrics['selection_frequency'] >= 0.7  # Selected in 70% of iterations
            
            # Make selection decision
            if meets_importance and meets_stability and is_consistent:
                selected_features.append(feature)
            else:
                reason = []
                if not meets_importance:
                    reason.append('low importance')
                if not meets_stability:
                    reason.append('low stability')
                if not is_consistent:
                    reason.append('inconsistent selection')
                rejected_features.append((feature, ', '.join(reason)))

        # Update selection diagnostics
        selection_diagnostics.update({
            'n_selected': len(selected_features),
            'n_rejected': len(rejected_features),
            'rejected_features': rejected_features
        })

        # Log selection results
        logger.info(f"\nFeature selection results:")
        logger.info(f"- Selected {len(selected_features)} features")
        logger.info(f"- Rejected {len(rejected_features)} features")
        
        return {
            'selected_features': selected_features,
            'aggregated_metrics': aggregated_results,
            'selection_diagnostics': selection_diagnostics,
            'importance': importance_metrics,
            'stability': {
                feature: {
                    'effect_cv': metrics['std_importance'] / max(metrics['mean_importance'], 1e-10),
                    'sign_consistency': metrics['mean_stability'],
                    'relative_range': (metrics.get('max_importance', 0.0) - 
                                     metrics.get('min_importance', 0.0)) / 
                                     max(abs(metrics['mean_importance']), 1e-10)
                }
                for feature, metrics in aggregated_results.items()
            }
        }
            
    def _save_selection_results(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """
        Save feature selection results in both human-readable and JSON formats.
        
        Args:
            results: Dictionary containing selection results
            output_dir: Output directory
        """
        # Save human-readable report
        report_file = output_dir / "feature_selection_report.txt"
        with open(report_file, 'w') as f:
            f.write("Feature Selection Report\n")
            f.write("=====================\n\n")
            
            # Selected Features
            selected = results.get('selected_features', [])
            f.write("Selected Features:\n")
            f.write("-----------------\n")
            if selected:
                for feature in selected:
                    metrics = results['aggregated_metrics'][feature]
                    f.write(f"\n{feature}:\n")
                    f.write(f"Mean importance: {metrics['mean_importance']:.3f} ± {metrics['std_importance']:.3f}\n")
                    f.write(f"Mean stability: {metrics['mean_stability']:.3f} ± {metrics['std_stability']:.3f}\n")
                    f.write(f"Selection frequency: {metrics['selection_frequency']:.1%}\n")
            
            # Rejected Features
            f.write("\nRejected Features:\n")
            f.write("----------------\n")
            for feature, metrics in results['aggregated_metrics'].items():
                if feature not in selected:
                    f.write(f"\n{feature}:\n")
                    f.write(f"Mean importance: {metrics['mean_importance']:.3f} ± {metrics['std_importance']:.3f}\n")
                    f.write(f"Mean stability: {metrics['mean_stability']:.3f} ± {metrics['std_stability']:.3f}\n")
                    f.write(f"Selection frequency: {metrics['selection_frequency']:.1%}\n")
            
            # Summary Statistics
            f.write("\nSummary Statistics:\n")
            f.write("-----------------\n")
            f.write(f"Total features evaluated: {len(results['aggregated_metrics'])}\n")
            f.write(f"Features selected: {len(selected)}\n")
            f.write(f"Features rejected: {len(results['aggregated_metrics']) - len(selected)}\n")
        # Save JSON for programmatic use
        json_file = output_dir / "feature_selection_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {report_file} and {json_file}")
                
    def _prepare_feature_matrices(
        self,
        dataset: PreferenceDataset,
        base_features: List[str],
        interactions: List[List[str]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrices with detailed validation and interactions."""
        logger.debug("\nDEBUG: Starting feature matrix preparation")
        
        X1 = []  # First bigram features
        X2 = []  # Second bigram features
        y = []   # Preferences
        
        # Start with base features
        logger.debug("Preparing base feature matrices...")
        logger.debug(f"Number of base features: {len(base_features)}")
        logger.debug(f"Base features: {base_features}")
        
        # Keep track of all features including interactions
        all_features = base_features.copy()
        
        for pref in dataset.preferences:
            # Log first few preferences for debugging
            if len(X1) < 5:
                logger.debug(f"\nPreference for pair {len(X1)+1}:")
                logger.debug(f"Bigram 1: {pref.bigram1}")
                logger.debug(f"Bigram 2: {pref.bigram2}")
                logger.debug(f"Preferred: {pref.preferred}")
                logger.debug("Features 1:")
                for fname, fval in pref.features1.items():
                    logger.debug(f"  {fname}: {fval}")
                logger.debug("Features 2:")
                for fname, fval in pref.features2.items():
                    logger.debug(f"  {fname}: {fval}")
            
            try:
                # Extract base features in consistent order
                feat1 = [pref.features1[f] for f in base_features]
                feat2 = [pref.features2[f] for f in base_features]
                
                X1.append(feat1)
                X2.append(feat2)
                y.append(float(pref.preferred))
                
            except KeyError as e:
                logger.error(f"Missing feature in preference: {str(e)}")
                logger.error(f"Available features1: {list(pref.features1.keys())}")
                logger.error(f"Available features2: {list(pref.features2.keys())}")
                raise
        
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)
        
        # Add interaction features
        logger.debug(f"\nAdding {len(interactions)} interaction features...")
        
        for interaction in interactions:
            if all(f in base_features for f in interaction):
                # Get indices for the interaction features
                indices = [base_features.index(f) for f in interaction]
                interaction_name = '_x_'.join(interaction)
                
                logger.debug(f"Creating interaction: {interaction_name}")
                
                # Calculate interaction values
                interaction1 = X1[:, indices[0]].copy()
                interaction2 = X2[:, indices[0]].copy()
                
                for idx in indices[1:]:
                    interaction1 *= X1[:, idx]
                    interaction2 *= X2[:, idx]
                
                # Add interaction features to matrices
                X1 = np.column_stack([X1, interaction1])
                X2 = np.column_stack([X2, interaction2])
                all_features.append(interaction_name)
                
                logger.debug(f"Added interaction feature: {interaction_name}")
        
        # Validate all features (base + interactions)
        logger.debug("\nFeature matrix validation:")
        logger.debug(f"Final matrix shapes - X1: {X1.shape}, X2: {X2.shape}, y: {y.shape}")
        logger.debug(f"Total features: {len(all_features)}")
        
        # Check for invalid values in each feature
        for i, feature in enumerate(all_features):
            n_nan1 = np.sum(np.isnan(X1[:, i]))
            n_nan2 = np.sum(np.isnan(X2[:, i]))
            n_inf1 = np.sum(np.isinf(X1[:, i]))
            n_inf2 = np.sum(np.isinf(X2[:, i]))
            n_unique1 = len(np.unique(X1[:, i]))
            n_unique2 = len(np.unique(X2[:, i]))
            
            is_interaction = feature not in base_features
            feature_type = "Interaction" if is_interaction else "Base"
            
            logger.debug(f"\n{feature_type} feature '{feature}':")
            logger.debug(f"  Bigram 1 features: {n_unique1} unique values, {n_nan1} NaN, {n_inf1} Inf")
            logger.debug(f"  Bigram 2 features: {n_unique2} unique values, {n_nan2} NaN, {n_inf2} Inf")
            logger.debug(f"  Range Bigram 1 features: [{np.min(X1[:, i])}, {np.max(X1[:, i])}]")
            logger.debug(f"  Range Bigram 2 features: [{np.min(X2[:, i])}, {np.max(X2[:, i])}]")
            
            # Keep these as warnings
            if n_unique1 == 1 or n_unique2 == 1:
                logger.warning(f"  WARNING: {feature_type} feature '{feature}' has constant values")
            if n_nan1 > 0 or n_nan2 > 0:
                logger.warning(f"  WARNING: {feature_type} feature '{feature}' has NaN values")
            if n_inf1 > 0 or n_inf2 > 0:
                logger.warning(f"  WARNING: {feature_type} feature '{feature}' has Inf values")
        
        return X1, X2, y, all_features
     
    def _calculate_feature_importance(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_results: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature importance with robust handling of edge cases.
        
        Args:
            X1: Feature matrix for first bigram in each pair
            X2: Feature matrix for second bigram in each pair
            y: Binary preference indicators
            feature_names: List of feature names
            model_results: Results from model cross-validation
        """
        importance = {}

        logger.info("Starting feature importance calculation...")
        
        for i, feature in enumerate(feature_names):
            logger.debug(f"\nAnalyzing feature: {feature}")
            
            # Calculate feature differences
            feat_diff = X1[:, i] - X2[:, i]
            
            # Diagnostic checks
            n_total = len(feat_diff)
            n_unique = len(np.unique(feat_diff[~np.isnan(feat_diff)]))
            n_nan = np.sum(np.isnan(feat_diff))
            n_inf = np.sum(np.isinf(feat_diff))
            n_zero = np.sum(feat_diff == 0)
            
            logger.debug(f"Feature '{feature}' diagnostics:")
            logger.debug(f"- Total values: {n_total}")
            logger.debug(f"- Unique values: {n_unique}")
            logger.debug(f"- NaN values: {n_nan}")
            logger.debug(f"- Inf values: {n_inf}")
            logger.debug(f"- Zero values: {n_zero}")
            
            # Initialize metrics
            importance[feature] = {
                'mutual_info': 0.0,
                'correlation': 0.0,
                'model_effect_mean': 0.0,
                'model_effect_std': 0.0,
                'combined_score': 0.0,
                'diagnostics': {
                    'n_total': n_total,
                    'n_unique': n_unique,
                    'n_nan': n_nan,
                    'n_inf': n_inf,
                    'n_zero': n_zero,
                    'is_valid': True,
                    'issues': []
                }
            }
            
            # Check for validity
            if n_nan == n_total or n_inf == n_total:
                importance[feature]['diagnostics']['is_valid'] = False
                importance[feature]['diagnostics']['issues'].append(
                    'all values invalid')
                continue
                
            if n_unique <= 1:
                importance[feature]['diagnostics']['is_valid'] = False
                importance[feature]['diagnostics']['issues'].append(
                    'constant values')
                continue
            
            # Get valid data for calculations
            valid_mask = ~(np.isnan(feat_diff) | np.isinf(feat_diff))
            if not np.any(valid_mask):
                importance[feature]['diagnostics']['is_valid'] = False
                importance[feature]['diagnostics']['issues'].append(
                    'no valid values')
                continue
                
            valid_diff = feat_diff[valid_mask]
            valid_y = y[valid_mask]
            
            # Calculate correlation with proper error handling
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(valid_diff, valid_y)
                    if np.isnan(corr):
                        corr = 0.0
                        importance[feature]['diagnostics']['issues'].append(
                            'correlation calculation failed')
            except Exception as e:
                logger.warning(f"Correlation calculation failed for {feature}: {str(e)}")
                corr = 0.0
                importance[feature]['diagnostics']['issues'].append(
                    'correlation error')
            
            # Calculate mutual information
            try:
                # Binarize feature differences for MI calculation
                median_diff = np.median(valid_diff)
                feat_diff_bin = valid_diff > median_diff
                mi_score = mutual_info_score(feat_diff_bin, valid_y)
                if np.isnan(mi_score):
                    mi_score = 0.0
                    importance[feature]['diagnostics']['issues'].append(
                        'MI calculation failed')
            except Exception as e:
                logger.warning(f"MI calculation failed for {feature}: {str(e)}")
                mi_score = 0.0
                importance[feature]['diagnostics']['issues'].append(
                    'MI error')
            
            # Get model-based importance
            model_effect = model_results.get('feature_effects', {}).get(feature, {})
            effect_mean = model_effect.get('mean', 0.0)
            effect_std = model_effect.get('std', 0.0)
            
            # Store results
            importance[feature].update({
                'mutual_info': float(mi_score),
                'correlation': float(corr),
                'model_effect_mean': float(effect_mean),
                'model_effect_std': float(effect_std),
                'combined_score': float(
                    0.4 * abs(effect_mean) +
                    0.3 * abs(corr) +
                    0.3 * mi_score
                )
            })
            
            # Log results
            logger.debug(f"""
    Feature: {feature}
    Correlation: {corr:.3f}
    Mutual Information: {mi_score:.3f}
    Model Effect: {effect_mean:.3f} ± {effect_std:.3f}
    Combined Score: {importance[feature]['combined_score']:.3f}
    Issues: {', '.join(importance[feature]['diagnostics']['issues']) if 
            importance[feature]['diagnostics']['issues'] else 'none'}
    """)
        
        return importance
    
    def _calculate_feature_correlations(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """Calculate feature correlations with detailed diagnostics."""
        logger.debug("Starting feature correlation analysis")
        logger.debug(f"Analyzing {len(feature_names)} features")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_features = len(feature_names)
            corr_matrix = np.zeros((n_features, n_features))
            diagnostics = {
                'feature_stats': {},
                'correlation_issues': [],
                'highly_correlated_pairs': []
            }
            
            # Analyze individual features
            for i, feature in enumerate(feature_names):
                if len(np.unique(X1[:, i] - X2[:, i])) <= 1:
                    logger.debug(f"Feature '{feature}' has constant differences")   
                                
                # Detailed analysis of the difference vector
                diff = X1[:, i] - X2[:, i]
                valid_mask = ~(np.isnan(diff) | np.isinf(diff))
                valid_diffs = diff[valid_mask]
                
                stats = {
                    'n_total': len(diff),
                    'n_valid': len(valid_diffs),
                    'n_unique': len(np.unique(valid_diffs)) if len(valid_diffs) > 0 else 0,
                    'n_nan': np.sum(np.isnan(diff)),
                    'n_inf': np.sum(np.isinf(diff)),
                    'n_zero': np.sum(diff == 0),
                    'mean': np.mean(valid_diffs) if len(valid_diffs) > 0 else np.nan,
                    'std': np.std(valid_diffs) if len(valid_diffs) > 0 else np.nan,
                    'min': np.min(valid_diffs) if len(valid_diffs) > 0 else np.nan,
                    'max': np.max(valid_diffs) if len(valid_diffs) > 0 else np.nan
                }
                
                diagnostics['feature_stats'][feature] = stats
                
                # Log detailed feature statistics at DEBUG level
                logger.debug(f"\nFeature '{feature}':")
                logger.debug(f"  Total values: {stats['n_total']}")
                logger.debug(f"  Valid values: {stats['n_valid']}")
                logger.debug(f"  Unique values: {stats['n_unique']}")
                if stats['n_unique'] < 5:
                    unique_vals = np.unique(valid_diffs)
                    logger.debug(f"  Unique values found: {unique_vals}")
                logger.debug(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                logger.debug(f"  Mean ± std: {stats['mean']:.3f} ± {stats['std']:.3f}")
                
                # Keep warnings at WARNING level
                if stats['n_unique'] == 1:
                    logger.warning(f"  WARNING: Feature '{feature}' has constant differences")
                if stats['n_nan'] > 0:
                    logger.warning(f"  WARNING: Feature '{feature}' has {stats['n_nan']} NaN differences")
                if stats['n_inf'] > 0:
                    logger.warning(f"  WARNING: Feature '{feature}' has {stats['n_inf']} infinite differences")
                if stats['n_zero'] == stats['n_valid']:
                    logger.warning(f"  WARNING: Feature '{feature}' has all zero differences")
            
            # Calculate correlations
            logger.debug("\nCalculating feature correlations:")
            for i in range(n_features):
                for j in range(n_features):
                    diff_i = X1[:, i] - X2[:, i]
                    diff_j = X1[:, j] - X2[:, j]
                    
                    # Skip if either feature has issues
                    has_issues = False
                    reasons = []
                    
                    # Only mark as constant if truly constant
                    unique_i = len(np.unique(diff_i[~np.isnan(diff_i)]))
                    unique_j = len(np.unique(diff_j[~np.isnan(diff_j)]))
                    
                    if unique_i == 1:
                        reasons.append(f"{feature_names[i]} has constant differences")
                        has_issues = True
                    if unique_j == 1:
                        reasons.append(f"{feature_names[j]} has constant differences")
                        has_issues = True
                    
                    if has_issues:
                        corr = 0.0
                        if i != j:  # Don't report self-correlations
                            diagnostics['correlation_issues'].append({
                                'feature1': feature_names[i],
                                'feature2': feature_names[j],
                                'reason': '; '.join(reasons)
                            })
                    else:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                result = spearmanr(diff_i, diff_j)
                                corr = result.correlation if not np.isnan(result.correlation) else 0.0
                        except Exception as e:
                            logger.error(f"Error calculating correlation between "
                                    f"'{feature_names[i]}' and '{feature_names[j]}': {str(e)}")
                            corr = 0.0
                    
                    corr_matrix[i, j] = corr
                    
                    # Track highly correlated features
                    if i < j and abs(corr) > self.correlation_threshold:
                        diagnostics['highly_correlated_pairs'].append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr)
                        })
            
            return (pd.DataFrame(corr_matrix, 
                            index=feature_names,
                            columns=feature_names),
                    diagnostics)
     
    def _generate_recommendations(
        self,
        importance: Dict[str, Dict[str, float]],
        stability: Dict[str, Dict[str, float]],
        correlations: pd.DataFrame,
        correlation_diagnostics: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Generate feature recommendations."""
        features = list(importance.keys())
        
        # Categorize features
        strong_features = []
        unstable_features = []
        weak_features = []
        
        for feature in features:
            imp = importance[feature]['combined_score']
            stab = stability[feature]['sign_consistency']
            
            if imp >= self.importance_threshold:
                if stab >= self.stability_threshold:
                    strong_features.append(feature)
                else:
                    unstable_features.append(feature)
            else:
                weak_features.append(feature)
        
        # Find correlated groups
        correlated_groups = []
        used_features = set()
        
        for i, feat1 in enumerate(features):
            if feat1 in used_features:
                continue
                
            group = [feat1]
            used_features.add(feat1)
            
            for feat2 in features[i+1:]:
                if feat2 in used_features:
                    continue
                    
                # Now correctly accessing correlation from DataFrame
                if abs(correlations.loc[feat1, feat2]) >= self.correlation_threshold:
                    group.append(feat2)
                    used_features.add(feat2)
                    
            if len(group) > 1:
                correlated_groups.append(group)
        
        return {
            'strong_features': strong_features,
            'unstable_features': unstable_features,
            'weak_features': weak_features,
            'correlated_groups': correlated_groups
        }
    
    def _save_diagnostic_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save comprehensive feature selection diagnostic report."""
        report_file = output_dir / "feature_selection_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("Feature Selection Report\n")
            f.write("======================\n\n")
            
            # Summary statistics
            diagnostics = results.get('selection_diagnostics', {})
            f.write("1. Summary\n")
            f.write("---------\n")
            f.write(f"Total features evaluated: {diagnostics.get('n_total_features', 0)}\n")
            f.write(f"Features selected: {diagnostics.get('n_selected', 0)}\n")
            f.write(f"Features rejected: {diagnostics.get('n_rejected', 0)}\n")
            f.write(f"Features flagged: {diagnostics.get('n_flagged', 0)}\n\n")
            
            # Selected features
            f.write("2. Selected Features\n")
            f.write("-----------------\n")
            for feature in results.get('selected_features', []):
                metrics = results.get('aggregated_metrics', {}).get(feature, {})
                importance = metrics.get('mean_importance', 0.0)
                stability = metrics.get('mean_stability', 0.0)
                f.write(f"{feature}:\n")
                f.write(f"  Importance: {importance:.3f}\n")
                f.write(f"  Stability: {stability:.3f}\n\n")

            # Rejected features with reasons
            f.write("3. Rejected Features\n")
            f.write("-----------------\n")
            for feature, reason in diagnostics.get('rejected_features', []):
                metrics = results.get('aggregated_metrics', {}).get(feature, {})
                importance = metrics.get('mean_importance', 0.0)
                f.write(f"{feature}:\n")
                f.write(f"  Importance: {importance:.3f}\n")
                f.write(f"  Reason: {reason}\n\n")

    def _save_detailed_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save comprehensive feature selection report."""
        report_file = output_dir / "feature_selection_report.txt"
        
        with open(report_file, 'w') as f:
            # Summary section
            f.write("Summary\n")
            f.write("---------\n")
            f.write(f"Total features evaluated: {len(results['importance'])}\n")
            f.write(f"Features selected: {len(results['selected_features'])}\n")
            f.write(f"Features rejected: {len(results.get('rejected_features', []))}\n")
            f.write(f"Features flagged: {len(results.get('flagged_features', []))}\n\n")
            
            # Selected Features section
            f.write("Selected Features\n")
            f.write("----------------\n")
            for feature in sorted(results['selected_features']):
                if '_x_' not in feature:  # Base features only
                    imp = results['importance'].get(feature, {})
                    stab = results['stability'].get(feature, {})
                    
                    f.write(f"\n{feature}:\n")
                    f.write(f"Importance: {imp.get('combined_score', 0.0):.3f}\n")
                    f.write(f"Stability: {stab.get('sign_consistency', 0.0):.3f}\n")
                    f.write(f"Model effect: {imp.get('model_effect_mean', 0.0):.3f} ± {imp.get('model_effect_std', 0.0):.3f}\n")
                    f.write(f"Correlation: {imp.get('correlation', 0.0):.3f}\n")
                    f.write(f"Mutual information: {imp.get('mutual_info', 0.0):.3f}\n")
                    f.write(f"Effect CV: {stab.get('effect_cv', 0.0):.3f}\n")
                    f.write(f"Relative range: {stab.get('relative_range', 0.0):.3f}\n")
            
            # Non-selected Features section
            f.write("\nNon-selected Features\n")
            f.write("--------------------\n")
            for feature in sorted(set(results['importance'].keys()) - set(results['selected_features'])):
                if '_x_' not in feature:  # Base features only
                    imp = results['importance'].get(feature, {})
                    stab = results['stability'].get(feature, {})
                    
                    f.write(f"\n{feature}:\n")
                    f.write(f"Importance: {imp.get('combined_score', 0.0):.3f}\n")
                    f.write(f"Stability: {stab.get('sign_consistency', 0.0):.3f}\n")
                    f.write(f"Model effect: {imp.get('model_effect_mean', 0.0):.3f} ± {imp.get('model_effect_std', 0.0):.3f}\n")
                    f.write(f"Correlation: {imp.get('correlation', 0.0):.3f}\n")
                    f.write(f"Mutual information: {imp.get('mutual_info', 0.0):.3f}\n")
                    f.write(f"Effect CV: {stab.get('effect_cv', 0.0):.3f}\n")
                    f.write(f"Relative range: {stab.get('relative_range', 0.0):.3f}\n")
                    if feature in results.get('rejected_features', []):
                        reasons = [r[1] for r in results['rejected_features'] if r[0] == feature]
                        if reasons:
                            f.write(f"Rejection reason: {reasons[0]}\n")

            # Feature Interactions section
            f.write("\nFeature Interactions\n")
            f.write("-------------------\n")
            interaction_features = sorted([f for f in results['importance'].keys() if '_x_' in f])
            
            if interaction_features:
                for feature in interaction_features:
                    imp = results['importance'].get(feature, {})
                    stab = results['stability'].get(feature, {})
                    status = "selected" if feature in results['selected_features'] else "rejected"
                    
                    f.write(f"\n{feature}:\n")
                    f.write(f"Status: {status}\n")
                    f.write(f"Importance: {imp.get('combined_score', 0.0):.3f}\n")
                    f.write(f"Stability: {stab.get('sign_consistency', 0.0):.3f}\n")
                    f.write(f"Model effect: {imp.get('model_effect_mean', 0.0):.3f} ± {imp.get('model_effect_std', 0.0):.3f}\n")
                    f.write(f"Correlation: {imp.get('correlation', 0.0):.3f}\n")
                    f.write(f"Mutual information: {imp.get('mutual_info', 0.0):.3f}\n")
                    f.write(f"Effect CV: {stab.get('effect_cv', 0.0):.3f}\n")
                    f.write(f"Relative range: {stab.get('relative_range', 0.0):.3f}\n")
                    if feature in results.get('rejected_features', []):
                        reasons = [r[1] for r in results['rejected_features'] if r[0] == feature]
                        if reasons:
                            f.write(f"Rejection reason: {reasons[0]}\n")
            else:
                f.write("\nNo feature interactions evaluated.\n")

                                        
        
