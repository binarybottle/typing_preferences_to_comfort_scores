"""
Feature selection

These methods define validate, evaluate, and select bigram features for keyboard layout analysis.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
import logging
from pathlib import Path
from collections import defaultdict
import warnings

from engram3.utils import load_interactions
from engram3.data import PreferenceDataset
from engram3.models.bayesian import BayesianPreferenceModel

logger = logging.getLogger(__name__)

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
        if interaction_file:
            interactions = load_interactions(interaction_file)
            logger.info(f"Loaded {len(interactions)} interactions from {interaction_file}")
        else:
            interactions = []

        # Prepare features including interactions
        X1, X2, y, valid_features = self._prepare_feature_matrices(
            dataset, 
            base_features=base_features,
            interactions=interactions
        )
        
        logger.info(f"\nStarting feature selection:")
        logger.info(f"Analyzing {len(valid_features)} features")
        
        if not valid_features:
            logger.error("No valid features found after preparation")
            return [], {'error': 'No valid features'}

        feature_results = defaultdict(list)
        all_diagnostics = {
            'importance': {},
            'importance_diagnostics': [],
            'correlation_diagnostics': [],
            'problematic_features': set(),
            'iteration_metrics': []
        }
                    
        cv_config = self.config.get('model', {}).get('cross_validation', {})
        n_repetitions = cv_config.get('n_repetitions', 1)
        for iter_idx in range(n_repetitions):
            logger.debug(f"Feature selection iteration {iter_idx+1}/{n_repetitions}")
            
            try:
                # Calculate correlations once
                correlations, correlation_diagnostics = self._calculate_feature_correlations(
                    X1, X2, valid_features)  # Changed from feature_names
                
                # Run cross-validation
                model = BayesianPreferenceModel(config=self.config)
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
                for feature in valid_features: 
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
            self._save_detailed_report(results, output_dir)

        # Update all_diagnostics with results before returning
        all_diagnostics['importance'] = results.get('importance', {})
        all_diagnostics['stability'] = results.get('stability', {})
        return results['selected_features'], all_diagnostics
    
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
            recommendations = self._recommend_existing_features(
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

    def analyze_feature_importance(dataset: PreferenceDataset) -> Dict[str, Any]:
        """
        Compute feature importance measures.
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
                    # Skip if either feature value is None
                    if pref.features1[feature] is None or pref.features2[feature] is None:
                        continue
                        
                    diff = pref.features1[feature] - pref.features2[feature]
                    if not np.isnan(diff):  # Skip NaN differences
                        diffs.append(diff)
                        prefs.append(1 if pref.preferred else -1)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Error processing feature {feature}: {str(e)}")
                    continue
                    
            # Convert to numpy arrays
            diffs = np.array(diffs)
            prefs = np.array(prefs)
            
            # Skip if no valid samples
            if len(diffs) == 0:
                logger.warning(f"No valid samples for feature {feature}")
                results['correlations'][feature] = 0.0
                results['statistics'][feature] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
                results['sample_sizes'][feature] = 0
                continue
                
            try:
                # Calculate correlation with proper error handling
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    correlation = scipy.stats.spearmanr(diffs, prefs)
                    results['correlations'][feature] = (
                        correlation.correlation if not np.isnan(correlation.correlation) else 0.0
                    )
            except Exception as e:
                logger.warning(f"Correlation calculation failed for {feature}: {str(e)}")
                results['correlations'][feature] = 0.0
            
            # Compute basic statistics
            results['statistics'][feature] = {
                'mean': float(np.mean(diffs)),
                'std': float(np.std(diffs)),
                'min': float(np.min(diffs)),
                'max': float(np.max(diffs))
            }
            
            # Store sample size
            results['sample_sizes'][feature] = len(diffs)
            
            # Log stats
            logger.debug(f"\nFeature: {feature}")
            logger.debug(f"Valid samples: {len(diffs)}")
            logger.debug(f"Correlation: {results['correlations'][feature]:.3f}")
            logger.debug(f"Range: [{results['statistics'][feature]['min']:.3f}, "
                        f"{results['statistics'][feature]['max']:.3f}]")
        
        return results

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
        aggregated_results = {}
        importance_metrics = {}  # Add this to store importance metrics
        
        # First pass: Calculate basic metrics for each valid feature
        for feature, iterations in feature_results.items():
            if not iterations:  # Skip features with no results
                continue

            # Extract scores across iterations
            importance_scores = [it['importance'].get('combined_score', 0.0) for it in iterations]
            stability_scores = [it['stability'].get('sign_consistency', 0.0) for it in iterations]
            
            # Get model effects for CV and range calculations
            model_effects = [it['importance'].get('model_effect_mean', 0.0) for it in iterations]
            model_effects = np.array([x for x in model_effects if not np.isnan(x)])
            
            if len(model_effects) > 0:
                effect_cv = np.std(model_effects) / max(abs(np.mean(model_effects)), 1e-10)
                effect_range = (np.max(model_effects) - np.min(model_effects)) / max(abs(np.mean(model_effects)), 1e-10)
            else:
                effect_cv = 0.0
                effect_range = 0.0
            
            # Calculate basic metrics
            aggregated_results[feature] = {
                'mean_importance': np.mean(importance_scores),
                'std_importance': np.std(importance_scores),
                'mean_stability': np.mean(stability_scores),
                'std_stability': np.std(stability_scores),
                'effect_cv': effect_cv,
                'relative_range': effect_range,
                'selection_frequency': sum(
                    s >= self.stability_threshold and i >= self.importance_threshold
                    for s, i in zip(stability_scores, importance_scores)
                ) / len(iterations),
                'is_problematic': feature in diagnostics.get('problematic_features', set())
            }
            
            # Calculate CV and range for each feature from raw effects
            feature_effects = [it['importance'].get('model_effect_mean', 0.0) for it in iterations]
            feature_effects = [x for x in feature_effects if not np.isnan(x)]  # Remove NaNs
            
            if feature_effects:
                effect_mean = np.mean(feature_effects)
                effect_std = np.std(feature_effects)
                effect_cv = effect_std / max(abs(effect_mean), 1e-10)
                relative_range = (max(feature_effects) - min(feature_effects)) / max(abs(effect_mean), 1e-10)
            else:
                effect_cv = 0.0
                relative_range = 0.0

            # Store importance metrics
            importance_metrics[feature] = {
                'combined_score': aggregated_results[feature]['mean_importance'],
                'model_effect_mean': np.mean([it['importance'].get('model_effect_mean', 0.0) for it in iterations]),
                'model_effect_std': np.mean([it['importance'].get('model_effect_std', 0.0) for it in iterations]),
                'correlation': np.mean([it['importance'].get('correlation', 0.0) for it in iterations]),
                'mutual_info': np.mean([it['importance'].get('mutual_info', 0.0) for it in iterations]),
                'effect_cv': effect_cv,
                'relative_range': relative_range
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
                # Extract base features in consistent order, converting None to NaN
                feat1 = [np.nan if pref.features1[f] is None else float(pref.features1[f]) 
                        for f in base_features]
                feat2 = [np.nan if pref.features2[f] is None else float(pref.features2[f]) 
                        for f in base_features]
                
                X1.append(feat1)
                X2.append(feat2)
                y.append(float(pref.preferred))
                
            except KeyError as e:
                logger.error(f"Missing feature in preference: {str(e)}")
                logger.error(f"Available features1: {list(pref.features1.keys())}")
                logger.error(f"Available features2: {list(pref.features2.keys())}")
                raise
        
        X1 = np.array(X1, dtype=np.float64)
        X2 = np.array(X2, dtype=np.float64)
        y = np.array(y)
        
        # Add interaction features
        logger.debug(f"\nAdding {len(interactions)} interaction features...")
        
        for interaction in interactions:
            if all(f in base_features for f in interaction):
                # Get indices for the interaction features
                indices = [base_features.index(f) for f in interaction]
                interaction_name = '_x_'.join(interaction)
                
                # Show feature values before multiplication
                logger.debug(f"\nCreating interaction: {interaction_name}")
                for idx, feat in zip(indices, interaction):
                    logger.debug(f"Feature {feat} values for first 5 bigram pairs:")
                    logger.debug(f"  First bigram:  {X1[:5, idx]}")  # First 5 bigrams of pair
                    logger.debug(f"  Second bigram: {X2[:5, idx]}")  # Their comparison bigrams
                
                # Calculate interaction values
                interaction1 = X1[:, indices[0]].copy()
                interaction2 = X2[:, indices[0]].copy()
                
                for idx in indices[1:]:
                    interaction1 *= X1[:, idx]
                    interaction2 *= X2[:, idx]
                
                # Show interaction results
                logger.debug("Resulting interaction for first 5 bigram pairs:")
                logger.debug(f"  First bigram:  {interaction1[:5]}")
                logger.debug(f"  Second bigram: {interaction2[:5]}")
                
                # Add interaction features to matrices
                X1 = np.column_stack([X1, interaction1])
                X2 = np.column_stack([X2, interaction2])
                all_features.append(interaction_name)
                                
                logger.debug(f"Added interaction feature: {interaction_name}")
        
        # Validate all features (base + interactions)
        logger.debug("\nFeature matrix validation:")
        logger.debug(f"Final matrix shapes - X1: {X1.shape}, X2: {X2.shape}, y: {y.shape}")
        logger.debug(f"Total features: {len(all_features)}")
        
        # Validate features before processing
        valid_features = []
        problematic_features = set()

        # Check for invalid values in each feature
        for i, feature in enumerate(all_features):
            diff = X1[:, i] - X2[:, i]
            n_valid = np.sum(~np.isnan(diff))  # Count non-NaN values
            n_total = len(diff)
            valid_ratio = n_valid / n_total
            n_unique = len(np.unique(diff[~np.isnan(diff)]))  # Count unique non-NaN values
            n_inf = np.sum(np.isinf(diff))
            
            is_interaction = feature not in base_features
            feature_type = "Interaction" if is_interaction else "Base"
            
            logger.debug(f"\n{feature_type} feature '{feature}':")
            logger.debug(f"  Valid values: {n_valid} of {n_total} ({valid_ratio*100:.1f}%)")
            logger.debug(f"  Unique values: {n_unique}")
            logger.debug(f"  Range: [{np.nanmin(diff)}, {np.nanmax(diff)}]")
            
            # Validate and warn
            if valid_ratio < 0.5:  # More than 50% missing
                logger.warning(f"  WARNING: {feature_type} feature '{feature}' has {n_total - n_valid} NaN values "
                             f"({(1-valid_ratio)*100:.1f}%) - will be excluded")
                problematic_features.add(feature)
            elif n_unique <= 1:
                logger.warning(f"  WARNING: {feature_type} feature '{feature}' has constant values - will be excluded")
                problematic_features.add(feature)
            elif n_inf > 0:
                logger.warning(f"  WARNING: {feature_type} feature '{feature}' has {n_inf} infinite values - will be excluded")
                problematic_features.add(feature)
            else:
                valid_features.append(feature)

        return X1, X2, y, valid_features
             
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
                    logger.debug(f"Raw correlation for {feature}: {corr}")
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
                logger.debug(f"Raw MI score for {feature}: {mi_score}")
                if np.isnan(mi_score):
                    mi_score = 0.0
                    importance[feature]['diagnostics']['issues'].append(
                        'MI calculation failed')
            except Exception as e:
                logger.warning(f"MI calculation failed for {feature}: {str(e)}")
                mi_score = 0.0
                importance[feature]['diagnostics']['issues'].append(
                    'MI error')
            
            # Add debug logging to see what's in model_results
            logger.debug(f"Model results keys: {model_results.keys()}")
            logger.debug(f"Feature effects: {model_results.get('feature_effects', {})}")

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
                                
                # Calculate differences
                diff = X1[:, i] - X2[:, i]
                valid_mask = ~(np.isnan(diff) | np.isinf(diff))
                valid_diffs = diff[valid_mask]
                
                stats = {
                    'n_total': len(diff),
                    'n_valid': len(valid_diffs),
                    'n_unique': len(np.unique(valid_diffs)) if len(valid_diffs) > 0 else 0,
                    'n_nan': np.sum(np.isnan(diff)),  # Add this
                    'n_inf': np.sum(np.isinf(diff)),  # Add this
                    'n_zero': np.sum(diff == 0),      # Add this
                    'mean': np.mean(valid_diffs) if len(valid_diffs) > 0 else np.nan,
                    'std': np.std(valid_diffs) if len(valid_diffs) > 0 else np.nan,
                    'min': np.min(valid_diffs) if len(valid_diffs) > 0 else np.nan,
                    'max': np.max(valid_diffs) if len(valid_diffs) > 0 else np.nan
                }

                # Feature difference logs
                logger.debug(f"\nFeature '{feature}' differences (bigram1 - bigram2):")
                logger.debug(f"  Total pairs analyzed: {stats['n_total']}")
                logger.debug(f"  Valid pairs: {stats['n_valid']}")
                logger.debug(f"  Unique difference values: {stats['n_unique']}")
                if stats['n_unique'] < 5:  # Show actual values if few
                    unique_vals = np.unique(valid_diffs)
                    logger.debug(f"  Unique differences found: {unique_vals}")
                logger.debug(f"  Range of differences: [{stats['min']:.3f}, {stats['max']:.3f}]")
                
                # Feature-specific logs
                if feature == 'typing_time':
                    logger.debug("    Negative means bigram2 was typed faster")
                elif feature in ['rows_apart', 'angle_apart']:
                    logger.debug("    Negative means bigram2 has larger distance/angle")
                elif feature == 'outward_roll':
                    logger.debug("    Negative means bigram2 has more outward roll")
                elif feature in ['sum_finger_values', 'sum_engram_position_values', 'sum_row_position_values']:
                    logger.debug("    Negative means bigram2 has higher position values")
                else:
                    logger.debug("    Negative means bigram2 has larger value")
                    
                logger.debug(f"  Mean ± std: {stats['mean']:.3f} ± {stats['std']:.3f}")

                diagnostics['feature_stats'][feature] = stats
                                
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
     
    def _recommend_existing_features(
        self,
        importance: Dict[str, Dict[str, float]],
        stability: Dict[str, Dict[str, float]],
        correlations: pd.DataFrame,
        correlation_diagnostics: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Recommend existing features."""
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
    
    def _save_detailed_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save comprehensive feature selection report in both text and CSV formats."""
        
        # Save text report
        report_file = output_dir / "feature_selection_metrics.txt"
        with open(report_file, 'w') as f:
            # Summary section
            f.write("Summary\n")
            f.write("---------\n")
            f.write(f"Total features evaluated: {len(results['importance'])}\n")
            f.write(f"Features selected: {len(results['selected_features'])}\n")
            f.write(f"Features rejected: {len(set(results['importance'].keys()) - set(results['selected_features']))}\n")
            f.write(f"Features flagged: {len(results.get('flagged_features', []))}\n\n")
            
            # Selected Features section
            f.write("Selected Features\n")
            f.write("----------------\n")

            # Write data for valid features only
            for feature in sorted(results['importance'].keys()):  # These are already valid features
                imp = results['importance'].get(feature, {})
                stab = results['stability'].get(feature, {})
                is_interaction = '_x_' in feature
                status = "selected" if feature in results['selected_features'] else "rejected"
                if '_x_' not in feature:  # Base features only
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

        # Save CSV report
        csv_file = output_dir / "feature_selection_metrics.csv"
        with open(csv_file, 'w') as f:
            # Write configuration as comments
            f.write("# Feature selection configuration:\n")
            f.write(f"# n_samples: {self.config['model']['n_samples']}\n")
            f.write(f"# n_splits: {self.config['model']['cross_validation']['n_splits']}\n")
            f.write(f"# random_seed: {self.config['model']['cross_validation']['random_seed']}\n")
            f.write(f"# importance_threshold: {self.config['feature_evaluation']['thresholds']['importance']}\n\n")
            
            # Write header
            f.write("feature_name,is_interaction,status,importance,stability,")
            f.write("model_effect_mean,model_effect_std,correlation,")
            f.write("mutual_information,effect_cv,relative_range\n")
            
            # Write data
            for feature in sorted(results['importance'].keys()):
                imp = results['importance'].get(feature, {})
                stab = results['stability'].get(feature, {})
                is_interaction = '_x_' in feature
                status = "selected" if feature in results['selected_features'] else "rejected"
                
                values = [
                    feature,
                    str(is_interaction).lower(),
                    status,
                    f"{imp.get('combined_score', 0.0):.3f}",
                    f"{stab.get('sign_consistency', 0.0):.3f}",
                    f"{imp.get('model_effect_mean', 0.0):.3f}",
                    f"{imp.get('model_effect_std', 0.0):.3f}",
                    f"{imp.get('correlation', 0.0):.3f}",
                    f"{imp.get('mutual_info', 0.0):.3f}",
                    f"{stab.get('effect_cv', 0.0):.3f}",
                    f"{stab.get('relative_range', 0.0):.3f}"
                ]
                f.write(','.join(values) + '\n')

        logger.info(f"Saved feature selection report to {report_file}")
        logger.info(f"Saved feature metrics to {csv_file}")

                                        
        
