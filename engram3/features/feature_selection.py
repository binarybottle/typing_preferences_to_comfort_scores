# engram3/features/feature_selection.py
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import mutual_info_score
import logging
from pathlib import Path
from collections import defaultdict
import json
import warnings

from engram3.data import PreferenceDataset
from engram3.models.bayesian import BayesianPreferenceModel

logger = logging.getLogger(__name__)

class FeatureEvaluator:
    """Comprehensive feature evaluation for preference learning."""
    
    def __init__(self, 
                 importance_threshold: float = 0.1,
                 stability_threshold: float = 0.7,
                 correlation_threshold: float = 0.7):
        """
        Initialize feature evaluator.
        
        Args:
            importance_threshold: Minimum importance score to keep feature
            stability_threshold: Minimum stability score to keep feature
            correlation_threshold: Threshold for identifying highly correlated features
        """
        self.importance_threshold = importance_threshold
        self.stability_threshold = stability_threshold
        self.correlation_threshold = correlation_threshold
        
        # Configure warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def run_feature_selection(
            self,
            dataset: PreferenceDataset,
            n_repetitions: int = 10,
            output_dir: Optional[Path] = None
        ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run feature selection multiple times to get stable feature recommendations.
        
        Args:
            dataset: PreferenceDataset containing preferences
            n_repetitions: Number of times to repeat feature evaluation
            output_dir: Optional directory to save results
            
        Returns:
            Tuple of:
            - List of recommended features
            - Dictionary of diagnostics
        """        
        # Log initial dataset info
        feature_names = dataset.get_feature_names()
        logger.info(f"\nStarting feature selection:")

        # Collect results and diagnostics across repetitions
        feature_results = defaultdict(list)
        all_diagnostics = {
            'importance_diagnostics': [],
            'correlation_diagnostics': [],
            'problematic_features': set(),
            'iteration_metrics': []
        }
                    
        for iter_idx in range(n_repetitions):
            logger.debug(f"Feature selection iteration {iter_idx+1}/{n_repetitions}")
            
            # Prepare feature matrices
            X1, X2, y = self._prepare_feature_matrices(dataset)
            feature_names = dataset.get_feature_names()
            
            # Calculate correlations once
            correlations, correlation_diagnostics = self._calculate_feature_correlations(
                X1, X2, feature_names)
            
            # Run cross-validation
            model = BayesianPreferenceModel()
            cv_results = model.cross_validate(dataset)  # Store the results
            
            # Evaluate features using pre-calculated correlations
            eval_results = self.evaluate_features(
                dataset, 
                cv_results,  # Pass the stored results
                output_dir=output_dir,
                feature_matrices=(X1, X2, y),
                correlation_results=(correlations, correlation_diagnostics)
            )
            
            # Store basic results
            for feature in feature_names:
                feature_results[feature].append({
                    'importance': eval_results['importance'][feature],
                    'stability': eval_results['stability'][feature]
                })
            
            # Store diagnostics
            all_diagnostics['importance_diagnostics'].append(
                eval_results.get('diagnostics', {}))
            all_diagnostics['correlation_diagnostics'].append(correlation_diagnostics)
            
            # Track problematic features
            for feature, diag in eval_results.get('diagnostics', {}).items():
                if (diag.get('n_unique_differences', 0) == 1 or
                    diag.get('n_nan', 0) > 0 or
                    diag.get('n_inf', 0) > 0):
                    all_diagnostics['problematic_features'].add(feature)
            
            # Store iteration metrics
            all_diagnostics['iteration_metrics'].append({
                'iteration': iter_idx + 1,
                'n_features': len(feature_names),
                'n_problematic': len(all_diagnostics['problematic_features']),
                'correlation_issues': len(correlation_diagnostics.get('correlation_issues', [])),
                'highly_correlated_pairs': len(correlation_diagnostics.get('highly_correlated_pairs', []))
            })
        
        # Aggregate results and make final recommendations
        final_recommendations = self._analyze_selection_results(
            feature_results, all_diagnostics)
        
        # Save results and diagnostics if output directory provided
        if output_dir:
            self._save_diagnostic_report(all_diagnostics, output_dir)
            self._save_detailed_report({
                'selected_features': final_recommendations['selected_features'],
                'rejected_features': final_recommendations.get('rejected_features', []),
                'all_features': feature_names,
                'importance': eval_results.get('importance', {}),
                'stability': eval_results.get('stability', {})
            }, output_dir)
        
        # Final summary only
        logger.info("\nFeature Selection Results:")
        logger.info(f"- Selected {len(final_recommendations['selected_features'])} features")
        if final_recommendations['selected_features']:
            logger.info("Selected features:")
            for feature in final_recommendations['selected_features']:
                logger.info(f"  - {feature}")
        logger.info(f"Results saved to {output_dir}")

        return (final_recommendations['selected_features'], all_diagnostics)
    
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
    
        Args:
            dataset: PreferenceDataset containing all preferences
            model_results: Results from model cross-validation
            output_dir: Optional directory to save detailed reports
            feature_matrices: Optional tuple of (X1, X2, y) matrices
            correlation_results: Optional tuple of (correlations, diagnostics)
        """
        feature_names = dataset.get_feature_names()
        
        # Use provided matrices if available, otherwise calculate them
        if feature_matrices is not None:
            X1, X2, y = feature_matrices
        else:
            X1, X2, y = self._prepare_feature_matrices(dataset)
        
        # Calculate feature importance
        importance = self._calculate_feature_importance(
            X1, X2, y, feature_names, model_results)
        
        # Use provided correlation results or calculate new ones
        if correlation_results is not None:
            correlations, corr_diagnostics = correlation_results
        else:
            correlations, corr_diagnostics = self._calculate_feature_correlations(
                X1, X2, feature_names)
            
        # Get stability metrics from model results
        stability = model_results['stability']
            
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
            'selected_features': recommendations['strong_features'],  # Add this line
            'rejected_features': recommendations.get('weak_features', []),  # And this line
            'all_features': feature_names
        }
        
        # Save detailed report if output_dir provided
        if output_dir:
            self._save_detailed_report(results, output_dir)
            
        return results
    
    def _analyze_selection_results(
        self,
        feature_results: Dict[str, List[Dict]],
        diagnostics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze feature selection results across iterations with diagnostics.
        
        Args:
            feature_results: Dictionary mapping features to their evaluation results
            diagnostics: Dictionary containing diagnostic information
            
        Returns:
            Dictionary containing:
            - selected_features: List of recommended features
            - aggregated_metrics: Detailed metrics for each feature
            - selection_diagnostics: Diagnostic information about selection process
        """
        logger = logging.getLogger(__name__)
        aggregated_results = {}
        
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
                'is_problematic': feature in diagnostics['problematic_features']
            }
            
            # Add correlation information
            high_correlations = []
            for corr_diag in diagnostics['correlation_diagnostics']:
                for pair in corr_diag.get('highly_correlated_pairs', []):
                    if feature in (pair['feature1'], pair['feature2']):
                        other_feature = pair['feature2'] if feature == pair['feature1'] else pair['feature1']
                        high_correlations.append((other_feature, pair['correlation']))
            
            if high_correlations:
                aggregated_results[feature]['high_correlations'] = high_correlations
        
        # Second pass: Make selection decisions
        selected_features = []
        rejected_features = []
        flagged_features = []
        
        for feature, metrics in aggregated_results.items():
            # Skip problematic features
            if metrics['is_problematic']:
                rejected_features.append((feature, 'problematic feature'))
                continue
            
            # Calculate selection criteria
            meets_importance = metrics['mean_importance'] >= self.importance_threshold
            meets_stability = metrics['mean_stability'] >= self.stability_threshold
            is_consistent = metrics['selection_frequency'] >= 0.7  # Selected in 70% of iterations
            has_high_variance = metrics['std_importance'] > 0.5 * metrics['mean_importance']
            
            # Make selection decision
            if meets_importance and meets_stability and is_consistent:
                if has_high_variance:
                    flagged_features.append((feature, 'high variance'))
                else:
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
        
        # Handle correlated features
        correlated_groups = []
        used_features = set()
        
        # Group highly correlated features
        for feature in selected_features:
            if feature in used_features:
                continue
                
            if 'high_correlations' in aggregated_results[feature]:
                group = [feature]
                used_features.add(feature)
                
                for other_feat, corr in aggregated_results[feature]['high_correlations']:
                    if other_feat in selected_features and other_feat not in used_features:
                        group.append(other_feat)
                        used_features.add(other_feat)
                        
                if len(group) > 1:
                    correlated_groups.append(group)
        
        # Prepare detailed selection report
        selection_diagnostics = {
            'n_iterations': len(next(iter(feature_results.values()))),
            'n_total_features': len(feature_results),
            'n_selected': len(selected_features),
            'n_rejected': len(rejected_features),
            'n_flagged': len(flagged_features),
            'correlated_groups': correlated_groups,
            'rejected_features': rejected_features,
            'flagged_features': flagged_features
        }
        
        # Log selection results
        logger.info(f"\nFeature selection results:")
        logger.info(f"- Selected {len(selected_features)} features")
        logger.info(f"- Rejected {len(rejected_features)} features")
        logger.info(f"- Flagged {len(flagged_features)} features")
        logger.info(f"- Found {len(correlated_groups)} groups of correlated features")
        
        if flagged_features:
            logger.warning("Flagged features (selected but require attention):")
            for feature, reason in flagged_features:
                logger.warning(f"  - {feature}: {reason}")
        
        if correlated_groups:
            logger.info("\nCorrelated feature groups:")
            for group in correlated_groups:
                logger.info(f"  - Group: {', '.join(group)}")
        
        return {
            'selected_features': selected_features,
            'aggregated_metrics': aggregated_results,
            'selection_diagnostics': selection_diagnostics
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
        dataset: PreferenceDataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrices with detailed validation."""
        feature_names = dataset.get_feature_names()
        logger.debug("\nDEBUG: Starting feature matrix preparation")   
        
        X1 = []  # First bigram features
        X2 = []  # Second bigram features
        y = []   # Preferences
        
        logger.debug("Preparing feature matrices...")   
        logger.debug(f"Number of features: {len(feature_names)}")   
        logger.debug(f"Features: {feature_names}")   
        
        for pref in dataset.preferences:
            # Log first few preferences for debugging
            if len(X1) < 5:
                logger.debug(f"\nPreference {len(X1)+1}:")   
                logger.debug(f"Bigram1: {pref.bigram1}")   
                logger.debug(f"Bigram2: {pref.bigram2}")   
                logger.debug(f"Preferred: {pref.preferred}")   
                logger.debug("Features 1:")   
                for fname, fval in pref.features1.items():
                    logger.debug(f"  {fname}: {fval}")   
                logger.debug("Features 2:")   
                for fname, fval in pref.features2.items():
                    logger.debug(f"  {fname}: {fval}")   
            
            try:
                # Extract features in consistent order
                feat1 = [pref.features1[f] for f in feature_names]
                feat2 = [pref.features2[f] for f in feature_names]
                
                X1.append(feat1)
                X2.append(feat2)
                y.append(float(pref.preferred))
                
            except KeyError as e:
                logger.error(f"Missing feature in preference: {str(e)}")  # Keep as error
                logger.error(f"Available features1: {list(pref.features1.keys())}")  # Keep as error
                logger.error(f"Available features2: {list(pref.features2.keys())}")  # Keep as error
                raise
            
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)
        
        # Validate matrices
        logger.debug("\nFeature matrix validation:")   
        logger.debug(f"X1 shape: {X1.shape}")   
        logger.debug(f"X2 shape: {X2.shape}")   
        logger.debug(f"y shape: {y.shape}")   
        
        # Check for invalid values in each feature
        for i, feature in enumerate(feature_names):
            n_nan1 = np.sum(np.isnan(X1[:, i]))
            n_nan2 = np.sum(np.isnan(X2[:, i]))
            n_inf1 = np.sum(np.isinf(X1[:, i]))
            n_inf2 = np.sum(np.isinf(X2[:, i]))
            n_unique1 = len(np.unique(X1[:, i]))
            n_unique2 = len(np.unique(X2[:, i]))
            
            logger.debug(f"\nFeature '{feature}':")   
            logger.debug(f"  X1: {n_unique1} unique values, {n_nan1} NaN, {n_inf1} Inf")   
            logger.debug(f"  X2: {n_unique2} unique values, {n_nan2} NaN, {n_inf2} Inf")   
            logger.debug(f"  Range X1: [{np.min(X1[:, i])}, {np.max(X1[:, i])}]")   
            logger.debug(f"  Range X2: [{np.min(X2[:, i])}, {np.max(X2[:, i])}]")   
            
            # Keep these as warnings
            if n_unique1 == 1 or n_unique2 == 1:
                logger.warning(f"  WARNING: Feature '{feature}' has constant values")
            if n_nan1 > 0 or n_nan2 > 0:
                logger.warning(f"  WARNING: Feature '{feature}' has NaN values")
            if n_inf1 > 0 or n_inf2 > 0:
                logger.warning(f"  WARNING: Feature '{feature}' has Inf values")
        
        return X1, X2, y
     
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
                    corr, _ = stats.spearmanr(valid_diff, valid_y)
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
                                correlation_result = scipy.stats.spearmanr(diff_i, diff_j)
                                corr = correlation_result.correlation if not np.isnan(correlation_result.correlation) else 0.0
                                if np.isnan(corr):
                                    corr = 0.0
                                    diagnostics['correlation_issues'].append({
                                        'feature1': feature_names[i],
                                        'feature2': feature_names[j],
                                        'reason': 'correlation calculation failed'
                                    })
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
    
    def _save_diagnostic_report(
        self,
        diagnostics: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Save detailed diagnostic report."""
        report_file = output_dir / "feature_selection_diagnostics.txt"
        
        with open(report_file, 'w') as f:
            f.write("Feature Selection Diagnostic Report\n")
            f.write("================================\n\n")
            
            # Write problematic features section
            f.write("1. Problematic Features\n")
            f.write("--------------------\n")
            for feature in sorted(diagnostics['problematic_features']):
                f.write(f"\n{feature}:\n")
                # Add statistics from importance diagnostics
                for iter_diag in diagnostics['importance_diagnostics']:
                    if feature in iter_diag:
                        f.write(f"  Diagnostics:\n")
                        for k, v in iter_diag[feature].items():
                            f.write(f"    {k}: {v}\n")
            
            # Write correlation issues section
            f.write("\n2. Correlation Issues\n")
            f.write("------------------\n")
            for iter_idx, corr_diag in enumerate(diagnostics['correlation_diagnostics']):
                f.write(f"\nIteration {iter_idx + 1}:\n")
                for issue in corr_diag.get('correlation_issues', []):
                    f.write(f"  {issue['feature1']} ↔ {issue['feature2']}: {issue['reason']}\n")
            
            # Write iteration metrics summary
            f.write("\n3. Iteration Metrics\n")
            f.write("-----------------\n")
            for metrics in diagnostics['iteration_metrics']:
                f.write(f"\nIteration {metrics['iteration']}:\n")
                for k, v in metrics.items():
                    if k != 'iteration':
                        f.write(f"  {k}: {v}\n")
    
    def _save_detailed_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save detailed report of final feature selection results."""
        report_file = output_dir / "feature_selection_results.txt"
        
        with open(report_file, 'w') as f:
            f.write("Feature Selection Results Report\n")
            f.write("============================\n\n")
            
            # 1. Feature Summary
            f.write("1. Feature Analysis Summary\n")
            f.write("-----------------------\n")
            
            # Group features by characteristics
            features_by_variance = {
                'High variance': [],
                'Moderate variance': [],
                'Low variance': []
            }
            
            for feature in results.get('all_features', []):
                if feature in results.get('selected_features', []):
                    status = "SELECTED"
                else:
                    status = "REJECTED"
                    
                imp = results['importance'].get(feature, {})
                stab = results['stability'].get(feature, {})
                
                f.write(f"\n{feature} ({status}):\n")
                f.write(f"  Importance score: {imp.get('combined_score', 0):.3f}\n")
                f.write(f"  Model effect: {imp.get('model_effect_mean', 0):.3f} ± {imp.get('model_effect_std', 0):.3f}\n")
                f.write(f"  Stability: {stab.get('sign_consistency', 0):.3f}\n")
                f.write(f"  Correlation: {imp.get('correlation', 0):.3f}\n")
                f.write(f"  Mutual information: {imp.get('mutual_info', 0):.3f}\n")
            
            # 2. Selection Results
            f.write("\n2. Selection Results\n")
            f.write("------------------\n")
            f.write(f"Total features evaluated: {len(results.get('all_features', []))}\n")
            f.write(f"Features selected: {len(results.get('selected_features', []))}\n")
            f.write(f"Features rejected: {len(results.get('rejected_features', []))}\n")
            
            # 3. Recommendations
            f.write("\n3. Recommendations\n")
            f.write("----------------\n")
            if not results.get('selected_features'):
                f.write("\nNo features met the selection criteria. Consider:\n")
                f.write("1. Lowering the importance threshold\n")
                f.write("2. Lowering the stability threshold\n")
                f.write("3. Creating composite features\n")
                f.write("4. Collecting more data\n")
                
