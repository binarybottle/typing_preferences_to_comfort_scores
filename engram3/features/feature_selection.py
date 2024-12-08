# engram3/features/feature_selection.py

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
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
        # Collect results and diagnostics across repetitions
        feature_results = defaultdict(list)
        all_diagnostics = {
            'importance_diagnostics': [],
            'correlation_diagnostics': [],
            'problematic_features': set(),
            'iteration_metrics': []
        }
        
        logger.info(f"Starting feature selection with {n_repetitions} repetitions")
        
        for i in range(n_repetitions):
            logger.info(f"Feature selection iteration {i+1}/{n_repetitions}")
            
            # Prepare feature matrices for diagnostics
            X1, X2, y = self._prepare_feature_matrices(dataset)
            feature_names = dataset.get_feature_names()
            
            # Calculate correlations with diagnostics
            corr_matrix, corr_diagnostics = self._calculate_feature_correlations(
                X1, X2, feature_names)
            
            # Run cross-validation and evaluation
            model = BayesianPreferenceModel()
            cv_results = model.cross_validate(dataset)
            
            # Evaluate features with diagnostics
            eval_results = self.evaluate_features(dataset, cv_results)
            
            # Store basic results
            for feature in feature_names:
                feature_results[feature].append({
                    'importance': eval_results['importance'][feature],
                    'stability': eval_results['stability'][feature]
                })
            
            # Store diagnostics
            all_diagnostics['importance_diagnostics'].append(
                eval_results.get('diagnostics', {}))
            all_diagnostics['correlation_diagnostics'].append(corr_diagnostics)
            
            # Track problematic features
            for feature, diag in eval_results.get('diagnostics', {}).items():
                if (diag.get('n_unique_differences', 0) == 1 or
                    diag.get('n_nan', 0) > 0 or
                    diag.get('n_inf', 0) > 0):
                    all_diagnostics['problematic_features'].add(feature)
            
            # Store iteration metrics
            all_diagnostics['iteration_metrics'].append({
                'iteration': i + 1,
                'n_features': len(feature_names),
                'n_problematic': len(all_diagnostics['problematic_features']),
                'correlation_issues': len(corr_diagnostics.get('correlation_issues', [])),
                'highly_correlated_pairs': len(corr_diagnostics.get('highly_correlated_pairs', []))
            })
        
        # Aggregate results and make final recommendations
        final_recommendations = self._analyze_selection_results(
            feature_results, all_diagnostics)
        
        # Save results and diagnostics if output directory provided
        if output_dir:
            # Save iteration-by-iteration diagnostics
            self._save_diagnostic_report(all_diagnostics, output_dir)
            
            # Save final results
            self._save_detailed_report({
                'selected_features': final_recommendations['selected_features'],
                'rejected_features': final_recommendations.get('rejected_features', []),
                'all_features': feature_names,
                'importance': eval_results.get('importance', {}),
                'stability': eval_results.get('stability', {})
            }, output_dir)
            
        
        # Log summary
        logger.info(f"Feature selection completed:")
        logger.info(f"- Selected {len(final_recommendations['selected_features'])} features")
        logger.info(f"- Found {len(all_diagnostics['problematic_features'])} problematic features")
        logger.info(f"- Average correlation issues per iteration: "
                f"{np.mean([m['correlation_issues'] for m in all_diagnostics['iteration_metrics']]):.1f}")
        
        return (final_recommendations['selected_features'], 
                all_diagnostics)

    def evaluate_features(
        self,
        dataset: PreferenceDataset,
        model_results: Dict[str, Dict],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Dict]:
        """
        Comprehensive feature evaluation.
        
        Args:
            dataset: PreferenceDataset containing all preferences
            model_results: Results from model cross-validation
            output_dir: Optional directory to save detailed reports
            
        Returns:
            Dictionary containing:
            - importance: Feature importance scores
            - stability: Feature stability metrics
            - correlations: Feature correlation matrix
            - recommendations: Feature recommendations
        """
        feature_names = dataset.get_feature_names()
        
        # Get feature matrices for analysis
        X1, X2, y = self._prepare_feature_matrices(dataset)
        
        # Calculate feature importance
        importance = self._calculate_feature_importance(
            X1, X2, y, feature_names, model_results)
            
        # Calculate feature correlations
        correlations, corr_diagnostics = self._calculate_feature_correlations(
            X1, X2, feature_names)
            
        # Get stability metrics from model results
        stability = model_results['stability']
            
        # Generate recommendations
        recommendations = self._generate_recommendations(
            importance, stability, correlations, corr_diagnostics)
            
        results = {
            'importance': importance,
            'stability': stability,
            'correlations': correlations,
            'correlation_diagnostics': corr_diagnostics,
            'recommendations': recommendations
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
        results: Dict,
        output_dir: Path
    ) -> None:
        """Save feature selection results."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        with open(output_dir / 'feature_selection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save detailed report
        with open(output_dir / 'feature_selection_report.txt', 'w') as f:
            f.write("Feature Selection Report\n")
            f.write("=====================\n\n")
            
            f.write("Selected Features:\n")
            f.write("-----------------\n")
            for feature in results['selected_features']:
                metrics = results['aggregated_metrics'][feature]
                f.write(f"\n{feature}:\n")
                f.write(f"  Mean importance: {metrics['mean_importance']:.3f} ± {metrics['std_importance']:.3f}\n")
                f.write(f"  Mean stability: {metrics['mean_stability']:.3f} ± {metrics['std_stability']:.3f}\n")
                f.write(f"  Selection frequency: {metrics['selection_frequency']:.1%}\n")
            
            f.write("\nRejected Features:\n")
            f.write("----------------\n")
            rejected = set(results['aggregated_metrics'].keys()) - set(results['selected_features'])
            for feature in rejected:
                metrics = results['aggregated_metrics'][feature]
                f.write(f"\n{feature}:\n")
                f.write(f"  Mean importance: {metrics['mean_importance']:.3f} ± {metrics['std_importance']:.3f}\n")
                f.write(f"  Mean stability: {metrics['mean_stability']:.3f} ± {metrics['std_stability']:.3f}\n")
                f.write(f"  Selection frequency: {metrics['selection_frequency']:.1%}\n")
        
    def _prepare_feature_matrices(
        self,
        dataset: PreferenceDataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrices for analysis."""
        feature_names = dataset.get_feature_names()
        
        X1 = []  # First bigram features
        X2 = []  # Second bigram features
        y = []   # Preferences
        
        for pref in dataset.preferences:
            feat1 = [pref.features1[f] for f in feature_names]
            feat2 = [pref.features2[f] for f in feature_names]
            X1.append(feat1)
            X2.append(feat2)
            y.append(float(pref.preferred))
            
        return np.array(X1), np.array(X2), np.array(y)
        
    def _calculate_feature_importance(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_results: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate feature importance using multiple metrics with detailed diagnostics.
        
        Args:
            X1: Feature matrix for first bigram in each pair
            X2: Feature matrix for second bigram in each pair
            y: Binary preference indicators
            feature_names: List of feature names
            model_results: Results from model cross-validation
            
        Returns:
            Dictionary mapping features to their importance metrics
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            importance = {}
            logger = logging.getLogger(__name__)
            logger.info("Starting feature importance calculation...")
            
            for i, feature in enumerate(feature_names):
                logger.debug(f"Processing feature: {feature}")
                
                # Calculate feature differences
                feat_diff = X1[:, i] - X2[:, i]
                
                # Diagnostic checks
                n_unique = len(np.unique(feat_diff))
                n_nan = np.sum(np.isnan(feat_diff))
                n_inf = np.sum(np.isinf(feat_diff))
                n_zero = np.sum(feat_diff == 0)
                
                if n_unique == 1:
                    logger.warning(f"Feature '{feature}' has constant differences (value: {feat_diff[0]})")
                    corr = 0.0
                    mi_score = 0.0
                elif n_nan > 0:
                    logger.warning(f"Feature '{feature}' has {n_nan} NaN differences")
                    corr = 0.0
                    mi_score = 0.0
                elif n_inf > 0:
                    logger.warning(f"Feature '{feature}' has {n_inf} infinite differences")
                    corr = 0.0
                    mi_score = 0.0
                elif n_zero == len(feat_diff):
                    logger.warning(f"Feature '{feature}' has all zero differences")
                    corr = 0.0
                    mi_score = 0.0
                else:
                    # Calculate correlation with proper error handling
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            corr, _ = stats.spearmanr(feat_diff, y)
                            if np.isnan(corr):
                                logger.warning(f"Feature '{feature}' produced NaN correlation")
                                corr = 0.0
                    except Exception as e:
                        logger.error(f"Error calculating correlation for feature '{feature}': {str(e)}")
                        corr = 0.0
                    
                    # Calculate mutual information with proper error handling
                    try:
                        # Binarize feature differences for MI calculation
                        feat_diff_bin = feat_diff > np.median(feat_diff[~np.isnan(feat_diff)])
                        mi_score = mutual_info_score(feat_diff_bin, y)
                        if np.isnan(mi_score):
                            logger.warning(f"Feature '{feature}' produced NaN mutual information")
                            mi_score = 0.0
                    except Exception as e:
                        logger.error(f"Error calculating mutual information for feature '{feature}': {str(e)}")
                        mi_score = 0.0
                
                # Get model-based importance
                model_effect = model_results['feature_effects'].get(feature, {})
                effect_mean = model_effect.get('mean', 0.0)
                effect_std = model_effect.get('std', 0.0)
                
                # Store results with diagnostics
                importance[feature] = {
                    'mutual_info': float(mi_score),
                    'correlation': float(corr),
                    'model_effect_mean': float(effect_mean),
                    'model_effect_std': float(effect_std),
                    'diagnostics': {
                        'n_unique_differences': int(n_unique),
                        'n_nan': int(n_nan),
                        'n_inf': int(n_inf),
                        'n_zero': int(n_zero),
                        'total_samples': len(feat_diff)
                    }
                }
                
                # Calculate combined score only if feature has valid metrics
                if n_unique > 1 and n_nan == 0 and n_inf == 0:
                    importance[feature]['combined_score'] = float(
                        0.4 * abs(effect_mean) +
                        0.3 * abs(corr) +
                        0.3 * mi_score
                    )
                else:
                    importance[feature]['combined_score'] = 0.0
                    
                # Log detailed results for each feature
                logger.debug(f"""
        Feature: {feature}
        - Correlation: {corr:.3f}
        - Mutual Information: {mi_score:.3f}
        - Model Effect: {effect_mean:.3f} ± {effect_std:.3f}
        - Combined Score: {importance[feature]['combined_score']:.3f}
        - Diagnostics:
            * Unique differences: {n_unique}
            * NaN values: {n_nan}
            * Inf values: {n_inf}
            * Zero values: {n_zero}
            * Total samples: {len(feat_diff)}
        """)
            
            logger.info("Feature importance calculation completed")
            
            # Log summary of problematic features
            problematic = [f for f, imp in importance.items() 
                        if imp['diagnostics']['n_unique_differences'] == 1 
                        or imp['diagnostics']['n_nan'] > 0 
                        or imp['diagnostics']['n_inf'] > 0]
            if problematic:
                logger.warning(f"Problematic features detected: {', '.join(problematic)}")
            
            return importance   
    
    def _calculate_feature_correlations(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Calculate feature correlations with diagnostics.
        
        Args:
            X1: Feature matrix for first bigram in each pair
            X2: Feature matrix for second bigram in each pair
            feature_names: List of feature names
            
        Returns:
            Tuple of:
            - DataFrame containing correlation matrix
            - Dictionary of correlation diagnostics
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            n_features = len(feature_names)
            corr_matrix = np.zeros((n_features, n_features))
            diagnostics = {
                'feature_stats': {},
                'correlation_issues': [],
                'highly_correlated_pairs': []
            }
            
            logger = logging.getLogger(__name__)
            logger.info("Starting feature correlation analysis...")
            
            # First analyze individual features
            for i, feature in enumerate(feature_names):
                diff = X1[:, i] - X2[:, i]
                stats = {
                    'n_unique': len(np.unique(diff)),
                    'n_nan': np.sum(np.isnan(diff)),
                    'n_inf': np.sum(np.isinf(diff)),
                    'n_zero': np.sum(diff == 0),
                    'mean': np.mean(diff[~np.isnan(diff)]) if not np.all(np.isnan(diff)) else np.nan,
                    'std': np.std(diff[~np.isnan(diff)]) if not np.all(np.isnan(diff)) else np.nan
                }
                diagnostics['feature_stats'][feature] = stats
                
                if stats['n_unique'] == 1:
                    logger.warning(f"Feature '{feature}' has constant differences")
                if stats['n_nan'] > 0:
                    logger.warning(f"Feature '{feature}' has {stats['n_nan']} NaN differences")
            
            # Then calculate correlations
            for i in range(n_features):
                for j in range(n_features):
                    diff_i = X1[:, i] - X2[:, i]
                    diff_j = X1[:, j] - X2[:, j]
                    
                    # Skip if either feature has issues
                    if (len(np.unique(diff_i)) == 1 or len(np.unique(diff_j)) == 1 or
                        np.any(np.isnan(diff_i)) or np.any(np.isnan(diff_j)) or
                        np.any(np.isinf(diff_i)) or np.any(np.isinf(diff_j))):
                        corr = 0.0
                        if i != j:  # Don't report self-correlations
                            diagnostics['correlation_issues'].append({
                                'feature1': feature_names[i],
                                'feature2': feature_names[j],
                                'reason': 'constant or invalid values'
                            })
                    else:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                corr, _ = stats.spearmanr(diff_i, diff_j)
                                if np.isnan(corr):
                                    corr = 0.0
                                    diagnostics['correlation_issues'].append({
                                        'feature1': feature_names[i],
                                        'feature2': feature_names[j],
                                        'reason': 'correlation calculation failed'
                                    })
                        except Exception as e:
                            logger.error(f"Error calculating correlation between '{feature_names[i]}' and '{feature_names[j]}': {str(e)}")
                            corr = 0.0
                    
                    corr_matrix[i, j] = corr
                    
                    # Track highly correlated features
                    if i < j and abs(corr) > self.correlation_threshold:
                        diagnostics['highly_correlated_pairs'].append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': float(corr)
                        })
            
            # Log summary statistics
            logger.info(f"Found {len(diagnostics['correlation_issues'])} correlation calculation issues")
            logger.info(f"Found {len(diagnostics['highly_correlated_pairs'])} highly correlated feature pairs")
            
            if diagnostics['highly_correlated_pairs']:
                logger.info("Highly correlated feature pairs:")
                for pair in sorted(diagnostics['highly_correlated_pairs'], 
                                key=lambda x: abs(x['correlation']), reverse=True):
                    logger.info(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
            
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

    def _save_detailed_report(
        self,
        results: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """Save detailed report of final feature selection results."""
        report_file = output_dir / "feature_selection_results.txt"
        
        with open(report_file, 'w') as f:
            f.write("Feature Selection Results Report\n")
            f.write("============================\n\n")
            
            # 1. Selected Features
            f.write("1. Selected Features\n")
            f.write("-----------------\n")
            for feature in results['selected_features']:
                f.write(f"\n{feature}:\n")
                if 'importance' in results:
                    imp = results['importance'].get(feature, {})
                    f.write(f"  Importance: {imp.get('combined_score', 0):.3f}\n")
                    f.write(f"  Model effect: {imp.get('model_effect_mean', 0):.3f}\n")
                if 'stability' in results:
                    stab = results['stability'].get(feature, {})
                    f.write(f"  Stability: {stab.get('sign_consistency', 0):.3f}\n")
            
            # 2. Rejected Features
            if 'rejected_features' in results:
                f.write("\n2. Rejected Features\n")
                f.write("-----------------\n")
                for feature, reason in results['rejected_features']:
                    f.write(f"  - {feature}: {reason}\n")
            
            # 3. Summary Statistics
            f.write("\n3. Summary Statistics\n")
            f.write("-----------------\n")
            f.write(f"Total features evaluated: {len(results.get('all_features', []))}\n")
            f.write(f"Features selected: {len(results['selected_features'])}\n")
            if 'rejected_features' in results:
                f.write(f"Features rejected: {len(results['rejected_features'])}\n")
                            