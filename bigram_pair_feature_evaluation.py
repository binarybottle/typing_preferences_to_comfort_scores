"""
Feature evaluation module for keyboard layout analysis.

This module provides functionality for:
1. Feature Set Evaluation - Cross-validation and information criteria analysis of feature combinations
2. Feature Analysis - Importance scoring, correlation analysis, and multicollinearity detection
3. Result Persistence - Saving and documenting evaluation results

Key capabilities:
- Hold-out test set separation and participant-grouped cross-validation
- Multiple evaluation metrics (CV performance, WAIC, LOO-CV)
- Feature importance and interaction analysis
- Comprehensive result documentation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score
import pymc as pm
import arviz as az
from scipy import stats
from dataclasses import dataclass
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import pytensor.tensor as pt 
import warnings
import json
import pickle
import logging

from bayesian_modeling import (train_bayesian_glmm, evaluate_model_performance)

logger = logging.getLogger(__name__)

#===================#
# Data Structures   #
#===================#
@dataclass
class FeatureEvaluationResults:
    """
    Container for feature evaluation results.
    
    Attributes:
        cv_scores: Dictionary mapping feature set names to their cross-validation R² scores
        waic_scores: Dictionary mapping feature set names to their WAIC scores
        loo_scores: Dictionary mapping feature set names to their LOO-CV scores
        feature_correlations: DataFrame of pairwise feature correlations
        feature_importance: Dictionary mapping feature names to importance scores (0-1)
        feature_groups: Dictionary mapping group names to lists of feature names
        interaction_scores: Optional dictionary of interaction term importances
        stability_metrics: Optional dictionary of feature stability metrics
    """
    cv_scores: Dict[str, List[float]]
    waic_scores: Dict[str, float]
    loo_scores: Dict[str, float]
    feature_correlations: pd.DataFrame
    feature_importance: Dict[str, float]
    feature_groups: Dict[str, List[str]]
    interaction_scores: Optional[Dict[Tuple[str, str], float]] = None
    stability_metrics: Optional[Dict[str, Dict[str, float]]] = None

#===================#
# Core Evaluation   #
#===================#
def load_interactions_from_file(filepath: str) -> List[List[str]]:
    """
    Load feature interactions from a YAML file.
    
    Args:
        filepath: Path to YAML file containing interaction definitions
        
    Returns:
        List of feature interaction lists
    """
    logger.info(f"Loading interactions from {filepath}")
    try:
        import yaml
        with open(filepath, 'r') as f:
            content = yaml.safe_load(f)
            
        if not content or 'interactions' not in content:
            logger.warning(f"No interactions found in {filepath}")
            return []
            
        interactions = content['interactions']
        
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
    
def create_interaction_features(
    feature_matrix: pd.DataFrame,
    base_features: List[str],
    interactions: List[List[str]]
) -> pd.DataFrame:
    """Create interaction features from base features."""
    augmented_matrix = feature_matrix.copy()
    
    for interaction in interactions:
        if not all(f in base_features for f in interaction):
            continue
            
        # Create interaction term
        interaction_name = "_".join(interaction)
        interaction_value = feature_matrix[interaction[0]].copy()
        for feature in interaction[1:]:
            interaction_value *= feature_matrix[feature]
            
        augmented_matrix[interaction_name] = interaction_value
        
    return augmented_matrix

def fit_hierarchical_model(
    X: np.ndarray,
    y: np.ndarray,
    participants: np.ndarray,
    n_samples: int
) -> Dict[str, float]:
    """
    Fit a hierarchical logistic regression model using PyMC.
    """
    unique_participants = np.unique(participants)
    n_participants = len(unique_participants)
    n_features = X.shape[1]
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        with pm.Model() as model:
            # Simple global priors
            beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
            
            # Simple participant-level effects
            participant_offset = pm.Normal('participant_offset', mu=0, sigma=1, shape=n_participants)
            
            # Linear predictor - keeping it simple
            participant_idx = pd.Categorical(participants).codes
            mu = pm.Deterministic('mu', pt.dot(X_scaled, beta) + participant_offset[participant_idx])
            
            # Likelihood
            y_obs = pm.Bernoulli('y', logit_p=mu, observed=y)
            
            # Basic sampling settings that were working
            trace = pm.sample(
                draws=n_samples,
                tune=1000,
                chains=4,
                return_inferencedata=True
            )
            
            # Simple prediction aggregation
            ppc = pm.sample_posterior_predictive(trace)
            y_pred_proba = np.mean(ppc.posterior_predictive['y'], axis=(0, 1))
        
        return {
            'auc': roc_auc_score(y, y_pred_proba),
            'accuracy': accuracy_score(y, y_pred_proba > 0.5),
            'trace': trace
        }
        
    except Exception as e:
        logger.error(f"Model fitting error: {str(e)}")
        raise

def evaluate_feature_stability(x: pd.Series, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate feature stability across different subsets of data with robust handling of edge cases.
    
    Args:
        x: Input feature values
        y: Target values
        
    Returns:
        Dictionary containing stability metrics
    """
    n_splits = 5
    r2_scores = []
    correlation_scores = []
    
    # Initial check for constant feature
    if len(np.unique(x)) == 1:
        logger.info(f"Feature has constant value {x.iloc[0]} - stability metrics will be zero")
        return {
            'r2_mean': 0.0,
            'r2_std': 0.0,
            'correlation_mean': 0.0,
            'correlation_std': 0.0,
            'stability_score': 0.0,
            'is_constant': True,
            'constant_value': x.iloc[0]
        }
    
    # Check for sufficient unique values
    unique_ratio = len(np.unique(x)) / len(x)
    if unique_ratio < 0.01:  # Less than 1% unique values
        logger.info(f"Feature has low variance (unique ratio: {unique_ratio:.3f})")
        return {
            'r2_mean': 0.0,
            'r2_std': 0.0,
            'correlation_mean': 0.0,
            'correlation_std': 0.0,
            'stability_score': 0.0,
            'is_low_variance': True,
            'unique_ratio': unique_ratio
        }
    
    # Create random splits while preserving ratio
    try:
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        split_size = len(indices) // n_splits
        
        for i in range(n_splits):
            # Get subset of data
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < n_splits - 1 else len(indices)
            subset_indices = indices[start_idx:end_idx]
            
            # Extract subset data
            x_subset = x.iloc[subset_indices]
            y_subset = y[subset_indices]
            
            # Skip if subset is constant
            if len(np.unique(x_subset)) == 1:
                logger.debug(f"Skipping constant subset in split {i+1}")
                continue
            
            # Calculate R² for subset
            try:
                r2 = calculate_univariate_r2(x_subset, y_subset)
                if not np.isnan(r2):
                    r2_scores.append(r2)
            except Exception as e:
                logger.debug(f"R² calculation failed for split {i+1}: {str(e)}")
            
            # Calculate correlation for subset
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    corr, _ = stats.spearmanr(x_subset, y_subset)
                    if not np.isnan(corr):
                        correlation_scores.append(corr)
            except Exception as e:
                logger.debug(f"Correlation calculation failed for split {i+1}: {str(e)}")
        
        # Calculate stability metrics if we have enough valid scores
        if len(r2_scores) >= 2 and len(correlation_scores) >= 2:
            r2_mean = np.mean(r2_scores)
            r2_std = np.std(r2_scores)
            corr_mean = np.mean(correlation_scores)
            corr_std = np.std(correlation_scores)
            
            # Calculate stability score (inverse of coefficient of variation)
            stability_score = 1.0 - (r2_std / (r2_mean + 1e-10))
            stability_score = max(0.0, min(1.0, stability_score))  # Clamp to [0,1]
            
            return {
                'r2_mean': r2_mean,
                'r2_std': r2_std,
                'correlation_mean': corr_mean,
                'correlation_std': corr_std,
                'stability_score': stability_score,
                'n_valid_splits': len(r2_scores),
                'is_constant': False,
                'is_low_variance': False
            }
        else:
            logger.warning("Insufficient valid splits for stability calculation")
            return {
                'r2_mean': 0.0,
                'r2_std': 0.0,
                'correlation_mean': 0.0,
                'correlation_std': 0.0,
                'stability_score': 0.0,
                'n_valid_splits': len(r2_scores),
                'insufficient_data': True
            }
            
    except Exception as e:
        logger.error(f"Stability evaluation failed: {str(e)}")
        return {
            'r2_mean': 0.0,
            'r2_std': 0.0,
            'correlation_mean': 0.0,
            'correlation_std': 0.0,
            'stability_score': 0.0,
            'error': str(e)
        }

def calculate_univariate_r2(x: pd.Series, y: np.ndarray) -> float:
    """
    Calculate R² for a single feature with improved handling of edge cases.
    
    Args:
        x: Input feature values
        y: Target values
        
    Returns:
        R² score, or 0.0 if calculation is not possible
    """
    # Handle constant input
    if len(np.unique(x)) == 1:
        return 0.0
    
    # Handle NaN/inf values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Check if we have enough data points after cleaning
    if len(x_clean) < 2:
        return 0.0
    
    try:
        # Use statsmodels for more robust linear regression
        X = sm.add_constant(x_clean)
        model = sm.OLS(y_clean, X)
        results = model.fit()
        
        # Calculate R² manually to ensure validity
        y_pred = results.predict(X)
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        
        # Handle division by zero
        if ss_tot == 0:
            return 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        
        # Clamp R² to valid range [0, 1]
        r2 = max(0.0, min(1.0, r2))
        
        return r2
        
    except Exception as e:
        logger.debug(f"R² calculation failed: {str(e)}")
        return 0.0
    
def evaluate_feature_sets(
   feature_matrix: pd.DataFrame,
   target_vector: np.ndarray,
   participants: np.ndarray,
   feature_sets: List[Dict[str, Any]],
   output_dir: Path,
   n_splits: int = 5,
   n_samples: int = 10000,
   chains: int = 8,
   target_accept: float = 0.85,
   stability_threshold: float = 0.6,
   min_effect_size: float = 0.1
) -> Dict[str, List[float]]:
    """
    Enhanced evaluation of different feature sets using cross-validation with stability analysis.
    
    Args:
        feature_matrix: Features to evaluate
        target_vector: Target values to predict
        participants: Participant IDs for grouping
        feature_sets: List of feature set configurations
        output_dir: Directory for output files
        n_splits: Number of cross-validation splits
        n_samples: Number of MCMC samples
        chains: Number of MCMC chains
        target_accept: Target acceptance rate for MCMC
        stability_threshold: Minimum stability score to consider a feature reliable
        min_effect_size: Minimum absolute effect size to consider a feature significant
        
    Returns:
        Dictionary containing evaluation results
    """
    results = {}
    all_cv_metrics = {}
    all_feature_effects = {}
    feature_stability_results = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Updated file paths with more descriptive names
    raw_metrics_path = output_dir / "raw_metrics.txt"
    feature_details_path = output_dir / "feature_details.txt"
    stability_report_path = output_dir / "stability_report.txt"
    
    logger.info("Starting enhanced feature set evaluation")
    logger.info(f"Number of feature sets: {len(feature_sets)}")
    logger.info(f"Target vector shape: {target_vector.shape}")
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    with open(raw_metrics_path, 'w') as raw_file, \
         open(feature_details_path, 'w') as details_file, \
         open(stability_report_path, 'w') as stability_file:
        
        raw_file.write("Raw Cross-Validation Metrics\n")
        raw_file.write("==========================\n\n")
        
        details_file.write("Detailed Feature Analysis\n")
        details_file.write("=======================\n")
        
        stability_file.write("Feature Stability Analysis\n")
        stability_file.write("========================\n")
        
        for feature_set in feature_sets:
            set_name = feature_set['name']
            features = feature_set['features'].copy()
            
            # Handle interactions
            interactions = []
            if 'interactions' in feature_set:
                if isinstance(feature_set['interactions'], str):
                    interactions = load_interactions_from_file(feature_set['interactions'])
                elif isinstance(feature_set['interactions'], list):
                    interactions = feature_set['interactions']
            
            logger.info(f"\nEvaluating feature set: {set_name}")
            logger.info(f"Base features: {features}")
            if interactions:
                logger.info(f"Interactions: {[' × '.join(i) for i in interactions]}")
            
            # Initial feature information
            details_file.write(f"\nFeature Set: {set_name}\n")
            details_file.write("=" * 50 + "\n")
            details_file.write("Base Features:\n")
            for feat in features:
                details_file.write(f"  - {feat}\n")
            if interactions:
                details_file.write("\nInteractions:\n")
                for inter in interactions:
                    details_file.write(f"  - {' × '.join(inter)}\n")
            
            # Select features and create interaction terms
            try:
                X = feature_matrix[features].copy()
            except KeyError as e:
                logger.error(f"Missing feature in matrix: {str(e)}")
                continue
            
            # Add interaction terms
            valid_interactions = []
            for interaction in interactions:
                if all(feat in features for feat in interaction):
                    try:
                        interaction_name = "_".join(interaction)
                        X[interaction_name] = X[interaction[0]].copy()
                        for feat in interaction[1:]:
                            X[interaction_name] *= X[feat]
                        features.append(interaction_name)
                        valid_interactions.append(interaction)
                        logger.info(f"Added interaction: {' × '.join(interaction)}")
                    except Exception as e:
                        logger.error(f"Failed to create interaction {interaction}: {str(e)}")
                        continue
            
            # Evaluate feature stability before cross-validation
            stability_file.write(f"\nStability Analysis for {set_name}:\n")
            stability_file.write("=" * 50 + "\n")
            
            feature_stability = {}
            for feature in features:
                try:
                    stability_metrics = evaluate_feature_stability(X[feature], target_vector)
                    feature_stability[feature] = stability_metrics
                    
                    stability_file.write(f"\n{feature}:\n")
                    stability_file.write(f"  R² stability: {stability_metrics['stability_score']:.3f}\n")
                    stability_file.write(f"  R² mean: {stability_metrics['r2_mean']:.3f} ± {stability_metrics['r2_std']:.3f}\n")
                    stability_file.write(f"  Correlation stability: {1.0 - stability_metrics['correlation_std']:.3f}\n")
                    
                    if stability_metrics['stability_score'] < stability_threshold:
                        stability_file.write("  WARNING: Low stability detected\n")
                except Exception as e:
                    logger.error(f"Stability evaluation failed for {feature}: {str(e)}")
                    continue
            
            feature_stability_results[set_name] = feature_stability
            
            # Initialize storage for feature effects
            feature_effects_all_folds = {feature: [] for feature in features}

            # Cross-validation setup
            cv_metrics = {
                'r2': [], 'rmse': [], 'mae': [], 
                'correlation': [], 'accuracy': []
            }
            
            group_kfold = GroupKFold(n_splits=n_splits)
            
            logger.info(f"Starting {n_splits}-fold cross-validation")
            
            details_file.write("\nFeature Analysis By Fold:\n")
            details_file.write("-" * 30 + "\n")
            
            for fold, (train_idx, val_idx) in enumerate(
                group_kfold.split(X, target_vector, participants)
            ):
                logger.info(f"\nFold {fold + 1}/{n_splits}")
                details_file.write(f"\nFold {fold + 1}:\n")
                
                try:
                    # Split data
                    X_train = X.iloc[train_idx]
                    X_val = X.iloc[val_idx]
                    y_train = target_vector[train_idx]
                    y_val = target_vector[val_idx]
                    participants_train = participants[train_idx]
                    participants_val = participants[val_idx]
                    
                    # Calculate raw feature relationships with error handling
                    for feature in features:
                        try:
                            # Train set relationships with robust calculation
                            train_metrics = evaluate_feature_stability(X_train[feature], y_train)
                            val_metrics = evaluate_feature_stability(X_val[feature], y_val)
                            
                            details_file.write(f"\n  {feature}:\n")
                            details_file.write(f"    Train R²: {train_metrics['r2_mean']:.3f} ± {train_metrics['r2_std']:.3f}\n")
                            details_file.write(f"    Val R²: {val_metrics['r2_mean']:.3f} ± {val_metrics['r2_std']:.3f}\n")
                            details_file.write(f"    Stability score: {train_metrics['stability_score']:.3f}\n")
                            
                            feature_data = {
                                'train_metrics': train_metrics,
                                'val_metrics': val_metrics,
                                'train_values': X_train[feature].values,
                                'train_target': y_train,
                                'val_values': X_val[feature].values,
                                'val_target': y_val
                            }
                            feature_effects_all_folds[feature].append(feature_data)
                            
                        except Exception as e:
                            logger.error(f"Feature analysis failed for {feature} in fold {fold}: {str(e)}")
                            continue
                    
                    # Train model with error handling
                    try:
                        trace, model, _ = train_bayesian_glmm(
                            feature_matrix=X_train,
                            target_vector=y_train,
                            participants=participants_train,
                            design_features=features,
                            control_features=[],
                            inference_method='mcmc',
                            n_samples=n_samples,
                            chains=chains,
                            target_accept=target_accept
                        )
                        
                        # Store model effects
                        try:
                            feature_effects = az.summary(trace, var_names=features)
                            for feature in features:
                                if feature in feature_effects_all_folds:
                                    fold_data = feature_effects_all_folds[feature][-1]
                                    effect_mean = float(feature_effects.loc[feature, 'mean'])
                                    effect_sd = float(feature_effects.loc[feature, 'sd'])
                                    
                                    fold_data.update({
                                        'model_effect_mean': effect_mean,
                                        'model_effect_sd': effect_sd,
                                        'effect_significant': abs(effect_mean) > min_effect_size and 
                                                           abs(effect_mean) > 2 * effect_sd
                                    })
                                    
                                    details_file.write(
                                        f"    Model effect: {effect_mean:.3f} ± {effect_sd:.3f}"
                                        f" ({'significant' if fold_data['effect_significant'] else 'not significant'})\n"
                                    )
                        except Exception as e:
                            logger.error(f"Could not compute feature effects for fold {fold}: {str(e)}")
                            continue
                        
                        # Evaluate
                        metrics = evaluate_model_performance(
                            trace=trace,
                            feature_matrix=X_val,
                            target_vector=y_val,
                            participants=participants_val,
                            design_features=features,
                            control_features=[]
                        )
                        
                        # Record metrics
                        for metric, value in metrics.items():
                            cv_metrics[metric].append(value)
                        
                        # Write fold results
                        raw_file.write(f"Feature Set {set_name}, Fold {fold + 1}:\n")
                        for metric, value in metrics.items():
                            raw_file.write(f"  {metric}: {value:.3f}\n")
                        raw_file.write("\n")
                        
                    except Exception as e:
                        logger.error(f"Model training/evaluation failed for fold {fold}: {str(e)}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Fold {fold} processing failed: {str(e)}")
                    continue
            
            # Calculate and write summary statistics
            if any(len(scores) > 0 for scores in cv_metrics.values()):
                avg_metrics = {metric: np.mean(values) for metric, values in cv_metrics.items() if values}
                std_metrics = {metric: np.std(values) for metric, values in cv_metrics.items() if values}
                
                results[set_name] = avg_metrics
                all_cv_metrics[set_name] = cv_metrics
                all_feature_effects[set_name] = feature_effects_all_folds
            else:
                logger.error(f"No valid metrics collected for feature set {set_name}")
    
    # Return comprehensive results
    return {
        'cv_metrics': all_cv_metrics,
        'feature_effects': all_feature_effects,
        'feature_stability': feature_stability_results,
        'evaluation_success': True
    }

def calculate_waic_loo(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate WAIC and LOO scores for model comparison.
    
    These information criteria help assess model fit while accounting for complexity:
    - WAIC (Widely Applicable Information Criterion): Generalization of AIC
    - LOO (Leave-One-Out Cross-Validation): Cross-validation based criterion
    
    Lower scores indicate better models.
    
    Args:
        feature_matrix: Features to evaluate
        target_vector: Target values to predict
        
    Returns:
        Tuple of (WAIC score, LOO score)
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    
    try:
        with pm.Model() as model:
            # Set up model
            feature_effects = {
                feat: pm.Normal(feat, mu=0, sigma=1)
                for feat in feature_matrix.columns
            }
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu = sum(feature_effects[feat] * X_scaled[:, i] 
                    for i, feat in enumerate(feature_matrix.columns))
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=target_vector)
            
            # Sample
            trace = pm.sample(
                draws=500,
                tune=2000,
                chains=4,
                target_accept=0.99,
                init='adapt_diag',
                return_inferencedata=True,
                compute_convergence_checks=True,
                cores=1
            )
            
            # Calculate criteria
            waic = az.waic(trace)
            loo = az.loo(trace)
            
            return waic.waic, loo.loo
            
    except Exception as e:
        logger.error(f"WAIC/LOO computation failed: {e}")
        return np.nan, np.nan

#===================#
# Feature Analysis  #
#===================#
def calculate_feature_importance(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculate feature importance using multiple methods and aggregate results.
    
    Combines three approaches:
    1. Correlation with target (30% weight)
    2. Mutual information (30% weight)
    3. Permutation importance (40% weight)
    
    This multi-method approach provides more robust importance scores than
    any single method alone.
    
    Args:
        feature_matrix: DataFrame of features to evaluate
        target_vector: Array of target values
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to their importance scores (0-1 scale)
    """
    # Method 1: Correlation with target
    corr_importance = {feature: abs(stats.spearmanr(feature_matrix[feature], 
                                                   target_vector)[0])
                      for feature in feature_names}
    
    # Method 2: Mutual information
    mi_importance = dict(zip(feature_names, 
                           mutual_info_regression(feature_matrix, target_vector)))
    
    # Method 3: Permutation importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(feature_matrix, target_vector)
    perm_importance = permutation_importance(rf, feature_matrix, target_vector,
                                           n_repeats=10, random_state=42)
    perm_scores = dict(zip(feature_names, perm_importance.importances_mean))
    
    # Normalize and combine scores
    importance_scores = {}
    for feature in feature_names:
        importance_scores[feature] = (
            0.3 * corr_importance[feature] / max(corr_importance.values()) +
            0.3 * mi_importance[feature] / max(mi_importance.values()) +
            0.4 * perm_scores[feature] / max(perm_scores.values())
        )
    
    return importance_scores

def calculate_feature_correlations(
    feature_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate pairwise correlations between all features.
    
    Args:
        feature_matrix: DataFrame of feature values to analyze
        
    Returns:
        DataFrame containing correlation matrix where both rows and columns
        are feature names and values are correlation coefficients (-1 to 1)
    """
    correlations = feature_matrix.corr()
    
    # Log high correlations
    high_corr_threshold = 0.7
    for i in range(len(correlations.columns)):
        for j in range(i + 1, len(correlations.columns)):
            corr = abs(correlations.iloc[i, j])
            if corr > high_corr_threshold:
                logger.warning(
                    f"High correlation ({corr:.3f}) between "
                    f"{correlations.columns[i]} and {correlations.columns[j]}"
                )
    
    return correlations

def check_multicollinearity_vif(
    feature_matrix: pd.DataFrame
) -> Dict[str, List]:
    """
    Check for multicollinearity among features using Variance Inflation Factor.
    
    Performs two types of checks:
    1. VIF calculation for each feature (>5 indicates high multicollinearity)
    2. Pairwise correlation analysis (>0.7 indicates high correlation)
    
    Args:
        feature_matrix: DataFrame of features to check
        
    Returns:
        Dictionary containing:
        - vif: List of VIF scores and status for each feature
        - high_correlations: List of highly correlated feature pairs
    """
    results = {
        'vif': [],
        'high_correlations': []
    }
    
    # Add constant term for VIF calculation
    X = sm.add_constant(feature_matrix)
    
    # Calculate VIF for each feature
    for i, column in enumerate(X.columns):
        if column != 'const':
            try:
                vif = variance_inflation_factor(X.values, i)
                results['vif'].append({
                    'Feature': column,
                    'VIF': vif,
                    'Status': 'High multicollinearity' if vif > 5 else 'Acceptable'
                })
                if vif > 5:
                    logger.warning(f"High VIF ({vif:.2f}) for feature: {column}")
            except Exception as e:
                logger.error(f"Could not calculate VIF for {column}: {str(e)}")
    
    # Check for high correlations
    corr_matrix = feature_matrix.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.7:
                results['high_correlations'].append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    return results

#===================#
# Results Handling  #
#===================#
def save_evaluation_results(results: Dict, output_dir: Path) -> None:
    """Save feature evaluation results to files for later analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "evaluation_results.pkl"
    
    # Convert any numpy arrays to lists for serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in value.items()
            }
        else:
            # This line had the bug - using v instead of value
            serializable_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
    
    with open(results_file, 'wb') as f:
        pickle.dump(serializable_results, f)
    
    logger.info(f"Saved evaluation results to {results_file}")

def analyze_and_recommend(
    output_dir: Path,
    cv_metrics: Dict[str, Dict[str, List[float]]],
    feature_effects: Dict[str, Dict[str, List[Any]]],
    feature_sets: List[Dict[str, Any]],
    feature_matrix: pd.DataFrame,
    correlation_threshold: float = 0.2,    # Reduced from 0.3
    importance_threshold: float = 0.03,    # Reduced from 0.05
    variability_threshold: float = 0.7,    # Increased from 0.5
    vif_threshold: float = 5.0,
    redundancy_correlation_threshold: float = 0.7
) -> None:
    """
    Analyze evaluation results and provide feature recommendations.
    """
    recommendations_path = output_dir / "recommendations.txt"
    redundancy_path = output_dir / "redundancy_analysis.txt"
    feature_metrics_path = output_dir / "feature_metrics.txt"
    
    logger.info(f"Analyzing features with thresholds:")
    logger.info(f"  Correlation: {correlation_threshold}")
    logger.info(f"  Importance: {importance_threshold}")
    logger.info(f"  Variability: {variability_threshold}")
    logger.info(f"  VIF: {vif_threshold}")
    logger.info(f"  Redundancy correlation: {redundancy_correlation_threshold}")

    # Calculate redundancy metrics first
    correlation_matrix = feature_matrix.corr()
    redundant_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = abs(correlation_matrix.iloc[i, j])
            if corr > redundancy_correlation_threshold:
                redundant_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr
                ))

    # Calculate VIF scores
    vif_data = {}
    X = sm.add_constant(feature_matrix)
    for i, feature in enumerate(X.columns):
        if feature != 'const':
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_data[feature] = vif
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {feature}: {str(e)}")
    
    # Write redundancy analysis
    with open(redundancy_path, 'w') as f:
        f.write("Feature Redundancy Analysis\n")
        f.write("=========================\n\n")
        
        f.write("1. Highly Correlated Feature Pairs\n")
        f.write("--------------------------------\n")
        if redundant_pairs:
            for feat1, feat2, corr in sorted(redundant_pairs, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"{feat1} ↔ {feat2}: {corr:.3f}\n")
        else:
            f.write("No highly correlated feature pairs found\n")
            
        f.write("\n2. Variance Inflation Factors\n")
        f.write("---------------------------\n")
        for feature, vif in sorted(vif_data.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{feature}: {vif:.2f}")
            if vif > vif_threshold:
                f.write(" (!)")
            f.write("\n")
    
    # Analyze each feature
    feature_metrics = {}
    all_features = set()
    
    # Collect all features
    for set_name, effects in feature_effects.items():
        all_features.update(effects.keys())
    
    # Calculate metrics for each feature
    for feature in all_features:
        feature_data = []
        for set_name, effects in feature_effects.items():
            if feature in effects:
                feature_data.extend(effects[feature])
        
        if feature_data:
            # Model effects
            model_effects = [d.get('model_effect_mean', 0) for d in feature_data if 'model_effect_mean' in d]
            effect_sds = [d.get('model_effect_sd', 0) for d in feature_data if 'model_effect_sd' in d]
            
            # Raw relationships
            train_corrs = [d.get('train_correlation', 0) for d in feature_data if 'train_correlation' in d]
            val_corrs = [d.get('val_correlation', 0) for d in feature_data if 'val_correlation' in d]
            train_r2s = [d.get('train_r2', 0) for d in feature_data if 'train_r2' in d]
            val_r2s = [d.get('val_r2', 0) for d in feature_data if 'val_r2' in d]
            
            feature_metrics[feature] = {
                'effect_mean': np.mean(model_effects) if model_effects else 0,
                'effect_std': np.std(model_effects) if model_effects else 0,
                'effect_stability': np.mean(effect_sds) if effect_sds else float('inf'),
                'train_correlation_mean': np.mean(train_corrs) if train_corrs else 0,
                'train_correlation_std': np.std(train_corrs) if train_corrs else 0,
                'val_correlation_mean': np.mean(val_corrs) if val_corrs else 0,
                'val_correlation_std': np.std(val_corrs) if val_corrs else 0,
                'train_r2_mean': np.mean(train_r2s) if train_r2s else 0,
                'val_r2_mean': np.mean(val_r2s) if val_r2s else 0,
                'r2_stability': np.std(val_r2s) / (np.mean(val_r2s) + 1e-10) if val_r2s else float('inf'),
                'consistent_sign': all(np.sign(e) == np.sign(model_effects[0]) for e in model_effects) if model_effects else False,
                'vif': vif_data.get(feature, 0),
                'n_appearances': len(feature_data)
            }
    
    # Write detailed feature metrics
    with open(feature_metrics_path, 'w') as f:
        f.write("Detailed Feature Metrics\n")
        f.write("=====================\n\n")
        
        for feature, metrics in sorted(feature_metrics.items()):
            f.write(f"{feature}:\n")
            f.write("-" * len(feature) + "\n")
            f.write(f"Model Effects:\n")
            f.write(f"  Mean: {metrics['effect_mean']:.3f}\n")
            f.write(f"  Std: {metrics['effect_std']:.3f}\n")
            f.write(f"  Stability: {metrics['effect_stability']:.3f}\n")
            f.write(f"  Consistent sign: {metrics['consistent_sign']}\n")
            
            f.write(f"\nCorrelations:\n")
            f.write(f"  Train: {metrics['train_correlation_mean']:.3f} ± {metrics['train_correlation_std']:.3f}\n")
            f.write(f"  Validation: {metrics['val_correlation_mean']:.3f} ± {metrics['val_correlation_std']:.3f}\n")
            
            f.write(f"\nR² Values:\n")
            f.write(f"  Train: {metrics['train_r2_mean']:.3f}\n")
            f.write(f"  Validation: {metrics['val_r2_mean']:.3f}\n")
            f.write(f"  Stability: {metrics['r2_stability']:.3f}\n")
            
            f.write(f"\nRedundancy:\n")
            f.write(f"  VIF: {metrics['vif']:.2f}\n")
            correlations = [(other_feat, correlation_matrix.loc[feature, other_feat])
                          for other_feat in correlation_matrix.columns
                          if other_feat != feature]
            top_corr = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:3]
            f.write("  Top correlations:\n")
            for other_feat, corr in top_corr:
                f.write(f"    - {other_feat}: {corr:.3f}\n")
            
            f.write(f"\nOther:\n")
            f.write(f"  Appearances: {metrics['n_appearances']}\n")
            f.write("\n" + "="*50 + "\n\n")
    
    # Categorize features
    strong_features = []
    unstable_features = []
    weak_features = []
    
    # Debug categorization
    logger.info("=== Categorization Debug ===")
    
    for feature, metrics in feature_metrics.items():
        logger.info(f"\nAnalyzing feature: {feature}")
        
        # Consider a feature strong if it meets ANY of these criteria
        is_strong = (
            abs(metrics['val_correlation_mean']) > correlation_threshold or
            abs(metrics['effect_mean']) > importance_threshold or
            abs(metrics['val_r2_mean']) > 0.1
        )
        
        logger.info("Strong criteria:")
        logger.info(f"  |val_corr| > {correlation_threshold}: {abs(metrics['val_correlation_mean']):.3f}")
        logger.info(f"  |effect| > {importance_threshold}: {abs(metrics['effect_mean']):.3f}")
        logger.info(f"  |val_R²| > 0.1: {abs(metrics['val_r2_mean']):.3f}")
        
        # Check stability
        is_stable = (
            metrics['effect_stability'] < variability_threshold and
            metrics['r2_stability'] < variability_threshold and
            metrics['consistent_sign']
        )
        
        logger.info("Stability criteria:")
        logger.info(f"  effect_stability < {variability_threshold}: {metrics['effect_stability']:.3f}")
        logger.info(f"  r2_stability < {variability_threshold}: {metrics['r2_stability']:.3f}")
        logger.info(f"  consistent_sign: {metrics['consistent_sign']}")
        
        if is_strong:
            if is_stable:
                strong_features.append((
                    feature,
                    metrics['effect_mean'],
                    metrics['val_correlation_mean'],
                    metrics['val_r2_mean'],
                    metrics['vif']
                ))
                logger.info("Categorized as: STRONG")
            else:
                unstable_features.append((
                    feature,
                    metrics['effect_mean'],
                    metrics['val_correlation_mean'],
                    metrics['val_r2_mean'],
                    metrics['vif']
                ))
                logger.info("Categorized as: UNSTABLE")
        else:
            weak_features.append((
                feature,
                metrics['effect_mean'],
                metrics['val_correlation_mean'],
                metrics['val_r2_mean'],
                metrics['vif']
            ))
            logger.info("Categorized as: WEAK")
    
    # Write recommendations
    with open(recommendations_path, 'w') as f:
        f.write("Feature Analysis and Recommendations\n")
        f.write("=================================\n\n")
        
        # Overall model performance
        f.write("1. Model Performance\n")
        f.write("-----------------\n")
        for set_name, metrics in cv_metrics.items():
            f.write(f"\n{set_name}:\n")
            for metric, values in metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                f.write(f"  {metric}: {mean_val:.3f} ± {std_val:.3f}\n")
        
        # Feature categorization
        f.write("\n2. Feature Categorization\n")
        f.write("----------------------\n")
        
        if strong_features:
            f.write("\nStrong Features (Keep):\n")
            for feat, effect, corr, r2, vif in sorted(strong_features, key=lambda x: abs(x[1]), reverse=True):
                f.write(f"\n{feat}:\n")
                f.write(f"  Effect size: {effect:.3f}\n")
                f.write(f"  Correlation: {corr:.3f}\n")
                f.write(f"  R²: {r2:.3f}\n")
                f.write(f"  VIF: {vif:.2f}\n")
                if vif > vif_threshold:
                    f.write("  WARNING: High multicollinearity\n")
                    
                    # Show correlated features
                    correlations = [(other_feat, correlation_matrix.loc[feat, other_feat])
                                  for other_feat in correlation_matrix.columns
                                  if other_feat != feat]
                    top_corr = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:3]
                    f.write("  Most correlated with:\n")
                    for other_feat, corr_val in top_corr:
                        f.write(f"    - {other_feat}: {corr_val:.3f}\n")
        
        if unstable_features:
            f.write("\nUnstable Features (Review):\n")
            for feat, effect, corr, r2, vif in sorted(unstable_features, key=lambda x: abs(x[1]), reverse=True):
                f.write(f"\n{feat}:\n")
                f.write(f"  Effect size: {effect:.3f}\n")
                f.write(f"  Correlation: {corr:.3f}\n")
                f.write(f"  R²: {r2:.3f}\n")
                f.write(f"  VIF: {vif:.2f}\n")
                
                metrics = feature_metrics[feat]
                f.write(f"  Stability issues:\n")
                if metrics['effect_stability'] >= variability_threshold:
                    f.write(f"    - High effect variability: {metrics['effect_stability']:.3f}\n")
                if metrics['r2_stability'] >= variability_threshold:
                    f.write(f"    - High R² variability: {metrics['r2_stability']:.3f}\n")
                if not metrics['consistent_sign']:
                    f.write("    - Inconsistent effect direction across folds\n")
        
        if weak_features:
            f.write("\nWeak Features (Consider Removing):\n")
            for feat, effect, corr, r2, vif in sorted(weak_features, key=lambda x: abs(x[3]), reverse=True):
                f.write(f"\n{feat}:\n")
                f.write(f"  Effect size: {effect:.3f}\n")
                f.write(f"  Correlation: {corr:.3f}\n")
                f.write(f"  R²: {r2:.3f}\n")
                f.write(f"  VIF: {vif:.2f}\n")
        
        # Redundancy recommendations
        f.write("\n3. Redundancy Analysis\n")
        f.write("--------------------\n")
        
        if redundant_pairs:
            f.write("\nHighly correlated feature pairs:\n")
            for feat1, feat2, corr in sorted(redundant_pairs, key=lambda x: abs(x[2]), reverse=True):
                f.write(f"{feat1} ↔ {feat2}: {corr:.3f}\n")
                
                # Add recommendations for which to keep
                if feat1 in feature_metrics and feat2 in feature_metrics:
                    metrics1 = feature_metrics[feat1]
                    metrics2 = feature_metrics[feat2]
                    
                    # Compare features based on multiple criteria
                    score1 = (abs(metrics1['effect_mean']) + 
                            abs(metrics1['val_correlation_mean']) + 
                            abs(metrics1['val_r2_mean']))
                    score2 = (abs(metrics2['effect_mean']) + 
                            abs(metrics2['val_correlation_mean']) + 
                            abs(metrics2['val_r2_mean']))
                    
                    if score1 > score2:
                        f.write(f"  → Consider keeping {feat1} over {feat2}\n")
                    else:
                        f.write(f"  → Consider keeping {feat2} over {feat1}\n")

        high_vif_features = [(feat, vif) for feat, vif in vif_data.items() if vif > vif_threshold]
        if high_vif_features:
            f.write("\nFeatures with high multicollinearity:\n")
            for feature, vif in sorted(high_vif_features, key=lambda x: x[1], reverse=True):
                f.write(f"\n{feature} (VIF: {vif:.2f}):\n")
                # Show correlations
                correlations = [(other_feat, correlation_matrix.loc[feature, other_feat])
                              for other_feat in correlation_matrix.columns
                              if other_feat != feature]
                top_corr = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)[:3]
                f.write("  Correlated with:\n")
                for other_feat, corr_val in top_corr:
                    f.write(f"    - {other_feat}: {corr_val:.3f}\n")

        # Final recommendations
        f.write("\n4. Final Recommendations\n")
        f.write("---------------------\n")
        
        # Recommended feature set
        f.write("\n4.1 Recommended Features\n")
        
        if strong_features:
            f.write("\nDefinitely keep:\n")
            for feat, _, _, _, _ in sorted(strong_features, key=lambda x: abs(x[1]), reverse=True):
                if feature_metrics[feat]['vif'] <= vif_threshold:
                    f.write(f"  - {feat}\n")
            
            f.write("\nKeep one from each correlated group:\n")
            high_vif_strong = [(feat, metrics) for feat, *_ in strong_features 
                             if feature_metrics[feat]['vif'] > vif_threshold]
            
            for feat, metrics in high_vif_strong:
                f.write(f"\nGroup containing {feat}:\n")
                correlations = [(other_feat, correlation_matrix.loc[feat, other_feat])
                              for other_feat in correlation_matrix.columns
                              if other_feat != feat and abs(correlation_matrix.loc[feat, other_feat]) > 0.5]
                for other_feat, corr in sorted(correlations, key=lambda x: abs(x[1]), reverse=True):
                    f.write(f"  - {other_feat} (correlation: {corr:.3f})\n")
        
        if unstable_features:
            f.write("\nConsider keeping after further investigation:\n")
            for feat, _, _, _, _ in sorted(unstable_features, key=lambda x: abs(x[1]), reverse=True):
                metrics = feature_metrics[feat]
                f.write(f"  - {feat} (effect: {metrics['effect_mean']:.3f}, ")
                f.write(f"stability: {metrics['effect_stability']:.3f})\n")
        
        if weak_features:
            f.write("\nConsider removing:\n")
            for feat, _, _, _, _ in sorted(weak_features, key=lambda x: abs(x[1])):
                f.write(f"  - {feat}\n")
        
        # Interaction recommendations if any exist
        interaction_features = [f for f in feature_metrics.keys() if "_" in f]
        if interaction_features:
            f.write("\n4.2 Interaction Recommendations\n")
            
            strong_interactions = []
            weak_interactions = []
            
            for feat in interaction_features:
                metrics = feature_metrics[feat]
                if abs(metrics['effect_mean']) > importance_threshold and metrics['effect_stability'] < variability_threshold:
                    strong_interactions.append((feat, metrics['effect_mean'], metrics['effect_stability']))
                else:
                    weak_interactions.append((feat, metrics['effect_mean'], metrics['effect_stability']))
            
            if strong_interactions:
                f.write("\nKeep these interactions:\n")
                for feat, effect, stab in sorted(strong_interactions, key=lambda x: abs(x[1]), reverse=True):
                    f.write(f"  - {feat} (effect: {effect:.3f}, stability: {stab:.3f})\n")
            
            if weak_interactions:
                f.write("\nConsider removing these interactions:\n")
                for feat, effect, stab in sorted(weak_interactions, key=lambda x: abs(x[1]), reverse=True):
                    f.write(f"  - {feat} (effect: {effect:.3f}, stability: {stab:.3f})\n")
        
        # Summary statistics
        f.write("\n5. Summary Statistics\n")
        f.write("-----------------\n")
        f.write(f"\nTotal features analyzed: {len(feature_metrics)}\n")
        f.write(f"Strong features: {len(strong_features)}\n")
        f.write(f"Unstable features: {len(unstable_features)}\n")
        f.write(f"Weak features: {len(weak_features)}\n")
        f.write(f"Redundant pairs: {len(redundant_pairs)}\n")
        f.write(f"High VIF features: {len(high_vif_features)}\n")
        
    logger.info("Analysis and recommendations completed")

def save_evaluation_results_to_file(
    results: Dict,
    output_dir: Path
) -> None:
    """
    Save feature evaluation results to files for later analysis.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "evaluation_results.pkl"
    
    # Convert any numpy arrays to lists for serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in value.items()
            }
        else:
            serializable_results[key] = (
                value.tolist() if isinstance(value, np.ndarray) else value
            )
    
    with open(results_file, 'wb') as f:
        pickle.dump(serializable_results, f)
    
    logger.info(f"Saved evaluation results to {results_file}")

def load_evaluation_results(output_dir: Path) -> Dict:
    """
    Load previously saved feature evaluation results.
    
    Args:
        output_dir: Directory containing saved results
        
    Returns:
        Dictionary containing evaluation results
    """
    results_file = output_dir / "evaluation_results.pkl"
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"No evaluation results found at {results_file}. "
            "Please run feature evaluation first."
        )
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded evaluation results from {results_file}")
    return results

def evaluate_features_only(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    config: Dict,
    output_dir: Path
) -> None:
    """Run feature evaluation with validation and basic reporting."""
    # Validate data first
    if not validate_data_for_evaluation(
        feature_matrix, 
        target_vector, 
        participants,
        config['validation']
    ):
        raise ValueError("Data validation failed")

    # Evaluate features
    results = evaluate_feature_sets(
        feature_matrix=feature_matrix,
        target_vector=target_vector,
        participants=participants,
        feature_sets=config['combinations'],
        output_dir=output_dir,
        n_splits=config['n_splits'],
        n_samples=config['n_samples'],
        chains=config['chains'],
        target_accept=config['target_accept']
    )
    
    # Save raw fold details if requested
    if config['fold_reporting']['save_fold_details']:
        fold_details_file = output_dir / "fold_details.json"
        with open(fold_details_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            fold_details = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results['fold_details'].items()
            }
            json.dump(fold_details, f, indent=2)
    
    # Save raw metrics if requested
    if config['fold_reporting']['save_raw_metrics']:
        metrics_file = output_dir / "raw_metrics.csv"
        pd.DataFrame(results['cv_metrics']).to_csv(metrics_file)
    
    # Save complete results for later analysis
    save_evaluation_results(results, output_dir)

def analyze_saved_results(
    output_dir: Path,
    feature_matrix: pd.DataFrame,
) -> None:
    """Load saved results and run analysis."""
    results = load_evaluation_results(output_dir)
    analyze_and_recommend(
        output_dir=output_dir,
        cv_metrics=results['cv_metrics'],
        feature_effects=results['feature_effects'],
        feature_sets=results['feature_sets'],
        feature_matrix=feature_matrix
    )

def validate_data_for_evaluation(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    validation_settings: Dict
) -> bool:
    """Validate data before feature evaluation."""
    n_samples = len(feature_matrix)
    
    if n_samples < validation_settings['min_training_samples']:
        logger.error(f"Insufficient samples: {n_samples} < {validation_settings['min_training_samples']}")
        return False
        
    if validation_settings['outlier_detection']:
        # Check for outliers in feature distributions
        for feature in feature_matrix.columns:
            z_scores = stats.zscore(feature_matrix[feature])
            outliers = abs(z_scores) > validation_settings['outlier_threshold']
            if outliers.sum() > 0:
                logger.warning(f"Found {outliers.sum()} outliers in feature {feature}")
    
    return True

# In bigram_pair_feature_evaluation.py

def evaluate_features_only(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    config: Dict,
    output_dir: Path
) -> None:
    """Run feature evaluation with validation and basic reporting."""
    # Validate data first
    if not validate_data_for_evaluation(
        feature_matrix, 
        target_vector, 
        participants,
        config['validation']
    ):
        raise ValueError("Data validation failed")

    # Evaluate features
    results = evaluate_feature_sets(
        feature_matrix=feature_matrix,
        target_vector=target_vector,
        participants=participants,
        feature_sets=config['combinations'],
        output_dir=output_dir,
        n_splits=config['n_splits'],
        n_samples=config['n_samples'],
        chains=config['chains'],
        target_accept=config['target_accept']
    )
    
    # Save raw metrics if requested - CHANGED THIS SECTION
    if config['fold_reporting']['save_raw_metrics']:
        metrics_file = output_dir / "raw_metrics.csv"
        cv_metrics_df = pd.DataFrame(results['cv_metrics'])
        cv_metrics_df.to_csv(metrics_file)
    
    # Save complete results for later analysis
    save_evaluation_results(results, output_dir)

def analyze_saved_results(
    output_dir: Path,
    feature_matrix: pd.DataFrame,
    analysis_config: Dict
) -> None:
    """Analyze saved evaluation results with detailed reporting."""
    # Load saved results
    results = load_evaluation_results(output_dir)
    
    # Create analysis output directory
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Analyze feature stability
    stability_results = analyze_feature_stability(
        results['feature_effects'],
        analysis_config['stability_thresholds']
    )
    
    # Analyze feature importance
    importance_results = analyze_feature_importance(
        results['cv_metrics'],
        analysis_config['feature_thresholds']
    )
    
    # Generate detailed reports based on reporting config
    if analysis_config['reporting']['detail_level'] == 'full':
        generate_full_report(
            stability_results,
            importance_results,
            results,
            analysis_dir,
            analysis_config
        )
    
    # Generate plots if requested
    if analysis_config['reporting']['plot_correlations']:
        plot_feature_correlations(
            feature_matrix,
            analysis_dir / "correlation_matrix.png"
        )
    
    if analysis_config['reporting']['plot_importance']:
        plot_feature_importance(
            importance_results,
            analysis_dir / "feature_importance.png"
        )

def generate_full_report(
    stability_results: Dict,
    importance_results: Dict,
    evaluation_results: Dict,
    output_dir: Path,
    config: Dict
) -> None:
    """Generate comprehensive analysis report."""
    report_file = output_dir / "full_analysis_report.txt"
    
    with open(report_file, 'w') as f:
        # Write report header
        f.write("Feature Analysis Report\n")
        f.write("=====================\n\n")
        
        # Stability Analysis
        f.write("1. Feature Stability Analysis\n")
        f.write("--------------------------\n")
        for feature, metrics in stability_results.items():
            f.write(f"\n{feature}:\n")
            f.write(f"  Stability Score: {metrics['stability_score']:.3f}")
            if metrics['stability_score'] < config['stability_thresholds']['feature_reliability']:
                f.write(" (WARNING: Low stability)")
            f.write(f"\n  Effect Size: {metrics['effect_size']:.3f}")
            f.write(f"\n  R² Score: {metrics['r2']:.3f}\n")
        
        # Importance Analysis
        f.write("\n2. Feature Importance Analysis\n")
        f.write("---------------------------\n")
        for feature, metrics in importance_results.items():
            f.write(f"\n{feature}:\n")
            f.write(f"  Importance Score: {metrics['importance']:.3f}")
            if metrics['importance'] < config['feature_thresholds']['importance']:
                f.write(" (WARNING: Low importance)")
            f.write(f"\n  Correlation: {metrics['correlation']:.3f}\n")
        
        # Recommendations
        f.write("\n3. Recommendations\n")
        f.write("----------------\n")
        write_recommendations(f, stability_results, importance_results, config)


