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
import yaml
import logging
from dataclasses import dataclass
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pytensor.tensor as pt 

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

def calculate_univariate_r2(x: pd.Series, y: np.ndarray) -> float:
    """Calculate R² for a single feature."""
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def evaluate_feature_sets(
   feature_matrix: pd.DataFrame,
   target_vector: np.ndarray,
   participants: np.ndarray,
   feature_sets: List[Dict[str, Any]],
   output_dir: Path,
   n_splits: int = 5,
   n_samples: int = 10000,
   chains: int = 8,
   target_accept: float = 0.85
) -> Dict[str, List[float]]:
    """
    Evaluate different feature sets using cross-validation.
    """
    results = {}
    all_cv_metrics = {}
    all_feature_effects = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Updated file paths with more descriptive names
    raw_metrics_path = output_dir / "raw_metrics.txt"
    feature_details_path = output_dir / "feature_details.txt"
    
    logger.info("Starting feature set evaluation")
    logger.info(f"Number of feature sets: {len(feature_sets)}")
    logger.info(f"Target vector shape: {target_vector.shape}")
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    with open(raw_metrics_path, 'w') as raw_file, \
         open(feature_details_path, 'w') as details_file:
        
        raw_file.write("Raw Cross-Validation Metrics\n")
        raw_file.write("==========================\n\n")
        
        details_file.write("Detailed Feature Analysis\n")
        details_file.write("=======================\n")
        
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
            X = feature_matrix[features].copy()
            
            # Add interaction terms
            valid_interactions = []
            for interaction in interactions:
                if all(feat in features for feat in interaction):
                    interaction_name = "_".join(interaction)
                    X[interaction_name] = X[interaction[0]].copy()
                    for feat in interaction[1:]:
                        X[interaction_name] *= X[feat]
                    features.append(interaction_name)
                    valid_interactions.append(interaction)
                    logger.info(f"Added interaction: {' × '.join(interaction)}")
            
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
                
                # Split data
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_train = target_vector[train_idx]
                y_val = target_vector[val_idx]
                participants_train = participants[train_idx]
                participants_val = participants[val_idx]
                
                # Calculate raw feature relationships
                for feature in features:
                    # Train set relationships
                    train_corr, train_p = stats.spearmanr(X_train[feature], y_train)
                    train_r2 = calculate_univariate_r2(X_train[feature], y_train)
                    
                    # Validation set relationships
                    val_corr, val_p = stats.spearmanr(X_val[feature], y_val)
                    val_r2 = calculate_univariate_r2(X_val[feature], y_val)
                    
                    # Write detailed feature metrics
                    details_file.write(f"\n  {feature}:\n")
                    details_file.write(f"    Train correlation: {train_corr:.3f} (p={train_p:.3e})\n")
                    details_file.write(f"    Train R²: {train_r2:.3f}\n")
                    details_file.write(f"    Val correlation: {val_corr:.3f} (p={val_p:.3e})\n")
                    details_file.write(f"    Val R²: {val_r2:.3f}\n")
                    
                    raw_feature_data = {
                        'train_correlation': train_corr,
                        'train_correlation_p': train_p,
                        'train_r2': train_r2,
                        'val_correlation': val_corr,
                        'val_correlation_p': val_p,
                        'val_r2': val_r2,
                        'train_values': X_train[feature].values,
                        'train_target': y_train,
                        'val_values': X_val[feature].values,
                        'val_target': y_val
                    }
                    feature_effects_all_folds[feature].append(raw_feature_data)
                
                # Train model
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
                        fold_data = feature_effects_all_folds[feature][-1]
                        fold_data.update({
                            'model_effect_mean': float(feature_effects.loc[feature, 'mean']),
                            'model_effect_sd': float(feature_effects.loc[feature, 'sd']),
                            'hdi_3%': float(feature_effects.loc[feature, 'hdi_3%']),
                            'hdi_97%': float(feature_effects.loc[feature, 'hdi_97%'])
                        })
                        details_file.write(f"    Model effect: {fold_data['model_effect_mean']:.3f} ± {fold_data['model_effect_sd']:.3f}\n")
                except Exception as e:
                    logger.warning(f"Could not compute feature effects for fold {fold}: {str(e)}")
                
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
            
            # Write feature summary statistics
            details_file.write("\nFeature Summary Statistics:\n")
            details_file.write("=" * 30 + "\n")
            
            for feature in features:
                effects_data = feature_effects_all_folds[feature]
                
                details_file.write(f"\n{feature}:\n")
                
                # Model effects
                model_effects = [d.get('model_effect_mean', np.nan) for d in effects_data]
                if not all(np.isnan(model_effects)):
                    details_file.write("Model Effects:\n")
                    details_file.write(f"  Mean: {np.nanmean(model_effects):.3f} ± {np.nanstd(model_effects):.3f}\n")
                
                # Correlations
                train_corrs = [d['train_correlation'] for d in effects_data]
                val_corrs = [d['val_correlation'] for d in effects_data]
                details_file.write("Correlations:\n")
                details_file.write(f"  Train: {np.mean(train_corrs):.3f} ± {np.std(train_corrs):.3f}\n")
                details_file.write(f"  Validation: {np.mean(val_corrs):.3f} ± {np.std(val_corrs):.3f}\n")
                
                # R² values
                train_r2s = [d['train_r2'] for d in effects_data]
                val_r2s = [d['val_r2'] for d in effects_data]
                details_file.write("R² Values:\n")
                details_file.write(f"  Train: {np.mean(train_r2s):.3f} ± {np.std(train_r2s):.3f}\n")
                details_file.write(f"  Validation: {np.mean(val_r2s):.3f} ± {np.std(val_r2s):.3f}\n")
            
            # Calculate average metrics
            avg_metrics = {metric: np.mean(values) for metric, values in cv_metrics.items()}
            std_metrics = {metric: np.std(values) for metric, values in cv_metrics.items()}
            
            # Store results
            results[set_name] = avg_metrics
            all_cv_metrics[set_name] = cv_metrics
            all_feature_effects[set_name] = feature_effects_all_folds
    
    # Debug logging for feature effects
    logger.info("=== Feature Effects Debug ===")
    for set_name, effects in all_feature_effects.items():
        logger.info(f"\nSet: {set_name}")
        for feature, effect_data in effects.items():
            mean_effect = np.mean([d.get('model_effect_mean', 0) for d in effect_data])
            mean_corr = np.mean([d.get('val_correlation', 0) for d in effect_data])
            mean_r2 = np.mean([d.get('val_r2', 0) for d in effect_data])
            logger.info(f"Feature {feature}:")
            logger.info(f"  Effect: {mean_effect:.3f}")
            logger.info(f"  Correlation: {mean_corr:.3f}")
            logger.info(f"  R²: {mean_r2:.3f}")
    
    logger.info("Feature evaluation complete")
    
    # Run analyze_and_recommend with complete data
    analyze_and_recommend(
        output_dir=output_dir,
        cv_metrics=all_cv_metrics,
        feature_effects=all_feature_effects,
        feature_sets=feature_sets,
        feature_matrix=feature_matrix
    )
    
    return {
        'cv_metrics': all_cv_metrics,
        'feature_effects': all_feature_effects,
        'feature_sets': feature_sets
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
def save_evaluation_results(
    cv_scores: Dict[str, List[float]],
    waic_scores: Dict[str, float],
    loo_scores: Dict[str, float],
    feature_importance: Dict[str, float],
    feature_correlations: pd.DataFrame,
    feature_combinations: List[Dict[str, Any]], 
    output_dir: Path,
    multicollinearity_results: Optional[Dict] = None
) -> None:
    """
    Save comprehensive evaluation results to files.
    
    Saves the following files in output_dir:
    1. evaluation_metrics.csv - Summary metrics for each feature set
    2. cv_scores_detailed.csv - Detailed cross-validation scores
    3. feature_importance.txt/.csv - Feature importance rankings
    4. feature_correlations.txt/.csv - Correlation analysis
    5. multicollinearity.txt - VIF and correlation warnings
    6. evaluation_summary.txt - Overall analysis summary
    
    Args:
        cv_scores: Cross-validation scores per feature set
        waic_scores: WAIC scores per feature set
        loo_scores: LOO-CV scores per feature set
        feature_importance: Feature importance scores
        feature_correlations: Feature correlation matrix
        feature_combinations: List of evaluated feature combinations
        output_dir: Directory for output files
        multicollinearity_results: Optional VIF analysis results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving evaluation results to {output_dir}")
    
    try:
        # 1. Create and save main results DataFrame
        results_df = pd.DataFrame({
            'feature_set': list(cv_scores.keys()),
            'mean_cv_score': [np.nanmean(scores) if scores and any(~np.isnan(scores)) else np.nan 
                             for scores in cv_scores.values()],
            'std_cv_score': [np.nanstd(scores) if scores and any(~np.isnan(scores)) else np.nan 
                            for scores in cv_scores.values()],
            'waic_score': [waic_scores.get(fs, np.nan) for fs in cv_scores.keys()],
            'loo_score': [loo_scores.get(fs, np.nan) for fs in cv_scores.keys()]
        })
        results_df = results_df.sort_values('mean_cv_score', ascending=False, na_position='last')
        results_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
        
        # 2. Save detailed CV scores
        cv_details_df = pd.DataFrame(cv_scores)
        cv_details_df.index.name = 'fold'
        cv_details_df.to_csv(output_dir / 'cv_scores_detailed.csv')
        
        # 3. Save feature importance analysis
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
        
        with open(output_dir / 'feature_importance.txt', 'w') as f:
            f.write("=== Feature Importance Analysis ===\n\n")
            for _, row in importance_df.iterrows():
                f.write(f"{row['feature']:<20} {row['importance']:.4f}\n")
        
        # 4. Save correlation analysis
        feature_correlations.to_csv(output_dir / 'feature_correlations.csv')
        
        with open(output_dir / 'feature_correlations.txt', 'w') as f:
            f.write("=== Feature Correlation Analysis ===\n\n")
            high_correlations = []
            for i in range(len(feature_correlations.columns)):
                for j in range(i+1, len(feature_correlations.columns)):
                    corr = abs(feature_correlations.iloc[i, j])
                    if corr >= 0.7:
                        high_correlations.append({
                            'feature1': feature_correlations.columns[i],
                            'feature2': feature_correlations.columns[j],
                            'correlation': corr
                        })
            
            if high_correlations:
                f.write("High Correlations (|r| >= 0.7):\n")
                for corr in sorted(high_correlations, key=lambda x: x['correlation'], reverse=True):
                    f.write(f"{corr['feature1']} - {corr['feature2']}: {corr['correlation']:.3f}\n")
            else:
                f.write("No high correlations found (|r| >= 0.7)\n")
        
        # 5. Save multicollinearity results if available
        if multicollinearity_results:
            with open(output_dir / 'multicollinearity.txt', 'w') as f:
                f.write("=== Multicollinearity Analysis ===\n\n")
                
                # VIF results
                f.write("Variance Inflation Factors:\n")
                vif_data = sorted(multicollinearity_results['vif'], 
                                key=lambda x: x['VIF'], 
                                reverse=True)
                
                for vif in vif_data:
                    f.write(f"{vif['Feature']:<20} VIF: {vif['VIF']:>8.3f} ({vif['Status']})\n")
                
                if multicollinearity_results['high_correlations']:
                    f.write("\nHigh Correlations (r > 0.7):\n")
                    for corr in sorted(multicollinearity_results['high_correlations'],
                                     key=lambda x: x['Correlation'],
                                     reverse=True):
                        f.write(f"{corr['Feature1']} - {corr['Feature2']}: {corr['Correlation']:.3f}\n")
        
        # 6. Create comprehensive summary report
        with open(output_dir / 'evaluation_summary.txt', 'w') as f:
            f.write("=== Feature Evaluation Summary ===\n\n")
            
            f.write("Feature Set Performance Overview:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total feature sets evaluated: {len(cv_scores)}\n")
            f.write(f"Cross-validation folds: {len(next(iter(cv_scores.values())))}\n\n")
            
            # Best performing combinations
            f.write("Top Performing Feature Sets:\n")
            f.write("-" * 50 + "\n")
            for i, row in results_df.head(3).iterrows():
                f.write(f"\n{i+1}. {row['feature_set']}\n")
                f.write(f"   Mean CV Score: {row['mean_cv_score']:.4f} ± {row['std_cv_score']:.4f}\n")
                f.write(f"   WAIC: {row['waic_score']:.2f}\n")
                f.write(f"   LOO: {row['loo_score']:.2f}\n")
            
            # Feature importance summary
            f.write("\nMost Important Features:\n")
            f.write("-" * 50 + "\n")
            for _, row in importance_df.head(5).iterrows():
                f.write(f"{row['feature']:<20} {row['importance']:.4f}\n")
            
            # Correlation and multicollinearity summary
            f.write("\nFeature Correlation Summary:\n")
            f.write("-" * 50 + "\n")
            if high_correlations:
                f.write(f"Found {len(high_correlations)} highly correlated feature pairs\n")
                f.write("Top 3 highest correlations:\n")
                for corr in sorted(high_correlations, 
                                 key=lambda x: x['correlation'], 
                                 reverse=True)[:3]:
                    f.write(f"- {corr['feature1']} - {corr['feature2']}: {corr['correlation']:.3f}\n")
            else:
                f.write("No concerning feature correlations found\n")
            
            # Multicollinearity concerns
            if multicollinearity_results:
                f.write("\nMulticollinearity Analysis:\n")
                f.write("-" * 50 + "\n")
                high_vif = [x for x in multicollinearity_results['vif'] if x['VIF'] > 5]
                if high_vif:
                    f.write("Features with high VIF (>5):\n")
                    for vif in sorted(high_vif, key=lambda x: x['VIF'], reverse=True):
                        f.write(f"- {vif['Feature']}: {vif['VIF']:.2f}\n")
                else:
                    f.write("No concerning VIF values found\n")
            
            # Final recommendations
            f.write("\nRecommendations:\n")
            f.write("-" * 50 + "\n")
            best_set = results_df.iloc[0]
            f.write(f"1. Best feature set: {best_set['feature_set']}\n")
            f.write(f"   Performance: {best_set['mean_cv_score']:.4f} ± {best_set['std_cv_score']:.4f}\n")
            
            if high_vif:
                f.write("\n2. Consider removing features with high multicollinearity:\n")
                for vif in sorted(high_vif, key=lambda x: x['VIF'], reverse=True)[:3]:
                    f.write(f"   - {vif['Feature']} (VIF: {vif['VIF']:.2f})\n")
            
            f.write("\n3. Most influential features to retain:\n")
            for _, row in importance_df.head(3).iterrows():
                f.write(f"   - {row['feature']} (importance: {row['importance']:.4f})\n")
        
        logger.info("Successfully saved all evaluation results")
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")
        raise

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