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
import pymc as pm
import arviz as az
from scipy import stats
import logging
from dataclasses import dataclass
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
   
   Args:
       feature_matrix: Matrix of feature values
       target_vector: Target values
       participants: Participant IDs for grouping
       feature_sets: List of dictionaries containing:
           - name: Name of the feature set
           - features: List of features to include
           - interactions: List of feature lists to test for interactions
       output_dir: Directory for output files
       n_splits: Number of cross-validation splits
       n_samples: Number of MCMC samples
       chains: Number of independent MCMC chains
       target_accept: Target acceptance rate for proposals in the NUTS sampler
   """
   results = {}
   all_cv_metrics = {}
   all_feature_effects = {}

   output_dir.mkdir(parents=True, exist_ok=True)
   
   summary_path = output_dir / "feature_evaluation_summary.txt"
   metrics_path = output_dir / "feature_set_metrics.txt"
   importance_path = output_dir / "feature_importance.txt"
   
   with open(summary_path, 'w') as summary_file, \
        open(metrics_path, 'w') as metrics_file, \
        open(importance_path, 'w') as importance_file:
        
       summary_file.write("Feature Set Evaluation Summary\n")
       summary_file.write("===========================\n\n")
       
       metrics_file.write("Detailed Cross-Validation Metrics\n")
       metrics_file.write("==============================\n\n")
       
       importance_file.write("Feature Importance Analysis\n")
       importance_file.write("========================\n\n")
       
       for feature_set in feature_sets:
           set_name = feature_set['name']
           features = feature_set['features'].copy()
           interactions = feature_set.get('interactions', [])
           
           logger.info(f"Evaluating {set_name}: {features}")
           logger.info(f"Testing interactions: {interactions}")
           
           summary_file.write(f"\nFeature Set: {set_name}\n")
           summary_file.write("-" * 50 + "\n")
           summary_file.write("Base Features: " + ", ".join(features) + "\n")
           if interactions:
               summary_file.write("Interactions: " + 
                                ", ".join(["×".join(interaction) for interaction in interactions]) + "\n")
           summary_file.write("\n")
           
           # Select features and create interaction terms
           X = feature_matrix[features].copy()
           
           # Add interaction terms only if all features are present
           valid_interactions = []
           for interaction in interactions:
               if all(feat in features for feat in interaction):
                   interaction_name = "_".join(interaction)
                   # Create interaction term as product of all features
                   X[interaction_name] = X[interaction[0]].copy()
                   for feat in interaction[1:]:
                       X[interaction_name] *= X[feat]
                   features.append(interaction_name)
                   valid_interactions.append(interaction)
               else:
                   logger.warning(f"Skipping interaction {' × '.join(interaction)} - "
                               f"not all features present in {set_name}")
           
           # Create storage dictionary for feature effects
           feature_effects_all_folds = {feature: [] for feature in features}

           # Cross-validation
           logger.info("Starting cross-validation")
           cv_metrics = {
               'r2': [], 'rmse': [], 'mae': [], 
               'correlation': [], 'accuracy': []
           }
           
           # Create cross-validation splits
           group_kfold = GroupKFold(n_splits=n_splits)
           
           for fold, (train_idx, val_idx) in enumerate(
               group_kfold.split(X, target_vector, participants)
           ):
               # Split data
               X_train = X.iloc[train_idx]
               y_train = target_vector[train_idx]
               X_val = X.iloc[val_idx]
               y_val = target_vector[val_idx]
               participants_train = participants[train_idx]
               
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
               
               # Evaluate
               metrics = evaluate_model_performance(
                   trace=trace,
                   feature_matrix=X_val,
                   target_vector=y_val,
                   participants=participants[val_idx],
                   design_features=features,
                   control_features=[]
               )
               
               # Record metrics
               for metric, value in metrics.items():
                   cv_metrics[metric].append(value)
               
               # Write fold results
               metrics_file.write(f"Feature Set {set_name}, Fold {fold}:\n")
               for metric, value in metrics.items():
                   metrics_file.write(f"  {metric}: {value:.3f}\n")
               metrics_file.write("\n")
                               
               # Store feature effects for each fold
               try:
                   feature_effects = az.summary(trace, var_names=features)
                   for feature in features:
                       feature_effects_all_folds[feature].append(feature_effects.loc[feature])
               except Exception as e:
                   logger.warning(f"Could not compute feature effects for fold {fold}: {str(e)}")

           # Write aggregate feature importance
           importance_file.write(f"\nFeature Set {set_name} Feature Importance (across {n_splits} folds):\n")
           importance_file.write("-" * 50 + "\n")
           for feature in features:
               effects = pd.DataFrame(feature_effects_all_folds[feature])
               importance_file.write(f"\n{feature}:\n")
               importance_file.write("Mean effects:\n")
               importance_file.write(effects.mean().to_string() + "\n")
               importance_file.write("\nStd across folds:\n")
               importance_file.write(effects.std().to_string() + "\n")
               importance_file.write("\nEffects by fold:\n")
               importance_file.write(effects.to_string() + "\n")
           importance_file.write("\n" + "="*50 + "\n")

           # Add interaction-specific analysis
           if valid_interactions:
               importance_file.write(f"\nFeature Interactions for {set_name}:\n")
               importance_file.write("-" * 50 + "\n")
               for interaction in valid_interactions:
                   interaction_name = "_".join(interaction)
                   effects = pd.DataFrame(feature_effects_all_folds[interaction_name])
                   importance_file.write(f"\n{' × '.join(interaction)}:\n")
                   importance_file.write("Mean interaction effect:\n")
                   importance_file.write(effects.mean().to_string() + "\n")
                   importance_file.write("\nStd across folds:\n")
                   importance_file.write(effects.std().to_string() + "\n")
                   
           # Calculate average metrics
           avg_metrics = {
               metric: np.mean(values) for metric, values in cv_metrics.items()
           }
           std_metrics = {
               metric: np.std(values) for metric, values in cv_metrics.items()
           }
           
           # Write summary results
           summary_file.write("Average Metrics:\n")
           for metric, mean_value in avg_metrics.items():
               std_value = std_metrics[metric]
               summary_file.write(f"{metric}: {mean_value:.3f} ± {std_value:.3f}\n")
           summary_file.write("\n")
           
           # Create visualization
           plt.figure(figsize=(10, 6))
           plt.boxplot([cv_metrics[m] for m in ['r2', 'correlation', 'accuracy']], 
                      labels=['R²', 'Correlation', 'Accuracy'])
           plt.title(f'Feature Set {set_name} Performance')
           plt.ylabel('Score')
           plt.savefig(output_dir / f"feature_set_{set_name}_performance.png")
           plt.close()
   
           # Store results with meaningful names
           results[set_name] = avg_metrics
           all_cv_metrics[set_name] = cv_metrics
           all_feature_effects[set_name] = feature_effects_all_folds
   
   # Analyze results
   analyze_and_recommend(
       output_dir=output_dir,
       cv_metrics=all_cv_metrics,
       feature_effects=all_feature_effects,
       feature_sets=feature_sets
   )
   
   return results

def perform_cross_validation(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    n_splits: int = 5
) -> List[float]:
    """
    Perform cross-validation to evaluate feature combinations while preserving participant grouping.
    Uses R² (R-squared) scores to assess predictive performance.
    
    R² measures the proportion of variance in comfort/preference scores that is predictable from 
    the features. It ranges from:
    - 1.0: Perfect prediction - features explain all variation in preferences
    - 0.0: Model only predicts as well as always guessing the mean preference
    - <0.0: Model performs worse than guessing the mean
    
    Args:
        feature_matrix: DataFrame of features to evaluate
        target_vector: Array of comfort/preference scores
        participants: Array of participant IDs/indices for grouped validation
        n_splits: Number of cross-validation folds
    
    Returns:
        List[float]: R² scores for each fold
    """
    cv_scores = []
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Convert to numpy and ensure correct shapes
    X = np.atleast_2d(feature_matrix.to_numpy())
    y = np.ravel(target_vector)
    groups = np.ravel(participants)
    
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("Inconsistent dimensions in input data")
    
    logger.info(f"Starting {n_splits}-fold cross-validation")
    logger.info(f"Data shapes - X: {X.shape}, y: {y.shape}, groups: {groups.shape}")
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups), 1):
        try:
            # Prepare data
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            with pm.Model() as model:
                # Set up priors
                feature_effects = {
                    f'feat_{i}': pm.Normal(f'feat_{i}', mu=0, sigma=1)
                    for i in range(X_train.shape[1])
                }
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # Expected value
                mu = sum(feature_effects[f'feat_{i}'] * X_train_scaled[:, i] 
                        for i in range(X_train.shape[1]))
                
                # Likelihood
                likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_train)
                
                # Sample
                trace = pm.sample(
                    draws=5000,
                    tune=2000,
                    chains=4,
                    target_accept=0.85,
                    init='jitter+adapt_diag',
                    return_inferencedata=True,
                    cores=4,
                    random_seed=42
                )
            
            # Make predictions
            feature_effects_mean = {
                f'feat_{i}': float(trace.posterior[f'feat_{i}'].mean())
                for i in range(X_train.shape[1])
            }
            y_pred = sum(feature_effects_mean[f'feat_{i}'] * X_val_scaled[:, i]
                        for i in range(X_val.shape[1]))
            
            # Calculate R² score
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            logger.info(f"Fold {fold}/{n_splits} R² score: {r2:.3f}")
            cv_scores.append(r2)
            
        except Exception as e:
            logger.error(f"Failed fold {fold}/{n_splits}: {e}")
            cv_scores.append(np.nan)
    
    mean_score = np.nanmean(cv_scores)
    logger.info(f"Cross-validation complete. Mean R²: {mean_score:.3f}")
    return cv_scores

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
    correlation_threshold: float = 0.7,
    importance_threshold: float = 0.1,
    variability_threshold: float = 0.5
) -> None:
    """
    Analyze evaluation results and provide feature recommendations.
    
    Args:
        output_dir: Directory containing evaluation results
        cv_metrics: Dictionary mapping set names to their cross-validation metrics
        feature_effects: Dictionary mapping set names to their feature effects
        feature_sets: List of feature set configurations
        correlation_threshold: Threshold for concerning feature correlations
        importance_threshold: Threshold for feature importance significance
        variability_threshold: Threshold for concerning effect variability
    """
    analysis_path = output_dir / "feature_recommendations.txt"
    
    with open(analysis_path, 'w') as f:
        f.write("Feature Analysis and Recommendations\n")
        f.write("=================================\n\n")
        
        # 1. Analyze Feature Set Performance
        f.write("1. Feature Set Performance\n")
        f.write("-----------------------\n")
        
        # Calculate and sort mean performance
        feature_set_performance = {}
        for set_name, metrics in cv_metrics.items():
            mean_r2 = np.mean(metrics['r2'])
            std_r2 = np.std(metrics['r2'])
            feature_set_performance[set_name] = {
                'r2_mean': mean_r2,
                'r2_std': std_r2,
                'correlation_mean': np.mean(metrics['correlation']),
                'correlation_std': np.std(metrics['correlation']),
                'rmse_mean': np.mean(metrics['rmse']),
                'rmse_std': np.std(metrics['rmse']),
                'accuracy_mean': np.mean(metrics['accuracy']),
                'accuracy_std': np.std(metrics['accuracy'])
            }
        
        # Sort by R² score
        sorted_sets = sorted(feature_set_performance.items(), 
                           key=lambda x: x[1]['r2_mean'], 
                           reverse=True)
        
        # Report all sets
        f.write("\nAll Feature Sets (sorted by R²):\n")
        for set_name, perf in sorted_sets:
            f.write(f"\n{set_name}:\n")
            f.write(f"  R² = {perf['r2_mean']:.3f} ± {perf['r2_std']:.3f}\n")
            f.write(f"  Correlation = {perf['correlation_mean']:.3f} ± {perf['correlation_std']:.3f}\n")
            f.write(f"  RMSE = {perf['rmse_mean']:.3f} ± {perf['rmse_std']:.3f}\n")
            f.write(f"  Accuracy = {perf['accuracy_mean']:.3f} ± {perf['accuracy_std']:.3f}\n")

        # Performance improvement analysis
        f.write("\nPerformance Improvement Analysis:\n")
        baseline_r2 = sorted_sets[-1][1]['r2_mean']  # Worst performing set as baseline
        for set_name, perf in sorted_sets[:-1]:  # Skip baseline
            improvement = (perf['r2_mean'] - baseline_r2) / baseline_r2 * 100
            f.write(f"{set_name}: {improvement:.1f}% improvement over baseline\n")
        
        # 2. Individual Feature Analysis
        f.write("\n2. Individual Feature Analysis\n")
        f.write("---------------------------\n")
        
        # Analyze each feature across all sets it appears in
        all_features = set()
        feature_performance = {}
        
        for set_name, effects in feature_effects.items():
            for feature, effect_data in effects.items():
                if not feature.endswith('_interaction'):  # Skip interaction terms for now
                    all_features.add(feature)
                    if feature not in feature_performance:
                        feature_performance[feature] = []
                    
                    effects_df = pd.DataFrame(effect_data)
                    mean_effect = effects_df['mean'].mean()
                    effect_std = effects_df['mean'].std()
                    
                    feature_performance[feature].append({
                        'set_name': set_name,
                        'mean_effect': mean_effect,
                        'effect_std': effect_std,
                        'stability': effect_std / abs(mean_effect) if abs(mean_effect) > 0 else float('inf'),
                        'consistent_sign': np.all(np.sign(effects_df['mean']) == 
                                               np.sign(effects_df['mean'].iloc[0]))
                    })
        
        # Categorize features
        strong_features = []
        weak_features = []
        unstable_features = []
        
        for feature in all_features:
            # Average metrics across sets
            metrics = feature_performance[feature]
            avg_effect = np.mean([m['mean_effect'] for m in metrics])
            avg_stability = np.mean([m['stability'] for m in metrics])
            consistent_across_sets = all(m['consistent_sign'] for m in metrics)
            
            if abs(avg_effect) > importance_threshold:
                if avg_stability < variability_threshold and consistent_across_sets:
                    strong_features.append((feature, avg_effect, avg_stability))
                else:
                    unstable_features.append((feature, avg_effect, avg_stability))
            else:
                weak_features.append((feature, avg_effect, avg_stability))
        
        # Report feature categories
        f.write("\nStrong Features (Keep):\n")
        for feature, effect, stability in sorted(strong_features, 
                                               key=lambda x: abs(x[1]), 
                                               reverse=True):
            f.write(f"{feature}:\n")
            f.write(f"  Average Effect: {effect:.3f}\n")
            f.write(f"  Stability: {stability:.3f}\n")
            f.write("  Performance across sets:\n")
            for metric in feature_performance[feature]:
                f.write(f"    {metric['set_name']}: {metric['mean_effect']:.3f} ± {metric['effect_std']:.3f}\n")
        
        f.write("\nUnstable Features (Review):\n")
        for feature, effect, stability in unstable_features:
            f.write(f"{feature}:\n")
            f.write(f"  Average Effect: {effect:.3f}\n")
            f.write(f"  Stability: {stability:.3f}\n")
            f.write("  Performance across sets:\n")
            for metric in feature_performance[feature]:
                f.write(f"    {metric['set_name']}: {metric['mean_effect']:.3f} ± {metric['effect_std']:.3f}\n")
        
        f.write("\nWeak Features (Consider Removing):\n")
        for feature, effect, stability in weak_features:
            f.write(f"{feature}:\n")
            f.write(f"  Average Effect: {effect:.3f}\n")
            
        # 3. Interaction Analysis
        f.write("\n3. Feature Interactions\n")
        f.write("---------------------\n")
        
        # Analyze interactions across all sets
        interaction_performance = {}
        
        for set_name, effects in feature_effects.items():
            feature_set = next(fs for fs in feature_sets if fs['name'] == set_name)
            interactions = feature_set.get('interactions', [])
            
            for feat1, feat2 in interactions:
                interaction_name = f"{feat1}_{feat2}"
                if interaction_name in effects:
                    if interaction_name not in interaction_performance:
                        interaction_performance[interaction_name] = []
                        
                    effects_df = pd.DataFrame(effects[interaction_name])
                    mean_effect = effects_df['mean'].mean()
                    effect_std = effects_df['mean'].std()
                    
                    interaction_performance[interaction_name].append({
                        'set_name': set_name,
                        'mean_effect': mean_effect,
                        'effect_std': effect_std,
                        'features': (feat1, feat2)
                    })
        
        if interaction_performance:
            f.write("\nInteraction Effects:\n")
            for interaction, metrics in interaction_performance.items():
                feat1, feat2 = metrics[0]['features']
                avg_effect = np.mean([m['mean_effect'] for m in metrics])
                
                f.write(f"\n{feat1} × {feat2}:\n")
                f.write(f"  Average Effect: {avg_effect:.3f}\n")
                f.write("  Performance across sets:\n")
                for metric in metrics:
                    f.write(f"    {metric['set_name']}: {metric['mean_effect']:.3f} ± {metric['effect_std']:.3f}\n")
                
                if abs(avg_effect) > importance_threshold:
                    f.write("  Recommendation: Keep interaction\n")
                else:
                    f.write("  Recommendation: Remove interaction\n")
        
        # 4. Final Recommendations
        f.write("\n4. Final Recommendations\n")
        f.write("----------------------\n")
        
        # Best feature set
        best_set_name = sorted_sets[0][0]
        best_set_perf = sorted_sets[0][1]
        f.write(f"\n1. Recommended Base Feature Set: {best_set_name}\n")
        f.write(f"   R² = {best_set_perf['r2_mean']:.3f} ± {best_set_perf['r2_std']:.3f}\n")
        
        # Feature modifications
        f.write("\n2. Feature Recommendations:\n")
        
        f.write("\n   Keep these features (strong, stable effects):\n")
        for feature, effect, _ in strong_features:
            f.write(f"   - {feature} (effect: {effect:.3f})\n")
        
        if unstable_features:
            f.write("\n   Review these features (important but unstable):\n")
            for feature, effect, stability in unstable_features:
                f.write(f"   - {feature} (effect: {effect:.3f}, stability: {stability:.3f})\n")
        
        if weak_features:
            f.write("\n   Consider removing these features (weak effects):\n")
            for feature, effect, _ in weak_features:
                f.write(f"   - {feature} (effect: {effect:.3f})\n")
        
        # Interaction recommendations
        if interaction_performance:
            f.write("\n3. Interaction Recommendations:\n")
            
            keep_interactions = []
            remove_interactions = []
            
            for interaction, metrics in interaction_performance.items():
                avg_effect = np.mean([m['mean_effect'] for m in metrics])
                if abs(avg_effect) > importance_threshold:
                    keep_interactions.append((interaction, avg_effect))
                else:
                    remove_interactions.append((interaction, avg_effect))
            
            if keep_interactions:
                f.write("\n   Keep these interactions:\n")
                for interaction, effect in sorted(keep_interactions, key=lambda x: abs(x[1]), reverse=True):
                    f.write(f"   - {interaction} (effect: {effect:.3f})\n")
            
            if remove_interactions:
                f.write("\n   Remove these interactions:\n")
                for interaction, effect in remove_interactions:
                    f.write(f"   - {interaction} (effect: {effect:.3f})\n")
        
        logger.info(f"Analysis and recommendations written to {analysis_path}")

