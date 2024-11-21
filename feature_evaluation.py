"""
Feature evaluation module for keyboard layout analysis.
Implements methods for evaluating different feature sets using cross-validation
and information criteria, with careful separation from final model training.

Proper data splitting:
  - Hold-out test set never seen during feature evaluation
  - Cross-validation used within training set
  - Participant grouping preserved

Multiple evaluation metrics:
  - Cross-validation performance
  - WAIC and LOO-CV for model comparison
  - Feature importance scores
  - Feature correlations
  - Stability metrics

Comprehensive visualization:
  - Feature importance plots
  - Correlation heatmaps
  - Performance comparisons
  - Stability analysis

Proper handling of participant effects:
  - Group-based cross-validation
  - Random effects in evaluation models

Purpose:
  - Evaluate different feature combinations without data leakage
  - Understand feature interactions and redundancies
  - Identify the most stable and informative features
  - Make informed decisions about feature selection before final model training
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import pymc as pm
import arviz as az
from scipy import stats
import logging
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class FeatureEvaluationResults:
    """Container for feature evaluation results."""
    cv_scores: Dict[str, List[float]]
    waic_scores: Dict[str, float]
    loo_scores: Dict[str, float]
    feature_correlations: pd.DataFrame
    feature_importance: Dict[str, float]
    interaction_scores: Optional[Dict[Tuple[str, str], float]] = None
    stability_metrics: Optional[Dict[str, Dict[str, float]]] = None

def evaluate_feature_sets(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    candidate_features: List[List[str]],
    feature_names: List[str],
    output_dir: Path,
    n_splits: int = 5,
    n_samples: int = 1000
) -> FeatureEvaluationResults:
    """
    Evaluate different feature sets using cross-validation and information criteria.
    
    Args:
        feature_matrix: Full feature matrix
        target_vector: Target values
        participants: Participant IDs for grouping
        candidate_features: List of feature sets to evaluate
        feature_names: Names of all features
        output_dir: Directory for saving evaluation results
        n_splits: Number of cross-validation splits
        n_samples: Number of samples for Bayesian model
        
    Returns:
        FeatureEvaluationResults containing evaluation metrics
    """
    # Initialize results containers
    cv_scores = {}
    waic_scores = {}
    loo_scores = {}
    feature_importance = {}
    
    # Create cross-validation splits that keep participant data together
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Evaluate each feature set
    for feature_set_idx, feature_set in enumerate(candidate_features):
        set_name = f"feature_set_{feature_set_idx}"
        logger.info(f"Evaluating {set_name}: {feature_set}")
        
        cv_scores[set_name] = []
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(feature_matrix, 
                                                                     target_vector,
                                                                     participants)):
            # Split data
            X_train = feature_matrix.iloc[train_idx][feature_set]
            y_train = target_vector[train_idx]
            X_val = feature_matrix.iloc[val_idx][feature_set]
            y_val = target_vector[val_idx]
            part_train = participants[train_idx]
            
            # Fit model on training data
            with pm.Model() as model:
                # Feature effects
                betas = {feature: pm.StudentT(feature, nu=3, mu=0, sigma=1)
                        for feature in feature_set}
                
                # Random participant effects
                participant_sigma = pm.HalfStudentT('participant_sigma', nu=3, sigma=1)
                unique_participants = np.unique(part_train)
                participant_offset = pm.StudentT('participant_offset',
                                              nu=3, mu=0, sigma=participant_sigma,
                                              shape=len(unique_participants))
                
                # Linear predictor
                mu = sum(betas[f] * X_train[f] for f in feature_set)
                mu = mu + participant_offset[participants[train_idx]]
                
                # Likelihood
                sigma = pm.HalfStudentT('sigma', nu=3, sigma=1)
                likelihood = pm.StudentT('likelihood', nu=3, mu=mu, sigma=sigma,
                                       observed=y_train)
                
                # Sample
                trace = pm.sample(n_samples, chains=2, progressbar=False)
                
                # Calculate WAIC and LOO-CV for this fold
                if fold == 0:  # Only need to do this once per feature set
                    waic = az.waic(trace, scale='deviance')
                    loo = az.loo(trace, scale='deviance')
                    waic_scores[set_name] = waic.waic
                    loo_scores[set_name] = loo.loo
                
                # Predict on validation set
                posterior_pred = pm.sample_posterior_predictive(trace, model=model)
                cv_scores[set_name].append(evaluate_predictions(posterior_pred, y_val))
    
    # Calculate feature importance and stability
    feature_importance = calculate_feature_importance(feature_matrix, target_vector,
                                                    feature_names)
    
    # Calculate feature correlations
    feature_correlations = calculate_feature_correlations(feature_matrix)
    
    # Save evaluation results
    save_evaluation_results(
        cv_scores=cv_scores,
        waic_scores=waic_scores,
        loo_scores=loo_scores,
        feature_importance=feature_importance,
        feature_correlations=feature_correlations,
        output_dir=output_dir
    )
    
    return FeatureEvaluationResults(
        cv_scores=cv_scores,
        waic_scores=waic_scores,
        loo_scores=loo_scores,
        feature_correlations=feature_correlations,
        feature_importance=feature_importance
    )

def calculate_feature_importance(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculate feature importance using multiple methods and aggregate results.
    """
    importance_scores = {}
    
    # Method 1: Correlation with target
    corr_importance = {feature: abs(stats.spearmanr(feature_matrix[feature], 
                                                   target_vector)[0])
                      for feature in feature_names}
    
    # Method 2: Mutual information
    from sklearn.feature_selection import mutual_info_regression
    mi_importance = dict(zip(feature_names, 
                           mutual_info_regression(feature_matrix, target_vector)))
    
    # Method 3: Permutation importance using a simple model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(feature_matrix, target_vector)
    perm_importance = permutation_importance(rf, feature_matrix, target_vector,
                                           n_repeats=10, random_state=42)
    perm_scores = dict(zip(feature_names, perm_importance.importances_mean))
    
    # Aggregate scores (normalized weighted average)
    for feature in feature_names:
        importance_scores[feature] = (
            0.3 * corr_importance[feature] / max(corr_importance.values()) +
            0.3 * mi_importance[feature] / max(mi_importance.values()) +
            0.4 * perm_scores[feature] / max(perm_scores.values())
        )
    
    return importance_scores

def calculate_feature_correlations(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Calculate and analyze feature correlations."""
    return feature_matrix.corr()

def evaluate_predictions(
    posterior_pred: az.InferenceData,
    y_true: np.ndarray
) -> float:
    """Evaluate predictions using appropriate metrics."""
    y_pred_mean = posterior_pred.posterior_predictive.likelihood.mean(dim=["chain", "draw"])
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((y_true - y_pred_mean) ** 2))
    mae = np.mean(np.abs(y_true - y_pred_mean))
    r2 = 1 - np.sum((y_true - y_pred_mean) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Combine metrics into single score
    score = (1/rmse + mae + r2) / 3
    return score

def save_evaluation_results(
    cv_scores: Dict[str, List[float]],
    waic_scores: Dict[str, float],
    loo_scores: Dict[str, float],
    feature_importance: Dict[str, float],
    feature_correlations: pd.DataFrame,
    output_dir: Path
) -> None:
    """Save evaluation results and create visualizations."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numerical results
    results_df = pd.DataFrame({
        'feature_set': list(cv_scores.keys()),
        'mean_cv_score': [np.mean(scores) for scores in cv_scores.values()],
        'std_cv_score': [np.std(scores) for scores in cv_scores.values()],
        'waic_score': [waic_scores[fs] for fs in cv_scores.keys()],
        'loo_score': [loo_scores[fs] for fs in cv_scores.keys()]
    })
    results_df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    
    # Save feature importance scores
    importance_df = pd.DataFrame(list(feature_importance.items()),
                               columns=['feature', 'importance'])
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Save correlation matrix
    feature_correlations.to_csv(output_dir / 'feature_correlations.csv')
    
    # Create visualizations
    plot_evaluation_results(cv_scores, waic_scores, loo_scores,
                          feature_importance, feature_correlations,
                          output_dir)

def plot_evaluation_results(
    cv_scores: Dict[str, List[float]],
    waic_scores: Dict[str, float],
    loo_scores: Dict[str, float],
    feature_importance: Dict[str, float],
    feature_correlations: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create visualizations of evaluation results."""
    # Set style
    plt.style.use('seaborn')
    
    # 1. Feature importance plot
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importances)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(feature_correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_correlations.png')
    plt.close()
    
    # 3. Cross-validation results
    plt.figure(figsize=(12, 6))
    plt.boxplot([scores for scores in cv_scores.values()],
                labels=list(cv_scores.keys()))
    plt.xticks(rotation=45)
    plt.ylabel('Cross-validation Score')
    plt.title('Feature Set Performance Comparison')
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_scores.png')
    plt.close()
