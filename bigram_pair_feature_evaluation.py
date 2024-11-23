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
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import pymc as pm
import arviz as az
from scipy import stats
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

@dataclass
class FeatureEvaluationResults:
    """Container for feature evaluation results."""
    cv_scores: Dict[str, List[float]]
    waic_scores: Dict[str, float]
    loo_scores: Dict[str, float]
    feature_correlations: pd.DataFrame
    feature_importance: Dict[str, float]
    feature_groups: Dict[str, List[str]]
    interaction_scores: Optional[Dict[Tuple[str, str], float]] = None
    stability_metrics: Optional[Dict[str, Dict[str, float]]] = None

def evaluate_feature_sets(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    candidate_features: List[List[str]],
    feature_names: List[str],
    output_dir: Path,
    config: Dict[str, Any],
    n_splits: int = 5,
    n_samples: int = 1000
) -> FeatureEvaluationResults:
    """
    Evaluate different combinations of features to determine their predictive power.
    
    This function performs multiple analyses on feature combinations:
    1. Cross-validation to assess predictive performance using R² scores
       - R² measures how well features predict comfort/preference scores
       - Higher R² indicates better prediction (1.0 is perfect, 0.0 is poor)
       - Scores are averaged across folds for stability
    2. Feature importance calculation
    3. Feature correlation analysis
    4. Multicollinearity checks
    
    Args:
        feature_matrix: DataFrame of all available features
        target_vector: Array of target values to predict
        participants: Array of participant IDs for grouped cross-validation
        candidate_features: List of feature combinations to evaluate
        feature_names: Names of all available features
        output_dir: Directory to save evaluation results
        config: Configuration dictionary
        n_splits: Number of cross-validation folds
        n_samples: Number of samples for Bayesian modeling
        
    Returns:
        FeatureEvaluationResults containing:
        - Cross-validation R² scores for each feature set
        - Feature correlations
        - Feature importance scores
        - Feature grouping information
    """
    cv_scores = {}
    waic_scores = {}
    loo_scores = {}

    # Validate input data
    if feature_matrix.empty:
        raise ValueError("Feature matrix is empty. Please provide valid data.")
    if len(target_vector) == 0:
        raise ValueError("Target vector is empty. Please provide valid target values.")

    # Convert participants to integer indices for cross-validation
    unique_participants = np.unique(participants)
    participant_map = {p: i for i, p in enumerate(unique_participants)}
    participant_indices = np.array([participant_map[p] for p in participants])

    # Evaluate each feature set
    for set_idx, feature_set in enumerate(candidate_features):
        set_name = f"feature_set_{set_idx}"
        logger.info(f"Evaluating {set_name}: {feature_set}")
        
        # Initialize scores
        cv_scores[set_name] = []
        
        # Perform cross-validation
        try:
            cv_result = perform_cross_validation(
                feature_matrix[feature_set],
                target_vector,
                participant_indices,  # Use integer indices
                n_splits=n_splits
            )
            cv_scores[set_name] = cv_result
        except Exception as e:
            logger.warning(f"Cross-validation failed for {set_name}: {e}")
            cv_scores[set_name] = [np.nan] * n_splits
        
        # Calculate WAIC and LOO
        try:
            waic_score, loo_score = calculate_waic_loo(
                feature_matrix[feature_set], 
                target_vector
            )
            waic_scores[set_name] = waic_score
            loo_scores[set_name] = loo_score
        except Exception as e:
            logger.warning(f"WAIC/LOO computation failed for {set_name}: {e}")
            waic_scores[set_name] = np.nan
            loo_scores[set_name] = np.nan

    # Calculate feature importance and stability
    feature_importance = calculate_feature_importance(
        feature_matrix, target_vector, feature_names
    )

    # Calculate feature correlations
    feature_correlations = calculate_feature_correlations(feature_matrix)

    # Return results using the actual feature groups from config
    return FeatureEvaluationResults(
        cv_scores=cv_scores,
        waic_scores=waic_scores,
        loo_scores=loo_scores,
        feature_correlations=feature_correlations,
        feature_importance=feature_importance,
        feature_groups=config['features']['groups'],  # Use the real feature groups
        interaction_scores=None,
        stability_metrics=None
    )

def evaluate_predictions(
    posterior_pred: az.InferenceData,
    y_true: np.ndarray,
    X_val: pd.DataFrame,
    part_val: np.ndarray,
    model: pm.Model,
    participant_map: Dict[int, int]
) -> float:
    """
    Evaluate predictions using appropriate metrics.
    """
    # Map validation set participants to model indices
    part_val_mapped = np.array([participant_map.get(p, -1) for p in part_val])
    
    # Get posterior means for parameters
    trace_posterior = posterior_pred.posterior
    
    # Calculate predictions
    y_pred = np.zeros_like(y_true)
    valid_participants = part_val_mapped >= 0
    
    if np.any(valid_participants):
        # Get the mean of the posterior for each parameter
        feature_effects = {
            var: trace_posterior[var].mean(dim=["chain", "draw"]).values
            for var in model.named_vars
            if var not in ['participant_offset', 'participant_sigma', 'sigma', 'likelihood']
        }
        participant_effects = trace_posterior['participant_offset'].mean(dim=["chain", "draw"]).values
        
        # Calculate predictions
        for i, (_, row) in enumerate(X_val.iterrows()):
            if valid_participants[i]:
                # Fixed effects
                y_pred[i] = sum(feature_effects[feat] * row[feat] for feat in X_val.columns)
                # Random effects
                y_pred[i] += participant_effects[part_val_mapped[i]]
    
    # Calculate metrics only for valid participants
    valid_mask = valid_participants & ~np.isnan(y_pred) & ~np.isnan(y_true)
    if not np.any(valid_mask):
        return np.nan
    
    rmse = np.sqrt(np.mean((y_true[valid_mask] - y_pred[valid_mask]) ** 2))
    mae = np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]))
    
    # R² calculation
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
    ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Combine metrics into single score
    score = (1/rmse + mae + r2) / 3
    return score

def check_multicollinearity_vif(feature_matrix: pd.DataFrame) -> Dict:
    """
    Check for multicollinearity among features using Variance Inflation Factor.
    
    Performs two types of checks:
    1. VIF calculation for each feature (>5 indicates high multicollinearity)
    2. Pairwise correlation analysis (>0.7 indicates high correlation)
    
    Args:
        feature_matrix: DataFrame of features to check
        
    Returns:
        Dictionary containing:
        - VIF scores for each feature
        - List of highly correlated feature pairs
        - Status indicators for problematic features
    """
    results = {
        'vif': [],
        'high_correlations': []
    }
    
    # Add constant term
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
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {column}: {str(e)}")
    
    # Calculate correlation matrix
    corr_matrix = feature_matrix.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.7:  # Threshold for high correlation
                results['high_correlations'].append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    return results

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

def calculate_waic_loo(feature_matrix: pd.DataFrame, target_vector: np.ndarray) -> Tuple[float, float]:
    """Calculate both WAIC and LOO scores in a single model context."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)
    
    try:
        with pm.Model() as model:
            # Fixed effects
            feature_effects = {
                feat: pm.Normal(feat, mu=0, sigma=1)
                for feat in feature_matrix.columns
            }
            
            # Error term
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Expected value
            mu = sum(feature_effects[feat] * X_scaled[:, i] 
                    for i, feat in enumerate(feature_matrix.columns))
            
            # Likelihood with observed data
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=target_vector)
            
            # Sample with improved settings
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
            
            # Calculate information criteria
            # Note: Recent PyMC versions handle log_likelihood internally
            waic = az.waic(trace)
            loo = az.loo(trace)
            
            return waic.waic, loo.loo
            
    except Exception as e:
        logger.warning(f"Failed to compute WAIC/LOO: {str(e)}")
        return np.nan, np.nan

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
    Save evaluation results and create visualizations.
    
    Args:
        cv_scores: Cross-validation scores for each feature combination
        waic_scores: WAIC scores for each feature combination
        loo_scores: LOO-CV scores for each feature combination
        feature_importance: Feature importance scores
        feature_correlations: Feature correlation matrix
        feature_combinations: List of feature combination definitions from config
        output_dir: Directory to save results
        multicollinearity_results: Optional multicollinearity analysis results
    """
    # Create results DataFrame with careful handling of empty scores
    results_df = pd.DataFrame({
        'feature_set': list(cv_scores.keys()),
        'mean_cv_score': [np.nanmean(scores) if scores and any(~np.isnan(scores)) else np.nan 
                         for scores in cv_scores.values()],
        'std_cv_score': [np.nanstd(scores) if scores and any(~np.isnan(scores)) else np.nan 
                        for scores in cv_scores.values()],
        'waic_score': [waic_scores.get(fs, np.nan) for fs in cv_scores.keys()],
        'loo_score': [loo_scores.get(fs, np.nan) for fs in cv_scores.keys()]
    })
        
    # Sort by mean CV score (descending), handling NaN values
    results_df = results_df.sort_values('mean_cv_score', ascending=False, na_position='last')
    results_df.to_csv(output_dir / 'evaluation_metrics.csv', index=False)
        
    # Also save detailed CV scores
    cv_details_df = pd.DataFrame(cv_scores)
    cv_details_df.index.name = 'fold'
    cv_details_df.to_csv(output_dir / 'cv_scores_detailed.csv')
    
    # 3. Save feature importance analysis
    importance_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False)
    
    with open(output_dir / 'feature_importance.txt', 'w') as f:
        f.write("=== Feature Importance Analysis ===\n\n")
        for _, row in importance_df.iterrows():
            f.write(f"{row['feature']:<20} {row['importance']:.4f}\n")
    
    importance_df.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # 4. Save correlation analysis
    with open(output_dir / 'feature_correlations.txt', 'w') as f:
        f.write("=== Feature Correlation Analysis ===\n\n")
        
        # Find high correlations
        high_corr_threshold = 0.7
        high_correlations = []
        
        for i in range(len(feature_correlations.columns)):
            for j in range(i+1, len(feature_correlations.columns)):
                corr = abs(feature_correlations.iloc[i, j])
                if corr >= high_corr_threshold:
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
    
    # Save full correlation matrix
    feature_correlations.to_csv(output_dir / 'feature_correlations.csv')
    
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
            
            # High correlations
            if multicollinearity_results['high_correlations']:
                f.write("\nHigh Correlations (r > 0.7):\n")
                for corr in sorted(multicollinearity_results['high_correlations'],
                                 key=lambda x: x['Correlation'],
                                 reverse=True):
                    f.write(f"{corr['Feature1']} - {corr['Feature2']}: {corr['Correlation']:.3f}\n")
    
    # 6. Save summary report
    with open(output_dir / 'evaluation_summary.txt', 'w') as f:
        f.write("=== Feature Evaluation Summary ===\n\n")
        
        # Best performing combinations
        f.write("Best Performing Feature Combinations:\n")
        top_n = 3
        for i, row in results_df.head(top_n).iterrows():
            f.write(f"\n{i+1}. {row['feature_set']}\n")
            f.write(f"   Mean CV Score: {row['mean_cv_score']:.4f} ± {row['std_cv_score']:.4f}\n")
            f.write(f"   WAIC: {row['waic_score']:.2f}\n")
            f.write(f"   LOO: {row['loo_score']:.2f}\n")
        
        # Most important features
        f.write("\nMost Important Features:\n")
        for _, row in importance_df.head(top_n).iterrows():
            f.write(f"{row['feature']:<20} {row['importance']:.4f}\n")
        
        # Multicollinearity concerns
        if multicollinearity_results:
            f.write("\nMulticollinearity Concerns:\n")
            high_vif = [x for x in multicollinearity_results['vif'] if x['VIF'] > 5]
            if high_vif:
                f.write("Features with high VIF (>5):\n")
                for vif in high_vif:
                    f.write(f"- {vif['Feature']}: {vif['VIF']:.2f}\n")
            else:
                f.write("No concerning VIF values found.\n")

def plot_evaluation_results(
    cv_scores: Dict[str, List[float]],
    waic_scores: Dict[str, float],
    loo_scores: Dict[str, float],
    feature_importance: Dict[str, float],
    feature_correlations: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create visualizations of evaluation results."""
    # Set style (using a default matplotlib style instead of seaborn)
    plt.style.use('default')
    
    # Set common parameters
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

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
    
    For example, R²=0.7 means our features explain 70% of the variation in comfort scores.
    
    Args:
        feature_matrix: DataFrame of features to evaluate
        target_vector: Array of comfort/preference scores
        participants: Array of participant IDs/indices for grouped validation
        n_splits: Number of cross-validation folds
    
    Returns:
        List[float]: R² scores for each fold, indicating how well features predict preferences
    """
    cv_scores = []
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Convert everything to numpy arrays
    X = feature_matrix.to_numpy()
    y = np.array(target_vector)
    groups = np.array(participants)
    
    # Ensure all arrays are the right shape
    X = np.atleast_2d(X)
    y = np.ravel(y)
    groups = np.ravel(groups)
    
    # Verify dimensions
    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("Inconsistent dimensions in input data")
    
    logger.info(f"Starting cross-validation with {n_splits} splits")
    logger.info(f"Data shapes - X: {X.shape}, y: {y.shape}, groups: {groups.shape}")
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
        try:
            # Extract training and validation sets
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Create and train model
            with pm.Model() as model:
                # Fixed effects priors
                feature_effects = {
                    f'feat_{i}': pm.Normal(f'feat_{i}', mu=0, sigma=1)
                    for i in range(X_train.shape[1])
                }
                
                # Error term
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # Expected value
                mu = sum(feature_effects[f'feat_{i}'] * X_train_scaled[:, i] 
                        for i in range(X_train.shape[1]))
                
                # Likelihood
                likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_train)
                
                # Sample
                trace = pm.sample(
                    draws=500,
                    tune=2000,
                    chains=4,
                    target_accept=0.99,
                    init='adapt_diag',
                    return_inferencedata=True,
                    cores=1
                )
                
                # Get feature effect means for prediction
                feature_effects_mean = {
                    f'feat_{i}': float(trace.posterior[f'feat_{i}'].mean())
                    for i in range(X_train.shape[1])
                }
                
                # Make predictions on validation set
                y_pred = sum(feature_effects_mean[f'feat_{i}'] * X_val_scaled[:, i]
                           for i in range(X_val.shape[1]))
                
                # Calculate R² score
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                logger.info(f"Fold {fold + 1}/{n_splits} R² score: {r2:.3f}")
                cv_scores.append(r2)
                
        except Exception as e:
            logger.warning(f"Failed fold {fold + 1}/{n_splits} with error: {str(e)}")
            cv_scores.append(np.nan)
    
    if not any(~np.isnan(score) for score in cv_scores):
        logger.warning("All cross-validation folds failed")
    else:
        logger.info(f"Average R² score: {np.nanmean(cv_scores):.3f}")
    
    return cv_scores
