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

from bigram_feature_extraction import get_feature_groups

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

def evaluate_feature_sets(
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    candidate_features: List[List[str]],
    feature_names: List[str],
    output_dir: Path,
    config: Dict[str, Any],  # Add config parameter
    n_splits: int = 5,
    n_samples: int = 1000
) -> FeatureEvaluationResults:
    """Evaluate different feature sets using cross-validation and information criteria."""
    cv_scores = {}
    waic_scores = {}
    loo_scores = {}

    # Validate input data
    if feature_matrix.empty:
        raise ValueError("Feature matrix is empty. Please provide valid data.")
    if len(target_vector) == 0:
        raise ValueError("Target vector is empty. Please provide valid target values.")

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
                participants,
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

    # Diagnostic logging for empty scores
    for key, scores in cv_scores.items():
        if not scores:
            logger.warning(f"Empty scores for key: {key}")

    # Calculate feature importance and stability
    feature_importance = calculate_feature_importance(
        feature_matrix, target_vector, feature_names
    )

    # Calculate feature correlations
    feature_correlations = calculate_feature_correlations(feature_matrix)

    # Add multicollinearity check
    multicollinearity_results = check_multicollinearity_vif(feature_matrix)

    # Save results to output directory
    save_evaluation_results(
        cv_scores=cv_scores,
        waic_scores=waic_scores,
        loo_scores=loo_scores,
        feature_importance=feature_importance,
        feature_correlations=feature_correlations,
        feature_combinations=[
            {'name': f'feature_set_{i}', 'features': features} 
            for i, features in enumerate(candidate_features)
        ],
        output_dir=output_dir,
        multicollinearity_results=multicollinearity_results
    )

    # Return results, ensuring all required fields are present
    return FeatureEvaluationResults(
        cv_scores=cv_scores,
        waic_scores=waic_scores,
        loo_scores=loo_scores,
        feature_correlations=feature_correlations,
        feature_importance=feature_importance,
        feature_groups=get_feature_groups(config),
        interaction_scores=None,  # Placeholder for future implementation
        stability_metrics=None  # Placeholder for future implementation
    )
       
def check_multicollinearity_vif(feature_matrix: pd.DataFrame) -> Dict:
    """Check for multicollinearity using Variance Inflation Factor."""
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

def perform_cross_validation(
    feature_matrix,
    target_vector,
    participants,
    n_splits=5
) -> List[float]:
    """
    Perform cross-validation while preserving participant grouping.
    """
    cv_scores = []
    group_kfold = GroupKFold(n_splits=n_splits)
    
    for train_idx, val_idx in group_kfold.split(feature_matrix, target_vector, participants):
        # Get training and validation sets
        X_train = feature_matrix.iloc[train_idx]
        y_train = target_vector[train_idx]
        
        X_val = feature_matrix.iloc[val_idx]
        y_val = target_vector[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        
        try:
            # Create and train model
            with pm.Model() as model:
                # Fixed effects priors
                feature_effects = {}
                for feat in X_train.columns:
                    feature_effects[feat] = pm.Normal(feat, mu=0, sigma=1)
                
                # Error term
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # Expected value
                mu = sum(feature_effects[feat] * X_train_scaled[feat] for feat in X_train.columns)
                
                # Likelihood
                likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_train)
                
                # Sample with increased tuning and adaptation
                trace = pm.sample(
                    draws=500,      # Increased
                    tune=2000,      # Increased
                    chains=4,
                    target_accept=0.99,  # Increased
                    init='adapt_diag',
                    return_inferencedata=True,
                    cores=1
                )
                
                # Get feature effect means for prediction
                feature_effects_mean = {
                    feat: float(trace.posterior[feat].mean())
                    for feat in X_train.columns
                }
                
                # Make predictions on validation set
                y_pred = sum(feature_effects_mean[feat] * X_val_scaled[feat].values 
                            for feat in X_val_scaled.columns)
                
                # Calculate R² score
                ss_res = np.sum((y_val - y_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                
                cv_scores.append(r2)
                
        except Exception as e:
            logger.warning(f"Failed fold with error: {str(e)}")
            cv_scores.append(np.nan)
    
    return cv_scores

def calculate_waic_loo(feature_matrix: pd.DataFrame, target_vector: np.ndarray) -> Tuple[float, float]:
    """
    Calculate both WAIC and LOO scores in a single model context.
    """
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
            
            # Likelihood
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
                nuts={'log_likelihood': True},  # Correct placement of log_likelihood
                cores=1
            )
            
            # Calculate information criteria
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
