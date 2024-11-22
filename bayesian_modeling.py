"""
Bayesian Modeling Module

This module implements the Bayesian GLMM analysis for keyboard layout optimization.
It provides functions for model training, validation, and comfort score generation.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Any, Dict
from sklearn.preprocessing import StandardScaler
import logging
import json

logger = logging.getLogger(__name__)

def validate_inputs(feature_matrix: pd.DataFrame, 
                   target_vector: np.ndarray, 
                   participants: Optional[np.ndarray]) -> bool:
    """Validate inputs for GLMM fitting."""
    if len(feature_matrix) != len(target_vector):
        raise ValueError("Feature matrix and target vector must have same length")
    if participants is not None and len(participants) != len(target_vector):
        raise ValueError("Participants array must match target vector length")
    if np.any(pd.isnull(feature_matrix)):
        raise ValueError("Feature matrix contains null values")
    return True

def train_bayesian_glmm(
   feature_matrix: pd.DataFrame,
   target_vector: np.ndarray,
   participants: np.ndarray,
   design_features: List[str],
   control_features: List[str],
   inference_method: str = 'nuts',
   num_samples: int = 1000,
   chains: int = 4
) -> Tuple:
   """
   Train Bayesian GLMM using NUTS sampling.

   Args:
       feature_matrix: DataFrame containing feature values
       target_vector: Array of target values
       participants: Array of participant IDs
       design_features: List of design feature names
       control_features: List of control feature names 
       inference_method: Sampling method ('nuts' recommended)
       num_samples: Number of samples to draw
       chains: Number of MCMC chains

   Returns:
       Tuple containing:
       - trace: Sampling trace
       - model: PyMC model
       - priors: Dictionary of model priors
   """
   logger.info("Starting Bayesian GLMM training")

   # Scale features
   scaler = StandardScaler()
   X_scaled = pd.DataFrame(
       scaler.fit_transform(feature_matrix),
       columns=feature_matrix.columns
   )

   # Create participant mapping for indexing
   unique_participants = np.unique(participants)
   participant_map = {p: i for i, p in enumerate(unique_participants)}
   n_participants = len(unique_participants)

   with pm.Model() as model:
       # Priors for design features
       design_effects = {
           feat: pm.Normal(feat, mu=0, sigma=1)
           for feat in design_features
       }

       # Priors for control features
       control_effects = {
           feat: pm.Normal(feat, mu=0, sigma=1)
           for feat in control_features
       }

       # Random effects for participants
       participant_sigma = pm.HalfNormal('participant_sigma', sigma=1)
       participant_offset = pm.Normal('participant_offset',
                                    mu=0,
                                    sigma=participant_sigma,
                                    shape=n_participants)

       # Error term
       sigma = pm.HalfNormal('sigma', sigma=1)

       # Expected value combining fixed and random effects
       mu = 0
       # Add design feature effects
       for feat in design_features:
           mu += design_effects[feat] * X_scaled[feat]
       # Add control feature effects
       for feat in control_features:
           mu += control_effects[feat] * X_scaled[feat]
       # Add participant random effects
       mu += participant_offset[[participant_map[p] for p in participants]]

       # Likelihood
       likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=target_vector)

       # Sample using NUTS
       logger.info(f"Using {inference_method} inference")
       trace = pm.sample(
            draws=num_samples,
            tune=2000,
            chains=chains,
            target_accept=0.99,
            return_inferencedata=True,
            compute_convergence_checks=True,
            cores=1
       )

       # Collect priors
       priors = {
           'design_effects': design_effects,
           'control_effects': control_effects,
           'participant_sigma': participant_sigma,
           'sigma': sigma
       }

       logger.info("Model training completed successfully")
       return trace, model, priors
   
def calculate_all_bigram_comfort_scores(
        trace: az.InferenceData,
        all_bigram_features: pd.DataFrame,
        features_for_design: Optional[List[str]] = None,
        mirror_scores: bool = True) -> Dict[Tuple[str, str], float]:
    """Calculate comfort scores for all possible bigrams."""
    if features_for_design is None:
        all_vars = list(trace.posterior.data_vars)
        params = [var for var in all_vars if var not in ['participant_intercept', 'sigma']]
        features_for_design = [p for p in params if p != 'freq']

    logger.info(f"Design features used for scoring: {features_for_design}")

    posterior_samples = {param: az.extract(trace, var_names=param).values
                        for param in features_for_design}

    all_bigram_scores = {}
    for bigram, features in all_bigram_features.iterrows():
        scores = np.zeros(len(next(iter(posterior_samples.values()))))
        for param in features_for_design:
            if param in features.index:
                scores += posterior_samples[param] * features[param]
        
        all_bigram_scores[bigram] = np.mean(scores)

    # Normalize scores to 0-1 range
    min_score = min(all_bigram_scores.values())
    max_score = max(all_bigram_scores.values())
    normalized_scores = {bigram: 1 - (score - min_score) / (max_score - min_score)
                        for bigram, score in all_bigram_scores.items()}

    if mirror_scores:
        left_keys = "qwertasdfgzxcvb"
        right_keys = "poiuy;lkjh/.,mn"
        key_mapping = dict(zip(left_keys, right_keys))
        
        right_scores = {}
        for bigram, score in normalized_scores.items():
            if isinstance(bigram, tuple) and len(bigram) == 2:
                right_bigram = (key_mapping.get(bigram[0], bigram[0]),
                              key_mapping.get(bigram[1], bigram[1]))
                right_scores[right_bigram] = score
        
        all_scores = {**normalized_scores, **right_scores}
    else:
        all_scores = normalized_scores

    return all_scores

def save_model_results(trace: az.InferenceData, 
                      model: pm.Model, 
                      base_filename: str) -> None:
    """Save model results to files."""
    # Save trace
    az.to_netcdf(trace, filename=f"{base_filename}_trace.nc")
    
    # Save point estimates
    point_estimates = az.summary(trace)
    point_estimates.to_csv(f"{base_filename}_point_estimates.csv")
    
    # Save model configuration
    model_config = {
        'input_vars': [var.name for var in model.named_vars.values() 
                      if hasattr(var, 'distribution')],
        'observed_vars': [var.name for var in model.observed_RVs],
        'free_vars': [var.name for var in model.free_RVs],
    }
    
    # Save prior information
    prior_info = {}
    for var in model.named_vars.values():
        if hasattr(var, 'distribution'):
            prior_info[var.name] = {
                'distribution': var.distribution.__class__.__name__,
                'parameters': {k: str(v) for k, v in var.distribution.parameters.items()
                             if k != 'name'}
            }
    
    model_info = {
        'config': model_config,
        'priors': prior_info
    }
    
    with open(f"{base_filename}_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

def load_model_results(base_filename: str) -> Tuple[az.InferenceData, pd.DataFrame, Dict]:
    """Load saved model results."""
    trace = az.from_netcdf(f"{base_filename}_trace.nc")
    point_estimates = pd.read_csv(f"{base_filename}_point_estimates.csv", index_col=0)
    
    with open(f"{base_filename}_model_info.json", "r") as f:
        model_info = json.load(f)
    
    return trace, point_estimates, model_info

def perform_sensitivity_analysis(
        feature_matrix: pd.DataFrame,
        target_vector: np.ndarray,
        participants: np.ndarray,
        design_features: List[str],
        control_features: List[str],
        prior_scale_factors: List[float]) -> Dict[str, Any]:
    """Perform sensitivity analysis on model priors."""
    logger.info("Starting sensitivity analysis")
    
    parameter_estimates = {}
    convergence_metrics = {}
    sampling_issues = {}
    
    baseline_sds = {feature: np.std(feature_matrix[feature])
                   for feature in design_features + control_features}
    
    for scale in prior_scale_factors:
        config_name = f"scale_{scale:.1f}"
        logger.info(f"Testing prior scale factor: {scale}")
        
        try:
            trace, _, _ = train_bayesian_glmm(
                feature_matrix=feature_matrix,
                target_vector=target_vector,
                participants=participants,
                design_features=design_features,
                control_features=control_features,
                num_samples=1000,
                chains=2
            )
            
            summary = az.summary(trace)
            parameter_estimates[config_name] = summary
            
            convergence_metrics[config_name] = {
                'r_hat': summary['r_hat'].values,
                'ess_bulk': summary['ess_bulk'].values,
                'ess_tail': summary['ess_tail'].values
            }
            
        except Exception as e:
            logger.error(f"Error with scale factor {scale}: {str(e)}")
            sampling_issues[config_name] = str(e)
            continue
    
    results = {
        'parameter_estimates': parameter_estimates,
        'convergence_metrics': convergence_metrics,
        'sampling_issues': sampling_issues
    }
    
    return results

def calculate_stability_metrics(parameter_estimates: Dict[str, pd.DataFrame],
                                features: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate metrics showing how stable each feature's effect is."""
    stability_metrics = {}
    
    for feature in features:
        means = [est.loc[feature, 'mean'] for est in parameter_estimates.values()]
        sds = [est.loc[feature, 'sd'] for est in parameter_estimates.values()]
        
        stability_metrics[feature] = {
            'direction_consistency': np.mean(np.sign(means) == np.sign(np.mean(means))),
            'relative_variation': np.std(means) / np.abs(np.mean(means)),
            'min_mean': np.min(means),
            'max_mean': np.max(means),
            'mean_of_means': np.mean(means),
            'sd_of_means': np.std(means),
            'mean_of_sds': np.mean(sds)
        }
    
    return stability_metrics

def bayesian_pairwise_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate score for Bayesian pairwise comparison predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred > 0).astype(int)
    
    accuracy = np.mean(y_true_binary == y_pred_binary)
    
    epsilon = 1e-15
    probs = 1 / (1 + np.exp(-np.abs(y_pred)))
    log_likelihood = np.mean(y_true_binary * np.log(probs + epsilon) +
                           (1 - y_true_binary) * np.log(1 - probs + epsilon))
    
    score = (accuracy + (log_likelihood + 1) / 2) / 2
    
    return score

#==============================================#
# Functions to visualize model characteristics #
#==============================================#
def plot_model_diagnostics(trace, output_base_path: str, inference_method: str) -> None:
    """Plot diagnostics with proper dimension handling."""
    try:
        # Get available variables excluding composite ones
        available_vars = [var for var in trace.posterior.variables 
                         if not any(dim in var for dim in ['chain', 'draw', 'dim'])]
        
        if available_vars:
            az.plot_trace(trace, var_names=available_vars)
            plt.savefig(output_base_path.format(inference_method=inference_method))
            plt.close()
        
        # Plot forest plot for parameters only
        param_vars = [var for var in available_vars 
                     if var not in ['participant_offset', 'participant_sigma', 'sigma']]
        if param_vars:
            az.plot_forest(trace, var_names=param_vars)
            plt.savefig(output_base_path.format(inference_method=inference_method)
                       .replace('diagnostics', 'forest'))
            plt.close()
            
    except Exception as e:
        logger.warning(f"Could not create some diagnostic plots: {str(e)}")

def plot_sensitivity_analysis(parameter_estimates: Dict[str, pd.DataFrame],
                              design_features: List[str],
                              control_features: List[str],
                              output_path: str) -> None:
    """
    Create visualization of sensitivity analysis results.
    
    Args:
        parameter_estimates: Dictionary of parameter estimates
        design_features: List of design features
        control_features: List of control features
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Design features plot
    plt.subplot(2, 1, 1)
    data = []
    labels = []
    for feature in design_features:
        means = [est.loc[feature, 'mean'] for est in parameter_estimates.values()]
        data.append(means)
        labels.append(feature)
    
    plt.boxplot(data, labels=labels)
    plt.title('Design Feature Effect Estimates')
    plt.ylabel('Effect Size')
    plt.grid(True, alpha=0.3)
    
    # Control features plot
    if control_features:
        plt.subplot(2, 1, 2)
        data = []
        labels = []
        for feature in control_features:
            means = [est.loc[feature, 'mean'] for est in parameter_estimates.values()]
            data.append(means)
            labels.append(feature)
        
        plt.boxplot(data, labels=labels)
        plt.title('Control Feature Effect Estimates')
        plt.ylabel('Effect Size')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
