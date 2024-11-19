"""
Bayesian Modeling Module

This module implements the Bayesian GLMM analysis for keyboard layout optimization.
It provides functions for model training, validation, and comfort score generation.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
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

def train_bayesian_glmm(feature_matrix: pd.DataFrame,
                        target_vector: np.ndarray,
                        participants: Optional[np.ndarray] = None,
                        design_features: List[str] = ['row_sum', 'engram_sum'],
                        control_features: List[str] = ['freq'],
                        inference_method: str = "mcmc",
                        num_samples: int = 1000,
                        chains: int = 4) -> Tuple[az.InferenceData, pm.Model, List[Any]]:
    """
    Train a Bayesian GLMM for keyboard layout analysis.
    
    Args:
        feature_matrix: DataFrame containing feature values
        target_vector: Target variable to predict
        participants: Optional participant IDs for random effects
        design_features: Features used for layout optimization
        control_features: Features to control for but not use in optimization
        inference_method: Either "mcmc" or "variational"
        num_samples: Number of posterior samples
        chains: Number of chains for MCMC
        
    Returns:
        Tuple containing:
        - trace: ArviZ InferenceData object
        - model: PyMC Model object
        - all_priors: List of prior distributions
        
    Raises:
        ValueError: If input validation fails
        RuntimeError: If model training fails
    """
    logger.info("Starting Bayesian GLMM training")
    
    # Additional data validation
    if np.any(np.isinf(target_vector)) or np.any(np.isnan(target_vector)):
        raise ValueError("Target vector contains inf or nan values")
    if np.any(np.isinf(feature_matrix)) or np.any(np.isnan(feature_matrix)):
        raise ValueError("Feature matrix contains inf or nan values")
    
    # Robust standardization
    def robust_standardize(x):
        median = np.median(x)
        mad = np.median(np.abs(x - median)) * 1.4826  # Scale factor for normal distribution
        return (x - median) / (mad + 1e-8)  # Add small constant to prevent division by zero
    
    # Standardize target using robust method
    standardized_target = robust_standardize(target_vector)
    
    # Standardize features using robust method
    features_standardized = feature_matrix.copy()
    for col in feature_matrix.columns:
        features_standardized[col] = robust_standardize(feature_matrix[col])
    
    if participants is None:
        participants = np.arange(len(target_vector))
    
    unique_participants = np.unique(participants)
    num_participants = len(unique_participants)
    participant_map = {p: i for i, p in enumerate(unique_participants)}
    participants_contiguous = np.array([participant_map[p] for p in participants])
    
    all_features = design_features + control_features
    
    with pm.Model() as model:
        try:
            # Feature effects with weakly informative priors
            beta = {}
            for feature in all_features:
                beta[feature] = pm.StudentT(
                    feature,
                    nu=3,
                    mu=0,
                    sigma=1,
                    initval=0.0
                )
            
            # Random participant effects
            participant_sigma = pm.HalfStudentT('participant_sigma',
                                              nu=3,
                                              sigma=1,
                                              initval=0.5)
            
            participant_offset = pm.StudentT('participant_offset',
                                           nu=3,
                                           mu=0,
                                           sigma=participant_sigma,
                                           shape=num_participants,
                                           initval=np.zeros(num_participants))
            
            # Construct linear predictor
            mu = 0
            for feature in all_features:
                mu += beta[feature] * features_standardized[feature]
            
            mu = mu + participant_offset[participants_contiguous]
            
            # Observation noise
            sigma = pm.HalfStudentT('sigma',
                                  nu=3,
                                  sigma=1,
                                  initval=1.0)
            
            # Likelihood
            likelihood = pm.StudentT('likelihood',
                                   nu=3,
                                   mu=mu,
                                   sigma=sigma,
                                   observed=standardized_target)
            
            # Sampling with automatic initialization
            if inference_method == "mcmc":
                try:
                    logger.info("Starting NUTS sampling")
                    trace = pm.sample(
                        draws=num_samples,
                        tune=1000,
                        chains=chains,
                        cores=min(chains, 2),
                        return_inferencedata=True,
                        target_accept=0.9  # Conservative acceptance rate
                    )
                except Exception as e:
                    logger.warning(f"NUTS sampling failed: {str(e)}")
                    logger.info("Falling back to Metropolis sampling")
                    trace = pm.sample(
                        draws=num_samples,
                        tune=2000,
                        step=pm.Metropolis(),
                        chains=1,
                        return_inferencedata=True
                    )
            else:
                logger.info("Using variational inference")
                try:
                    approx = pm.fit(
                        n=20000,
                        method='advi'
                    )
                    trace = approx.sample(num_samples)
                except Exception as e:
                    logger.warning(f"Variational inference failed: {str(e)}")
                    logger.info("Falling back to simple MCMC")
                    trace = pm.sample(
                        draws=num_samples,
                        tune=500,
                        chains=1,
                        return_inferencedata=True
                    )

            logger.info("Model training completed successfully")
            
            all_priors = list(beta.values()) + [participant_sigma, sigma]
            return trace, model, all_priors
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}")

def _sample_from_model(inference_method: str, 
                      num_samples: int, 
                      chains: int) -> az.InferenceData:
    """Sample from the PyMC model using specified inference method."""
    if inference_method == "mcmc":
        try:
            trace = pm.sample(num_samples,
                            chains=chains,
                            cores=min(chains, 2),
                            return_inferencedata=True)
        except Exception as e:
            logger.warning(f"Multi-chain sampling failed: {str(e)}")
            try:
                trace = pm.sample(num_samples,
                                chains=1,
                                return_inferencedata=True)
            except Exception as e:
                logger.warning(f"NUTS sampling failed: {str(e)}")
                trace = pm.sample(num_samples,
                                chains=1,
                                step=pm.Metropolis(),
                                return_inferencedata=True)
    elif inference_method == "variational":
        try:
            approx = pm.fit(method="advi", n=num_samples)
            trace = approx.sample(num_samples)
        except Exception as e:
            logger.warning(f"Variational inference failed: {str(e)}")
            trace = pm.sample(num_samples,
                            chains=1,
                            return_inferencedata=True)
    else:
        raise ValueError("inference_method must be 'mcmc' or 'variational'")

    return trace

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
