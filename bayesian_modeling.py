"""
Bayesian Modeling Module

This module implements the Bayesian GLMM analysis for keyboard layout optimization.
Provides:
1. Model Training - GLMM with participant random effects
2. Model Evaluation - Performance metrics and diagnostics
3. Score Generation - Comfort score calculation and validation
4. Model Analysis - Sensitivity analysis and visualization
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Any, Dict
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
import logging
import sys
from datetime import datetime
import colorama
from colorama import Fore, Style

from data_processing import ProcessedData

logger = logging.getLogger(__name__)

#=========================#
# Core Model Functions    #
#=========================#
def train_bayesian_glmm(
   feature_matrix: pd.DataFrame,
   target_vector: np.ndarray,
   participants: np.ndarray,
   design_features: List[str],
   control_features: List[str],
   inference_method: str = 'mcmc',
   num_samples: int = 1000,
   chains: int = 4
) -> Tuple:
    """
    Train Bayesian Generalized Linear Mixed Model (GLMM) for keyboard layout analysis.
    
    This function fits a hierarchical model that accounts for:
    - Fixed effects from design features (layout characteristics)
    - Fixed effects from control features (e.g., typing time)
    - Random effects per participant
    
    Args:
        feature_matrix: DataFrame of feature values for bigram pairs
        target_vector: Array of comfort scores or preference ratings
        participants: Array of participant IDs for random effects
        design_features: Names of features related to keyboard layout design
        control_features: Names of features to control for (e.g., typing time)
        inference_method: MCMC method to use ('nuts' recommended)
        num_samples: Number of posterior samples to draw
        chains: Number of independent MCMC chains
        
    Returns:
        Tuple containing:
        - trace: ArviZ InferenceData object with posterior samples
        - model: Fitted PyMC model
        - priors: Dictionary of model prior distributions
    """
    logger.info("MODEL: Starting Bayesian GLMM training")
    logger.info("MODEL: Validating features...")
    
    # For the MCMC sampling progress
    class SamplingProgress:
        def __init__(self, total_chains):
            self.total_chains = total_chains
            self.current_chain = 0
        
        def update(self, chain_num, progress):
            logger.info(f"MODEL: Chain {chain_num+1}/{self.total_chains} "
                       f"[{'='*int(progress*20)}{' '*(20-int(progress*20))}] "
                       f"{progress*100:.1f}%")

    # Validate features
    available_features = set(feature_matrix.columns)
    missing_features = []

    for feature in design_features + control_features:
        if feature not in available_features:
            missing_features.append(feature)

    if missing_features:
        raise ValueError(
            f"The following features are not available in the feature matrix: "
            f"{missing_features}\nAvailable features are: {sorted(available_features)}"
        )

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
        #likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=target_vector)

        # Different sampling based on inference method
        logger.info(f"Using {inference_method} inference")
        
        if inference_method.lower() == 'variational':
            # Use ADVI for variational inference
            logger.info(f"Note: Using variational inference can lead to shape validation failure")
            approx = pm.fit(
                n=num_samples,
                method='advi',
                obj_optimizer=pm.adam(learning_rate=0.1)
            )
            trace = approx.sample(num_samples)
            
        else:  # default to MCMC/NUTS
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
   
def calculate_bigram_comfort_scores(
    trace: az.InferenceData,
    feature_matrix: pd.DataFrame,
    features_for_design: List[str],
    mirror_left_right_scores: bool = True
) -> Dict[Tuple[str, str], float]:
    """Calculate comfort scores for bigram pairs using trained model parameters."""
    logger.info(f"Calculating comfort scores for {len(feature_matrix)} bigram pairs")
    
    # Extract design feature effects from posterior
    posterior_samples = {}
    for param in features_for_design:
        if param in trace.posterior.variables:
            samples = az.extract(trace, var_names=param).values
            if samples.size > 0:
                posterior_samples[param] = samples
    
    if not posterior_samples:
        logger.error("No valid posterior samples found")
        return {}
    
    # Normalize index format
    normalized_index = []
    for idx in feature_matrix.index:
        norm_bigram = normalize_bigram_format(idx)
        if norm_bigram:
            normalized_index.append(norm_bigram)
        else:
            logger.warning(f"Could not normalize bigram format: {idx}")
    
    # Create new DataFrame with normalized index
    feature_matrix = feature_matrix.copy()
    feature_matrix.index = normalized_index
    
    logger.info(f"Processing {len(normalized_index)} normalized bigram pairs")
    
    scores = {}
    sample_size = next(iter(posterior_samples.values())).shape[0]
    
    for idx, features in feature_matrix.iterrows():
        effect = np.zeros(sample_size, dtype=np.float64)
        
        for param, samples in posterior_samples.items():
            if param in features:
                feature_value = features[param]
                if not np.isnan(feature_value):
                    effect += samples * feature_value
        
        if effect.size > 0 and not np.all(np.isnan(effect)):
            scores[idx] = float(np.nanmean(effect))
    
    logger.info(f"Generated scores for {len(scores)} bigram pairs")
    
    if scores:
        scores = normalize_scores(scores)
        if mirror_left_right_scores:
            scores = add_mirrored_scores(scores)
        logger.info(f"Final number of scores (including mirrored): {len(scores)}")
    
    return scores

def validate_comfort_scores(
    comfort_scores: Dict[Tuple[str, str], float],
    test_data: 'ProcessedData'
) -> Dict[str, float]:
    """Validate comfort scores against held-out test data."""
    validation_metrics = {}
    
    if not hasattr(test_data, 'bigram_pairs'):
        logger.error("Test data missing bigram pairs attribute")
        return validation_metrics
        
    logger.info(f"Test data bigram pairs: {len(test_data.bigram_pairs)}")
    
    # Convert scores to preferences
    predicted_prefs = []
    actual_prefs = []
    skipped_pairs = set()
    
    for (bigram1, bigram2), pref in test_data.bigram_pairs:
        # Normalize both bigrams
        norm_bigram1 = normalize_bigram_format(bigram1)
        norm_bigram2 = normalize_bigram_format(bigram2)
        
        if norm_bigram1 is None or norm_bigram2 is None:
            logger.debug(f"Could not normalize bigrams: {bigram1}, {bigram2}")
            continue
        
        if norm_bigram1 in comfort_scores and norm_bigram2 in comfort_scores:
            pred = comfort_scores[norm_bigram1] - comfort_scores[norm_bigram2]
            print("\n\nPRED = ", pred)
            predicted_prefs.append(pred)
            actual_prefs.append(pref)
        else:
            print("\n\nNOT IN = ", skipped_pairs)
            if (norm_bigram1, norm_bigram2) not in skipped_pairs:
                missing = []
                if norm_bigram1 not in comfort_scores:
                    missing.append(str(bigram1))
                if norm_bigram2 not in comfort_scores:
                    missing.append(str(bigram2))
                logger.info(f"Skipping pair: {missing} - Missing in comfort scores")
                skipped_pairs.add((norm_bigram1, norm_bigram2))
    
    predicted_prefs = np.array(predicted_prefs)
    actual_prefs = np.array(actual_prefs)
    
    logger.info(f"Predicted prefs: {len(predicted_prefs)}, Actual prefs: {len(actual_prefs)}")
    
    if len(predicted_prefs) > 0:
        validation_metrics['accuracy'] = float(np.mean(
            np.sign(predicted_prefs) == np.sign(actual_prefs)
        ))
        
        if len(predicted_prefs) > 1:
            validation_metrics['correlation'] = float(stats.spearmanr(
                predicted_prefs, actual_prefs
            )[0])
    else:
        logger.warning("Empty arrays detected: Skipping accuracy calculation.")
    
    # Add coverage metric
    total_pairs = len(test_data.bigram_pairs)
    pairs_evaluated = len(predicted_prefs)
    validation_metrics['coverage'] = pairs_evaluated / total_pairs if total_pairs > 0 else 0.0
    
    logger.info(f"Coverage: {validation_metrics['coverage']:.2%} ({pairs_evaluated}/{total_pairs} pairs)")
    
    return validation_metrics

#=========================#
# Model Evaluation       #
#=========================#
def evaluate_model_performance(
    trace: az.InferenceData,
    feature_matrix: pd.DataFrame,
    target_vector: np.ndarray,
    participants: np.ndarray,
    design_features: List[str],
    control_features: List[str]
) -> Dict[str, float]:
    """
    Evaluate model performance on data using posterior predictions.
    
    Args:
        trace: ArviZ InferenceData object containing posterior samples
        feature_matrix: Features to evaluate
        target_vector: True target values
        participants: Participant IDs
        design_features: Names of design features
        control_features: Names of control features
    
    Returns:
        Dictionary containing performance metrics:
        - r2: R² score
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - correlation: Spearman correlation
    """
    # Get posterior means for parameters
    posterior = trace.posterior
    
    # Initialize predictions array as float64 to avoid type casting issues
    predictions = np.zeros_like(target_vector, dtype=np.float64)
    
    # Ensure target vector is float64 for consistent calculations
    target_vector = target_vector.astype(np.float64)
    
    # Add fixed effects
    for feature in design_features + control_features:
        if feature in feature_matrix.columns:
            # Handle case where feature might be missing in posterior
            if feature in posterior:
                effect = float(posterior[feature].mean())
                # Ensure feature values are float64
                feature_values = feature_matrix[feature].values.astype(np.float64)
                predictions += effect * feature_values
    
    # Add random participant effects if present
    if 'participant_offset' in posterior:
        # Get unique participants and their effects
        unique_participants = np.unique(participants)
        participant_effects = posterior['participant_offset'].mean(dim=['chain', 'draw']).values
        
        # Ensure participant effects are float64
        if not isinstance(participant_effects, np.ndarray):
            participant_effects = np.array([participant_effects], dtype=np.float64)
        else:
            participant_effects = participant_effects.astype(np.float64)
        
        # Create a mapping from participant ID to effect index
        participant_map = {p: i for i, p in enumerate(unique_participants)}
        
        # Add participant effects to predictions
        for i, p in enumerate(participants):
            if p in participant_map:
                p_idx = participant_map[p]
                if p_idx < len(participant_effects):
                    predictions[i] += participant_effects[p_idx]
    
    # Calculate metrics using float64 arrays
    target_mean = np.mean(target_vector)
    r2 = 1 - (np.sum((target_vector - predictions) ** 2) / 
              np.sum((target_vector - target_mean) ** 2))
    
    rmse = np.sqrt(np.mean((target_vector - predictions) ** 2))
    mae = np.mean(np.abs(target_vector - predictions))
    correlation, _ = stats.spearmanr(predictions, target_vector)
    
    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation)
    }

#=========================#
# Model Visualization    #
#=========================#
def plot_model_diagnostics(trace, output_base_path: str, inference_method: str) -> None:
    """
    Plot model diagnostic visualizations.
    
    Creates two plots:
    1. Trace plots for model parameters (saved as 'diagnostics_{inference_method}.png')
    2. Forest plot for fixed effects (saved as 'forest_{inference_method}.png')
    
    Related to train_bayesian_glmm() for model fitting and 
    evaluate_model_performance() for model assessment.
    
    Args:
        trace: ArviZ InferenceData object from train_bayesian_glmm()
        output_base_path: Base path template for saving plots
        inference_method: Name of inference method used
        
    See Also:
        train_bayesian_glmm: Main model training function
        evaluate_model_performance: Model evaluation metrics
    """    
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

#=========================#
# Model Persistence      #
#=========================#
def save_model_results(trace: az.InferenceData, 
                       model: pm.Model, 
                       base_filename: str) -> None:
    """
    Save model results to disk.
    
    Saves three files:
    1. {base_filename}_trace.nc: Model trace in NetCDF format
    2. {base_filename}_point_estimates.csv: Parameter point estimates
    3. {base_filename}_model_info.json: Model configuration and priors
    
    Args:
        trace: ArviZ InferenceData object from train_bayesian_glmm()
        model: Fitted PyMC model
        base_filename: Base filename for output files
        
    See Also:
        load_model_results: Function to load saved model results
        train_bayesian_glmm: Main model training function
    """
    logger.info(f"Saving model results to {base_filename}")
    try:
        # Save trace
        logger.debug("Saving model trace")
        az.to_netcdf(trace, filename=f"{base_filename}_trace.nc")
        
        # Save point estimates
        logger.debug("Saving point estimates")
        point_estimates = az.summary(trace)
        point_estimates.to_csv(f"{base_filename}_point_estimates.csv")
        
        # Save model configuration
        logger.debug("Saving model configuration")
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
            
        logger.info("Successfully saved all model results")
        
    except Exception as e:
        logger.error(f"Error saving model results: {str(e)}")
        raise

def load_model_results(
    base_filename: str
) -> Tuple[az.InferenceData, pd.DataFrame, Dict[str, Any]]:
    """
    Load saved model results from disk.
    
    Loads model results previously saved by save_model_results():
    1. Trace file (.nc)
    2. Point estimates (.csv)
    3. Model information (.json)
    
    Args:
        base_filename: Base filename used when saving results
        
    Returns:
        Tuple containing:
        - trace: ArviZ InferenceData object with model trace
        - point_estimates: DataFrame of parameter point estimates
        - model_info: Dictionary of model configuration and priors
        
    Raises:
        FileNotFoundError: If any required files are missing
        
    See Also:
        save_model_results: Function that saved these model results
    """
    logger.info(f"Loading model results from {base_filename}")
    try:
        # Load trace
        logger.debug("Loading model trace")
        trace = az.from_netcdf(f"{base_filename}_trace.nc")
        
        # Load point estimates
        logger.debug("Loading point estimates")
        point_estimates = pd.read_csv(f"{base_filename}_point_estimates.csv", 
                                    index_col=0)
        
        # Load model info
        logger.debug("Loading model configuration")
        with open(f"{base_filename}_model_info.json", "r") as f:
            model_info = json.load(f)
            
        logger.info("Successfully loaded all model results")
        return trace, point_estimates, model_info
        
    except FileNotFoundError as e:
        logger.error(f"Missing model file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading model results: {str(e)}")
        raise

#=========================#
# Helper Functions       #
#=========================#
def validate_inputs(feature_matrix: pd.DataFrame, 
                   target_vector: np.ndarray, 
                   participants: Optional[np.ndarray]) -> bool:
    """
    Validate data inputs for Bayesian GLMM model fitting.
    
    Args:
        feature_matrix: DataFrame of feature values, where each row corresponds to a bigram pair
        target_vector: Array of comfort scores or preference ratings
        participants: Optional array of participant IDs for mixed effects modeling
        
    Returns:
        bool: True if validation passes, raises ValueError otherwise
        
    Raises:
        ValueError: If dimensions don't match or if feature matrix contains null values
    """
    if len(feature_matrix) != len(target_vector):
        raise ValueError("Feature matrix and target vector must have same length")
    if participants is not None and len(participants) != len(target_vector):
        raise ValueError("Participants array must match target vector length")
    if np.any(pd.isnull(feature_matrix)):
        raise ValueError("Feature matrix contains null values")
    return True

def normalize_bigram_format(bigram):
    """
    Normalize different bigram formats to a consistent tuple format.
    
    Handles:
    - String tuples: "('a', 'b')"
    - List of single-char tuples: ["('a',)", "('b',)"]
    - Simple strings: "ab"
    - Existing tuples: ('a', 'b')
    
    Returns:
    - Tuple of two characters: ('a', 'b')
    """
    if isinstance(bigram, tuple) and len(bigram) == 2:
        return bigram
        
    if isinstance(bigram, list) and len(bigram) == 2:
        # Handle ["('a',)", "('b',)"] format
        chars = []
        for item in bigram:
            if isinstance(item, str) and item.startswith("('") and item.endswith(",)"):
                # Extract character from "('a',)" format
                char = item[2]
                chars.append(char)
        if len(chars) == 2:
            return tuple(chars)
            
    if isinstance(bigram, str):
        if bigram.startswith("(") and bigram.endswith(")"):
            # Handle "('a', 'b')" format
            try:
                return eval(bigram)
            except:
                pass
        elif len(bigram) == 2:
            # Handle "ab" format
            return tuple(bigram)
    
    return None

def normalize_scores(scores: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """
    Normalize comfort scores to 0-1 range.
    
    Args:
        scores: Dictionary of bigram pairs to raw scores
        
    Returns:
        Dictionary of bigram pairs to normalized scores between 0 and 1
    """
    if not scores:
        return {}
        
    values = np.array(list(scores.values()))
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) == 0:
        return scores
        
    min_score = np.min(valid_values)
    max_score = np.max(valid_values)
    
    # Handle case where all scores are the same
    if np.isclose(min_score, max_score):
        return {k: 0.5 for k in scores.keys()}
    
    return {
        bigram: (score - min_score) / (max_score - min_score)
        if not np.isnan(score) else 0.0
        for bigram, score in scores.items()
    }

def add_mirrored_scores(scores: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    """
    Add mirrored scores for right hand keys.
    
    Args:
        scores: Dictionary of bigram pairs to scores
        
    Returns:
        Dictionary including original and mirrored scores
    """
    left_keys = "qwertasdfgzxcvb"
    right_keys = "poiuy;lkjh/.,mn"
    key_mapping = dict(zip(left_keys, right_keys))
    
    mirrored = {}
    for bigram, score in scores.items():
        if isinstance(bigram, tuple) and len(bigram) == 2 and not np.isnan(score):
            char1, char2 = bigram
            right_char1 = key_mapping.get(char1, char1)
            right_char2 = key_mapping.get(char2, char2)
            right_bigram = (right_char1, right_char2)
            mirrored[right_bigram] = score
    
    # Combine original and mirrored scores
    combined = {**scores, **mirrored}
    logger.debug(f"Added {len(mirrored)} mirrored scores to {len(scores)} original scores")
    return combined



