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
   n_samples: int = 10000,
   chains: int = 8,
   target_accept: float = 0.85
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
        n_samples: Number of posterior samples to draw
        chains: Number of independent MCMC chains
        target_accept: Target acceptance rate for proposals in the NUTS sampler

    Returns:
        Tuple containing:
        - trace: ArviZ InferenceData object with posterior samples
        - model: Fitted PyMC model
        - priors: Dictionary of model prior distributions
    """
    logger.info("MODEL: Starting Bayesian GLMM training")
    logger.info("MODEL: Validating features...")
    
    # Assertions to catch potential float values early
    assert isinstance(n_samples, int), "n_samples must be an integer"
    assert isinstance(chains, int), "chains must be an integer"

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

    effects_sigma = 2
    with pm.Model() as model:
        # Priors for design features
        design_effects = {
            feat: pm.Normal(feat, mu=0, sigma=effects_sigma)
            for feat in design_features
        }

        # Priors for control features
        control_effects = {
            feat: pm.Normal(feat, mu=0, sigma=effects_sigma)
            for feat in control_features
        }

        # Random effects for participants
        participant_sigma = pm.HalfNormal('participant_sigma', sigma=effects_sigma)
        participant_offset = pm.Normal('participant_offset',
                                    mu=0,
                                    sigma=participant_sigma,
                                    shape=n_participants)

        # Error term
        sigma = pm.HalfNormal('sigma', sigma=effects_sigma)

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

        # Likelihood with potential for outliers
        likelihood = pm.StudentT('likelihood', 
                               nu=4,  # Degrees of freedom - more robust than Normal
                               mu=mu, 
                               sigma=sigma, 
                               observed=target_vector)

        # Different sampling based on inference method
        logger.info(f"Using {inference_method} inference")
        
        # Add adaptation steps for better sampling
        with model:
            # Start with MAP to find good initial values
            start = pm.find_MAP()

            trace = pm.sample(
                draws=n_samples,              # Actual samples kept for inference (match config)
                tune=n_samples//2,             # "Burn-in" period: sampler adjusts parameters and discards samples
                chains=chains,                # Independent sampling sequences, used to assess convergence (match config)
                cores=chains,                 # Number of CPU cores for parallel chain sampling
                target_accept=target_accept,  # Target acceptance rate for proposals in NUTS sampler (0.8-0.9 good for exploration)
                init='jitter+adapt_diag',
                initvals=start,            
                return_inferencedata=True,
                compute_convergence_checks=True,
                random_seed=42                # For reproducibility
            )

            # Collect priors
            priors = {
                'design_effects': design_effects,
                'control_effects': control_effects,
                'participant_sigma': participant_sigma,
                'sigma': sigma
            }

        logger.info("Model training completed successfully")
        logger.info("\nSampling diagnostics:")
        summary = az.summary(trace)
        for param in summary.index:
            ess = summary.loc[param, 'ess_bulk']
            rhat = summary.loc[param, 'r_hat']
            logger.info(f"{param}: ESS={ess:.1f}, R-hat={rhat:.3f}")

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
    
    # Process bigram pairs differently than index normalization
    scores = {}
    sample_size = next(iter(posterior_samples.values())).shape[0]
    
    # Iterate through original feature matrix
    for idx, features in feature_matrix.iterrows():
        effect = np.zeros(sample_size, dtype=np.float64)
        
        # Calculate effect for each feature
        for param, samples in posterior_samples.items():
            if param in features:
                feature_value = features[param]
                if not np.isnan(feature_value):
                    effect += samples * feature_value
        
        # Extract bigram from the index
        try:
            # If idx is already a tuple of two chars
            if isinstance(idx, tuple) and len(idx) == 2 and all(isinstance(x, str) for x in idx):
                bigram = idx
            # If idx is a tuple of tuples (bigram pair)
            elif isinstance(idx, tuple) and len(idx) == 2 and all(isinstance(x, tuple) for x in idx):
                # Take the first bigram of the pair
                bigram = idx[0]
            # If idx is a string representation
            elif isinstance(idx, str):
                if idx.startswith("(") and idx.endswith(")"):
                    try:
                        evaluated = eval(idx)
                        if isinstance(evaluated, tuple):
                            if len(evaluated) == 2 and all(isinstance(x, str) for x in evaluated):
                                bigram = evaluated
                            elif len(evaluated) == 2 and all(isinstance(x, tuple) for x in evaluated):
                                bigram = evaluated[0]
                    except:
                        logger.warning(f"Could not evaluate bigram string: {idx}")
                        continue
                elif len(idx) == 2:
                    bigram = tuple(idx)
            
            if bigram and len(bigram) == 2:
                if effect.size > 0 and not np.all(np.isnan(effect)):
                    scores[bigram] = float(np.nanmean(effect))
            
        except Exception as e:
            logger.warning(f"Could not process bigram index: {idx}, error: {str(e)}")
            continue
    
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
    """
    Validate comfort scores against held-out test data.
    
    Data structure:
    - test_data.bigram_pairs: List of [bigram1, bigram2] where each bigram is [char1, char2]
    - comfort_scores: Dict of (char1, char2) -> float
    - Implicit preference: bigram1 is preferred over bigram2
    """
    validation_metrics = {}
    
    if not hasattr(test_data, 'bigram_pairs'):
        logger.error("Test data missing bigram pairs attribute")
        return validation_metrics
        
    logger.info(f"Test data bigram pairs: {len(test_data.bigram_pairs)}")
    
    # Convert scores to preferences
    predicted_prefs = []
    actual_prefs = []
    skipped_pairs = set()
    
    for bigram_pair in test_data.bigram_pairs:
        try:
            # Each bigram_pair should be [bigram1, bigram2]
            if not isinstance(bigram_pair, list) or len(bigram_pair) != 2:
                logger.warning(f"Invalid bigram pair format: {bigram_pair}")
                continue
            
            bigram1, bigram2 = bigram_pair
            
            # Convert list bigrams to tuples for dictionary lookup
            tuple1 = tuple(bigram1) if isinstance(bigram1, list) else bigram1
            tuple2 = tuple(bigram2) if isinstance(bigram2, list) else bigram2
            
            logger.debug(f"Looking up bigrams: {tuple1} and {tuple2}")
            
            # Get comfort scores
            score1 = comfort_scores.get(tuple1)
            score2 = comfort_scores.get(tuple2)
            
            if score1 is not None and score2 is not None:
                # Calculate predicted preference (score difference)
                pred_diff = score1 - score2
                # Actual preference is 1.0 (bigram1 preferred over bigram2)
                predicted_prefs.append(pred_diff)
                actual_prefs.append(1.0)  # bigram1 is always preferred
                logger.debug(f"Added: pred={pred_diff:.3f}, actual=1.0")
            else:
                if (tuple1, tuple2) not in skipped_pairs:
                    missing = []
                    if score1 is None:
                        missing.append(str(tuple1))
                    if score2 is None:
                        missing.append(str(tuple2))
                    logger.debug(f"Skipping bigrams - missing scores: {missing}")
                    skipped_pairs.add((tuple1, tuple2))
                    
        except Exception as e:
            logger.warning(f"Error processing bigram pair {bigram_pair}: {str(e)}")
            continue
    
    # Convert to numpy arrays with explicit float type
    predicted_prefs = np.array(predicted_prefs, dtype=np.float64)
    actual_prefs = np.array(actual_prefs, dtype=np.float64)
    
    logger.info(f"Predicted prefs: {len(predicted_prefs)}, Actual prefs: {len(actual_prefs)}")
    
    if len(predicted_prefs) > 0:
        try:
            # Calculate accuracy (how often we predict the correct preference)
            # Positive prediction means we agree with the preference
            sign_matches = (predicted_prefs > 0)
            validation_metrics['accuracy'] = float(np.mean(sign_matches))
            logger.info(f"Accuracy: {validation_metrics['accuracy']:.3f}")
            
            if len(predicted_prefs) > 1:
                # Calculate correlation with preference strength
                correlation, p_value = stats.spearmanr(predicted_prefs, actual_prefs)
                validation_metrics['correlation'] = float(correlation)
                validation_metrics['correlation_p_value'] = float(p_value)
                logger.info(f"Correlation: {validation_metrics['correlation']:.3f} (p={p_value:.3f})")
            
            # Add prediction distribution metrics
            validation_metrics['correct_direction'] = float(np.mean(predicted_prefs > 0))
            validation_metrics['avg_preference_strength'] = float(np.mean(np.abs(predicted_prefs)))
            logger.info(f"Predictions in correct direction: {validation_metrics['correct_direction']:.2%}")
            logger.info(f"Average preference strength: {validation_metrics['avg_preference_strength']:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            logger.debug(f"Predicted prefs sample: {predicted_prefs[:5]}")
            logger.debug(f"Actual prefs sample: {actual_prefs[:5]}")
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
    
    threshold = 0.5  # or whatever makes sense for your scale
    accuracy = np.mean(np.abs(predictions - target_vector) < threshold)
    
    return {
        'r2': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'correlation': float(correlation),
        'accuracy': float(accuracy)
    }

#=========================#
# Model Visualization    #
#=========================#
def plot_model_diagnostics(trace, output_base_path: str, inference_method: str) -> None:
    """
    Plot model diagnostics with better formatting and documentation.
    
    Creates two types of plots:
    1. Trace plots (left) show parameter values across MCMC iterations
       - Each line is a different chain
       - Good mixing shown by overlapping "fuzzy caterpillars"
       - Different colors indicate different chains
       
    2. Posterior distributions (right) show parameter value distributions
       - Shows where parameter values are concentrated
       - Wider distributions indicate more uncertainty
       - Narrower peaks indicate more certainty
    """
    try:
        if trace is None:
            logger.warning("No trace provided for plotting")
            return
            
        available_vars = [var for var in trace.posterior.variables 
                         if not any(dim in var for dim in ['chain', 'draw', 'dim'])]
        
        if available_vars:
            # Increase figure size and spacing
            plt.figure(figsize=(15, len(available_vars)))
            az.plot_trace(trace, var_names=available_vars)
            plt.suptitle('Model Parameter Traces and Distributions\n\n' +
                        'Left: Parameter values across iterations (chains)\n' +
                        'Right: Parameter value distributions',
                        y=0.98, fontsize=14)
            plt.subplots_adjust(hspace=0.8)  # Increase vertical spacing
            plt.savefig(output_base_path.format(inference_method=inference_method),
                       bbox_inches='tight')
            plt.close()
        
        # Forest plot with better margins and documentation
        if available_vars:
            plt.figure(figsize=(12, len(available_vars) * 0.5))
            az.plot_forest(trace, var_names=available_vars)
            plt.title('Parameter Estimates with 95% Highest Density Interval (HDI)\n' +
                     'Dots show median estimates, bars show uncertainty ranges\n' +
                     'Parameters crossing zero may not have reliable effects',
                     pad=20)
            plt.tight_layout()
            plt.savefig(output_base_path.format(inference_method=inference_method)
                       .replace('diagnostics', 'forest'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
    except Exception as e:
        logger.warning(f"Could not create diagnostic plots: {str(e)}")

def save_plot_documentation(output_dir: str) -> None:
    """Save documentation explaining the diagnostic plots."""
    doc_path = os.path.join(output_dir, 'plot_documentation.txt')
    
    with open(doc_path, 'w') as f:
        f.write("Model Diagnostic Plot Documentation\n")
        f.write("================================\n\n")
        
        f.write("Trace Plots (Left Side):\n")
        f.write("----------------------\n")
        f.write("- Show parameter values across MCMC iterations\n")
        f.write("- Multiple chains shown in different colors\n")
        f.write("- Good mixing shown by overlapping 'fuzzy caterpillars'\n")
        f.write("- Chains should explore similar regions\n\n")
        
        f.write("Posterior Distributions (Right Side):\n")
        f.write("--------------------------------\n")
        f.write("- Show distribution of parameter values\n")
        f.write("- Narrow peaks indicate more certainty\n")
        f.write("- Wide distributions indicate uncertainty\n")
        f.write("- Multi-modal distributions may indicate problems\n\n")
        
        f.write("Forest Plot:\n")
        f.write("-----------\n")
        f.write("- Shows parameter estimates with uncertainty\n")
        f.write("- Dots indicate median estimates\n")
        f.write("- Thick bars show 50% HDI (Highest Density Interval)\n")
        f.write("- Thin bars show 95% HDI\n")
        f.write("- Parameters crossing zero may not be reliable\n")
        f.write("- HDI shows most credible parameter values\n")

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
    Normalize bigrams into a consistent tuple format.

    Handles:
    - Bigram pairs: (('g', 'c'), ('q', 'w')) -> ('g', 'c')
    - Single bigrams: ('a', 'b') -> ('a', 'b')
    - String bigrams: "ab" -> ('a', 'b')
    - String tuples: "('a', 'b')" -> ('a', 'b')
    - List format: ['a', 'b'] -> ('a', 'b')
    - Single character tuples: ('a',) -> a for building pairs
    
    Returns:
    - Tuple of two characters: ('a', 'b')
    - Single character string for building pairs
    - None if the input cannot be normalized
    """
    logger.debug(f"Normalizing bigram: {type(bigram).__name__}={bigram}")
    
    try:
        # Case 1: Single character handling
        if isinstance(bigram, str) and len(bigram) == 1:
            logger.debug(f"Found single character: {bigram}")
            return bigram
        if isinstance(bigram, tuple) and len(bigram) == 1:
            if isinstance(bigram[0], str) and len(bigram[0]) == 1:
                logger.debug(f"Found single character tuple: {bigram[0]}")
                return bigram[0]
            
        # Case 2: Handle string representation of single character tuple
        if isinstance(bigram, str) and bigram.startswith("('") and bigram.endswith(",)"):
            char = bigram[2]
            if len(char) == 1:
                logger.debug(f"Found string single character tuple: {char}")
                return char
            
        # Case 3: Handle bigram pairs
        if isinstance(bigram, tuple) and len(bigram) == 2:
            if all(isinstance(x, tuple) and len(x) == 2 for x in bigram):
                # It's a bigram pair, take the first one
                logger.debug(f"Found bigram pair, using first: {bigram[0]}")
                return bigram[0]
            elif all(isinstance(x, str) and len(x) == 1 for x in bigram):
                # It's already a properly formatted single bigram
                logger.debug(f"Found properly formatted bigram: {bigram}")
                return bigram
        
        # Case 4: Handle list format
        if isinstance(bigram, list):
            if len(bigram) == 2 and all(isinstance(x, str) and len(x) == 1 for x in bigram):
                logger.debug(f"Converting list to tuple: {tuple(bigram)}")
                return tuple(bigram)
        
        # Case 5: Handle string formats
        if isinstance(bigram, str):
            # Handle "('a', 'b')" format
            if bigram.startswith("(") and bigram.endswith(")"):
                try:
                    evaluated = eval(bigram)
                    logger.debug(f"Evaluating string representation: {evaluated}")
                    return normalize_bigram_format(evaluated)
                except:
                    pass
            # Handle simple "ab" format
            elif len(bigram) == 2:
                logger.debug(f"Converting string to tuple: {tuple(bigram)}")
                return tuple(bigram)
        
        logger.warning(f"Could not normalize bigram format: {bigram}")
        return None
        
    except Exception as e:
        logger.warning(f"Error normalizing bigram {bigram}: {str(e)}")
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



