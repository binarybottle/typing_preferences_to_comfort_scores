#!/usr/bin/env python3
"""
Estimates memory requirements for preference learning pipeline.
"""
import psutil
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

def estimate_stan_memory(config_path: str, mode: str = "analyze_features") -> dict:
    """
    Estimate Stan memory requirements based on config settings.
    
    Args:
        config_path: Path to config.yaml
        mode: Either "analyze_features" or "select_features"
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get model parameters
    chains = config['model']['chains']
    warmup = config['model']['warmup']
    n_samples = config['model']['n_samples']
    total_iterations = chains * (warmup + n_samples)

    # Load dataset
    data_path = Path(config['data']['input_file'])
    df = pd.read_csv(data_path)
    n_preferences = len(df)
    print(f"Dataset size: {n_preferences} preferences")

    # Get feature counts
    base_features = config['features']['base_features']
    interactions = config['features']['interactions']
    control_features = config['features']['control_features']
    
    n_base_features = len(base_features)
    n_interactions = len(interactions)
    n_control = len(control_features)
    
    # In select_features mode, we use all features. In analyze_features, we use one at a time
    if mode == "select_features":
        total_features = n_base_features + n_interactions + n_control
        print(f"\nFeature counts (select_features mode):")
        print(f"Base features: {n_base_features}")
        print(f"Interactions: {n_interactions}")
        print(f"Control features: {n_control}")
        print(f"Total features: {total_features}")
    else:  # analyze_features mode
        total_features = 1 + n_control  # One feature at a time plus controls
        print(f"\nFeature counts (analyze_features mode):")
        print(f"Analyzing one feature at a time")
        print(f"Control features: {n_control}")
        print(f"Total features per analysis: {total_features}")

    # Memory estimation in bytes
    bytes_per_float = 8

    # Matrix sizes for BOTH models (baseline and feature model) in a fold
    feature_matrices_bytes = (
        n_preferences *       # number of rows
        total_features *      # number of columns
        bytes_per_float *     # size per value
        2 *                  # X1 and X2
        2                    # Two models per fold
    )
    
    control_matrices_bytes = (
        n_preferences * 
        n_control * 
        bytes_per_float * 
        2 *                  # C1 and C2
        2                    # Two models per fold
    )

    # Stan storage per chain
    stan_parameter_storage = (
        total_features +     # beta coefficients
        n_control +         # gamma coefficients
        n_preferences       # predictions
    ) * bytes_per_float

    # Total Stan memory per chain including gradients and adaption
    stan_chain_memory = (
        stan_parameter_storage * 5 *    # Parameters + gradients + working memory
        total_iterations              # Total iterations (warmup + sampling)
    )

    # Total Stan memory for all chains for both models
    stan_total_memory = stan_chain_memory * chains * 2  # Two models per fold

    # Memory per CV fold
    cv_fold_memory = (
        feature_matrices_bytes +
        control_matrices_bytes +
        stan_total_memory
    )

    print(f"\nDetailed Memory Breakdown ({mode} mode, per fold):")
    print(f"Feature matrices: {feature_matrices_bytes / (1024**3):.2f} GB")
    print(f"Control matrices: {control_matrices_bytes / (1024**3):.2f} GB")
    print(f"Stan memory (all chains): {stan_total_memory / (1024**3):.2f} GB")

    # Total memory with safety factor
    safety_factor = 1.3
    total_memory = cv_fold_memory * safety_factor  # Only 1 fold at a time

    # Get system memory info
    mem = psutil.virtual_memory()

    if mode == "select_features":
        # For select_features, account for concurrent model runs
        total_memory *= 2  # Two models running concurrently

    return {
        'mode': mode,
        'estimated_bytes': total_memory,
        'estimated_gb': total_memory / (1024**3),
        'available_bytes': mem.available,
        'available_gb': mem.available / (1024**3),
        'is_sufficient': mem.available > total_memory,
        'details': {
            'feature_matrices_gb': feature_matrices_bytes / (1024**3),
            'control_matrices_gb': control_matrices_bytes / (1024**3),
            'stan_per_fold_gb': cv_fold_memory / (1024**3),
            'total_cv_memory_gb': cv_fold_memory / (1024**3)  # Only 1 fold at a time
        }
    }

def get_memory_info():
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / (1024**3),
        'available_gb': mem.available / (1024**3),
        'used_gb': (mem.total - mem.available) / (1024**3),
        'percent_used': mem.percent
    }

def main():
    config_path = 'config.yaml'
    
    # Get estimates for both modes
    analyze_estimates = estimate_stan_memory(config_path, "analyze_features")
    select_estimates = estimate_stan_memory(config_path, "select_features")
    mem_info = get_memory_info()

    print("\nSystem Memory:")
    print(f"Total RAM: {mem_info['total_gb']:.2f} GB")
    print(f"Currently used: {mem_info['used_gb']:.2f} GB ({mem_info['percent_used']:.1f}%)")
    print(f"Currently available: {mem_info['available_gb']:.2f} GB")

    print("\nMemory Requirements by Mode:")
    
    print("\n1. analyze_features mode:")
    print(f"Total estimated memory needed: {analyze_estimates['estimated_gb']:.2f} GB")
    print("Breakdown per fold:")
    print(f"- Feature matrices: {analyze_estimates['details']['feature_matrices_gb']:.2f} GB")
    print(f"- Control matrices: {analyze_estimates['details']['control_matrices_gb']:.2f} GB")
    print(f"- Stan memory: {analyze_estimates['details']['stan_per_fold_gb']:.2f} GB")
    
    print("\n2. select_features mode:")
    print(f"Total estimated memory needed: {select_estimates['estimated_gb']:.2f} GB")
    print("Breakdown per fold:")
    print(f"- Feature matrices: {select_estimates['details']['feature_matrices_gb']:.2f} GB")
    print(f"- Control matrices: {select_estimates['details']['control_matrices_gb']:.2f} GB")
    print(f"- Stan memory: {select_estimates['details']['stan_per_fold_gb']:.2f} GB")
    
    # Check if memory is sufficient for both modes
    max_memory = max(analyze_estimates['estimated_gb'], select_estimates['estimated_gb'])
    if max_memory > mem_info['available_gb']:
        print("\n⚠️  WARNING: Available memory may not be sufficient!")
        print(f"Need {max_memory:.2f} GB but only {mem_info['available_gb']:.2f} GB available")
    else:
        print("\n✅ Available memory should be sufficient for both modes")

if __name__ == '__main__':
    main()