# bigram_typing_preferences_to_comfort_scores/config.py
"""
Configuration and data structure definitions for keyboard layout optimization.

Core Components:
  1. Data Structures:
    - Preference: Typing preference data container
    - FeatureConfig: Feature extraction settings
    - ModelPrediction: Prediction output structure
    - StabilityMetrics: Feature stability measurements
    
  2. Feature Configuration:
    - FeatureSelectionSettings: Feature selection parameters
    - FeatureSelectionThresholds: Selection cutoff values
    - FeaturesConfig: Feature definitions and interactions
    
  3. Model Configuration:
    - ModelSettings: MCMC sampling parameters
    - DataConfig: Dataset parameters
    - RecommendationsConfig: Bigram recommendation settings
    
  4. System Configuration:
    - LoggingConfig: Log settings
    - PathsConfig: Directory management
    - VisualizationConfig: Plot settings
    
  5. Validation Rules:
    - Numeric range verification
    - Path existence checking
    - Feature interaction validation
    - Weight normalization
    - Type safety enforcement

Exceptions:
    ModelError: Base model exception
    FeatureError: Feature processing error
    NotFittedError: Unfit model usage error
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, TypedDict, NamedTuple
from pathlib import Path
import numpy as np
from pydantic import BaseModel, validator, Field, field_validator

#--------------------------------------------
# Base exceptions
#--------------------------------------------   
class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class FeatureError(ModelError):
    """Raised for feature-related errors."""
    pass

class NotFittedError(ModelError):
    """Raised when trying to use model that hasn't been fit."""
    pass

#--------------------------------------------
# Core data structures
#--------------------------------------------   
@dataclass
class Preference:
    """Single preference instance with all needed data."""
    bigram1: str
    bigram2: str
    participant_id: str
    preferred: bool
    features1: Dict[str, float]
    features2: Dict[str, float]
    confidence: Optional[float] = None
    typing_time1: Optional[float] = None
    typing_time2: Optional[float] = None

    def __str__(self) -> str:
        """Return human-readable preference."""
        preferred_bigram = self.bigram1 if self.preferred else self.bigram2
        other_bigram = self.bigram2 if self.preferred else self.bigram1
        return f"'{preferred_bigram}' preferred over '{other_bigram}'"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"Preference('{self.bigram1}' vs '{self.bigram2}', preferred: {self.bigram1 if self.preferred else self.bigram2})"

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    column_map: Dict[str, int]
    row_map: Dict[str, int]
    finger_map: Dict[str, int]
    engram_position_values: Dict[str, float]
    row_position_values: Dict[str, float]
    angles: Dict[Tuple[str, str], float]
    bigrams: List[str]  # List of English bigrams ordered by frequency
    bigram_frequencies_array: np.ndarray  # Array of corresponding frequency values

@dataclass
class ModelPrediction:
    """Structured prediction output."""
    probability: float
    uncertainty: float
    features_used: List[str]
    computation_time: float

class StabilityMetrics(TypedDict):
    """Stability metric results."""
    effect_cv: float
    sign_consistency: float
    relative_range: float

#--------------------------------------------
# Feature-related configuration
#--------------------------------------------   
class FeatureSelectionSettings(BaseModel):
    """Feature selection configuration."""
    importance_threshold: float = Field(
        default=0.05,  # Threshold for aligned effect importance
        gt=0.0         # Must be positive
    )
    cross_validation: Dict[str, int] = Field(
        default={
            'n_splits': 5,        # Number of CV folds for evaluating importance
            'min_fold_size': 100  # Minimum number of preferences per fold
        }
    )
    metrics_file: str  # File to store feature metrics
    model_file: str    # File to store feature selection model
    
class FeatureSelectionThresholds(BaseModel):
    model_effect: float = 0.2
    effect_consistency: float = 0.5
    predictive_power: float = 0.1

class FeaturesConfig(BaseModel):
    """Configuration for features and their interactions."""
    base_features: List[str]
    control_features: List[str]
    interactions: List[List[str]]

    @validator('interactions')
    def validate_interactions(cls, v: List[List[str]], values: Dict[str, Any]) -> List[List[str]]:
        """Validate that interaction features only use base features."""
        if 'base_features' not in values:
            raise ValueError("base_features must be defined before interactions")
        
        base_features = values['base_features']
        for interaction in v:
            # Allow 2 or more features in interactions
            if len(interaction) < 2:
                raise ValueError(f"Each interaction must have at least 2 features: {interaction}")
            # Check that each feature in the interaction is a valid base feature
            for feature in interaction:
                if feature not in base_features:
                    raise ValueError(f"Interaction feature {feature} not in base_features")
            # Ensure no duplicate features in a single interaction
            if len(set(interaction)) != len(interaction):
                raise ValueError(f"Duplicate features in interaction: {interaction}")
        return v

    @validator('control_features')
    def validate_control_features(cls, v: List[str], values: Dict[str, Any]) -> List[str]:
        """Validate that control features don't overlap with base features."""
        if 'base_features' not in values:
            raise ValueError("base_features must be defined before control_features")
        
        base_features = values['base_features']
        overlap = set(v) & set(base_features)
        if overlap:
            raise ValueError(f"Control features overlap with base features: {overlap}")
        return v

    def create_interaction_name(self, features: List[str]) -> str:
        """Create standardized name for an interaction."""
        return '_x_'.join(sorted(features))  # Sort for consistency

    def get_all_interaction_names(self) -> List[str]:
        """Get standardized names for all interactions."""
        return [self.create_interaction_name(interaction) 
                for interaction in self.interactions]

    class Config:
        validate_assignment = True
        
#--------------------------------------------
# Model settings
#--------------------------------------------   
class ModelSettings(BaseModel):
    """Model configuration with validation."""
    chains: int = Field(gt=0)
    warmup: int = Field(gt=0)
    n_samples: int = Field(gt=0)
    adapt_delta: float = Field(gt=0, lt=1)
    max_treedepth: int = Field(gt=0)
    feature_scale: float = Field(gt=0)
    participant_scale: float = Field(gt=0)
    required_temp_mb: int = 2000  # Default value for required temp space
    bigram_comfort_predictions_file: str = "output/data/estimated_bigram_scores.csv"
    key_comfort_predictions_file: str = "output/data/key_comfort_scores.csv"
    model_file: str="output/data/bigram_score_prediction_model.pkl"
    
class DataConfig(BaseModel):
    """Data configuration."""
    input_file: str
    splits: Dict[str, Any]
    layout: Dict[str, List[str]]

class RecommendationsConfig(BaseModel):
    """Recommendations configuration."""
    weights: Dict[str, float]
    n_recommendations: int = Field(gt=0)
    max_candidates: int = Field(gt=0)
    recommendations_file: str
    min_comfort_score: float = Field(default=0.333)
    min_feature_coverage: float = Field(default=0.168)
    min_diversity: float = Field(default=0.632)

    @validator('weights')
    def weights_must_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Recommendation weights must sum to 1.0, got {total}")
        return v

    @validator('max_candidates')
    def max_candidates_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_candidates must be positive")
        return v

    @validator('n_recommendations')
    def n_recommendations_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_recommendations must be positive")
        return v

#--------------------------------------------
# System configuration
#--------------------------------------------   
class LoggingConfig(BaseModel):
    """Logging configuration."""
    format: str
    console_level: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file_level: str = Field(pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

class PathsConfig(BaseModel):
    """Path configuration."""
    root_dir: Path
    metrics_dir: Path
    plots_dir: Path
    logs_dir: Path

    @validator('*')
    def create_directories(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

class VisualizationConfig(BaseModel):
    """Visualization settings."""
    dpi: int = 300
    alpha: float = 0.6
    figure_size: tuple = (12, 8)  # Add this line
    color_map: str = "viridis"

#--------------------------------------------
# Main configuration
#--------------------------------------------   
class Config(BaseModel):
    """Complete configuration with nested validation."""
    paths: PathsConfig
    model: ModelSettings
    feature_selection: FeatureSelectionSettings
    features: FeaturesConfig
    data: DataConfig
    recommendations: RecommendationsConfig
    logging: LoggingConfig
    visualization: VisualizationConfig

    @validator('features')
    def validate_feature_config(cls, v: FeaturesConfig) -> FeaturesConfig:
        """Additional validation for feature configuration."""
        # Ensure we have at least one base feature
        if not v.base_features:
            raise ValueError("Must specify at least one base feature")
        
        # Ensure control features are specified (empty list is ok)
        if not hasattr(v, 'control_features'):
            v.control_features = []
        
        return v

    class Config:
        arbitrary_types_allowed = True