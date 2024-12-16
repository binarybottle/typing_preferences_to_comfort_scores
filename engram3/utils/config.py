# engram3/config.py
"""
Configuration management and data structures for the keyboard preference system.

Provides:
  - Type-safe configuration validation using Pydantic
  - Data structures for preferences and features
  - Configuration classes for:
    - Feature selection and weighting
    - Model parameters and validation
    - Data loading and processing
    - Path management
    - Logging settings

All configuration classes include validation rules to ensure:
  - Required fields are present
  - Numeric ranges are valid
  - Weights sum to 1.0
  - Paths exist and are writable
  - Proper data types and formats
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, TypedDict, NamedTuple
from pathlib import Path
import numpy as np
from pydantic import BaseModel, validator, Field, field_validator

#------------------------------------------------
# data.py
#------------------------------------------------
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

#------------------------------------------------
# feature_extraction.py
#------------------------------------------------
@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    column_map: Dict[str, int]
    row_map: Dict[str, int]
    finger_map: Dict[str, int]
    engram_position_values: Dict[str, float]
    row_position_values: Dict[str, float]
    key_metrics: Dict[str, Dict[str, float]]

#------------------------------------------------
# feature_importance.py
#------------------------------------------------
class FeatureSelectionConfig(BaseModel):
    """Validate feature selection configuration."""
    metric_weights: Dict[str, float]

    @validator('metric_weights')
    def weights_must_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Metric weights must sum to 1.0, got {total}")
        return v

    class Config:
        validate_assignment = True

#------------------------------------------------
# feature_visualization.py
#------------------------------------------------
class VisualizationConfig(BaseModel):
    """Visualization settings."""
    dpi: int = 300

#------------------------------------------------
# model.py
#------------------------------------------------
class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class FeatureError(ModelError):
    """Raised for feature-related errors."""
    pass

class NotFittedError(ModelError):
    """Raised when trying to use model that hasn't been fit."""
    pass

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

#------------------------------------------------
# remaining Config
#------------------------------------------------
class ModelSettings(BaseModel):
    """Model configuration with validation."""
    chains: int = Field(gt=0)
    warmup: int = Field(gt=0)
    n_samples: int = Field(gt=0)
    adapt_delta: float = Field(gt=0, lt=1)
    max_treedepth: int = Field(gt=0)
    feature_scale: float = Field(gt=0)
    participant_scale: float = Field(gt=0)
    predictions_file: str
    model_file: str

    class Config:
        validate_assignment = True
    
class FeatureSelectionSettings(BaseModel):
    """Feature selection configuration."""
    metric_weights: Dict[str, float]
    metrics_file: str
    model_file: str

class FeaturesConfig(BaseModel):
    """Features configuration."""
    base_features: List[str]
    interactions: List[List[str]] = []

    @validator('interactions')
    def validate_interactions(cls, v: List[List[str]], values: Dict[str, Any]) -> List[List[str]]:
        """Validate interaction features."""
        if 'base_features' not in values:
            raise ValueError("base_features must be defined before interactions")
        
        base_features = values['base_features']
        for interaction in v:
            # Check each interaction is a list
            if not isinstance(interaction, list):
                raise ValueError(f"Interaction must be a list: {interaction}")
            
            # Check interaction has at least 2 features
            if len(interaction) < 2:
                raise ValueError(f"Interaction must have at least 2 features: {interaction}")
            
            # Check all features exist in base_features
            for feature in interaction:
                if feature not in base_features:
                    raise ValueError(f"Feature {feature} in interaction not found in base_features")
        
        return v

    def get_all_features(self) -> List[str]:
        """Get list of all features including interactions."""
        features = self.base_features.copy()
        
        # Interaction features
        for interaction in self.interactions:
            features.append('_x_'.join(interaction))
            
        return features

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

    class Config:
        arbitrary_types_allowed = True
