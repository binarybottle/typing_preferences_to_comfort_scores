"""
Data Preprocessing Module

This module handles data loading, cleaning, and preparation for keyboard layout analysis.
"""
import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict, Optional
import ast
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass, replace
import json

logger = logging.getLogger(__name__)

@dataclass
class ProcessedData:
    """Container for processed data."""
    bigram_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]]
    feature_matrix: pd.DataFrame
    target_vector: np.ndarray
    participants: np.ndarray
    typing_times: Optional[np.ndarray]

class DataPreprocessor:
    """Class for handling data preprocessing tasks."""
    
    def __init__(self, csv_file_path: str):
        """
        Initialize preprocessor with data file path.
        
        Args:
            csv_file_path: Path to the CSV file containing bigram data
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.bigram_pairs = None
        self.feature_matrix = None
        self.target_vector = None
        self.participants = None
        self.typing_times = None
        
    def load_data(self) -> None:
        """Load and perform initial data validation."""
        logger.info(f"Loading data from {self.csv_file_path}")
        try:
            self.data = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded {len(self.data)} rows of data")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_bigram_pairs(self) -> None:
        """Convert string representations of bigram pairs to tuples."""
        logger.info("Preparing bigram pairs")
        try:
            # Convert string representations to actual tuples
            bigram_pairs = [ast.literal_eval(pair) for pair in self.data['bigram_pair']]
            
            # Split each bigram in the pair into its individual characters
            self.bigram_pairs = [
                ((bigram1[0], bigram1[1]), (bigram2[0], bigram2[1]))
                for bigram1, bigram2 in bigram_pairs
            ]
            
            logger.info(f"Processed {len(self.bigram_pairs)} bigram pairs")
        except Exception as e:
            logger.error(f"Error preparing bigram pairs: {str(e)}")
            raise
            
    def extract_target_vector(self) -> None:
        """Extract and prepare the target vector from slider values."""
        logger.info("Extracting target vector")
        try:
            self.target_vector = self.data['abs_sliderValue'].to_numpy()
            logger.info(f"Extracted target vector of length {len(self.target_vector)}")
        except Exception as e:
            logger.error(f"Error extracting target vector: {str(e)}")
            raise
            
    def process_participants(self) -> None:
        """Process participant IDs into numeric codes."""
        logger.info("Processing participant IDs")
        try:
            # Convert participant IDs to categorical codes
            self.participants = pd.Categorical(self.data['user_id']).codes
            self.participants = self.participants.astype(int)
            
            n_participants = len(np.unique(self.participants))
            logger.info(f"Processed {n_participants} unique participants")
        except Exception as e:
            logger.error(f"Error processing participants: {str(e)}")
            raise
            
    def extract_typing_times(self) -> None:
        """Extract typing times if available."""
        logger.info("Extracting typing times")
        try:
            if 'chosen_bigram_time' in self.data.columns:
                self.typing_times = self.data['chosen_bigram_time'].to_numpy()
                logger.info(f"Extracted {len(self.typing_times)} typing times")
            else:
                logger.warning("No typing times found in data")
                self.typing_times = None
        except Exception as e:
            logger.error(f"Error extracting typing times: {str(e)}")
            raise

    def create_feature_matrix(
            self, 
            all_feature_differences: Dict, 
            feature_names: List[str],
            config: Dict  # Add config parameter
        ) -> None:
            """
            Create feature matrix using provided feature differences, 
            and including timing data.
            """
            logger.info("Creating feature matrix")
            try:
                # Create initial feature matrix
                valid_pairs = [
                    bigram for bigram in self.bigram_pairs
                    if bigram in all_feature_differences
                ]
                
                feature_matrix_data = [
                    all_feature_differences[bigram_pair]
                    for bigram_pair in valid_pairs
                ]
                
                self.feature_matrix = pd.DataFrame(
                    feature_matrix_data,
                    columns=feature_names,
                    index=valid_pairs
                )
                
                # Add typing time as a feature
                if self.typing_times is not None:
                    self.feature_matrix['typing_time'] = self.typing_times

                # Add interactions if enabled
                if config['features']['interactions']['enabled']:
                    for pair in config['features']['interactions']['pairs']:
                        interaction_name = f"{pair[0]}_{pair[1]}"
                        self.feature_matrix[interaction_name] = (
                            self.feature_matrix[pair[0]] * self.feature_matrix[pair[1]]
                        )
                
                self._update_arrays_after_filtering(valid_pairs)
                
                logger.info(f"Created feature matrix with shape {self.feature_matrix.shape}")
            except Exception as e:
                logger.error(f"Error creating feature matrix: {str(e)}")
                raise

    def _update_arrays_after_filtering(self, valid_pairs: List[Tuple]) -> None:
        """
        Update arrays after filtering bigram pairs.
        
        Args:
            valid_pairs: List of valid bigram pairs
        """
        # Get indices of valid pairs
        valid_indices = [i for i, pair in enumerate(self.bigram_pairs)
                        if pair in valid_pairs]
        
        # Update arrays
        self.bigram_pairs = valid_pairs
        self.target_vector = self.target_vector[valid_indices]
        self.participants = self.participants[valid_indices]
        if self.typing_times is not None:
            self.typing_times = self.typing_times[valid_indices]

    def add_feature_interactions(self, 
                               interaction_features: List[Tuple[str, str]]) -> None:
        """
        Add interaction terms between specified features.
        
        Args:
            interaction_features: List of tuples containing feature pairs to interact
        """
        logger.info("Adding feature interactions")
        try:
            for feature1, feature2 in interaction_features:
                interaction_name = f"{feature1}_{feature2}"
                self.feature_matrix[interaction_name] = (
                    self.feature_matrix[feature1] * self.feature_matrix[feature2]
                )
            
            logger.info(f"Added {len(interaction_features)} feature interactions")
        except Exception as e:
            logger.error(f"Error adding feature interactions: {str(e)}")
            raise

    def scale_features(self) -> None:
        """Scale features using StandardScaler."""
        logger.info("Scaling features")
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(self.feature_matrix)
            self.feature_matrix = pd.DataFrame(
                scaled_features,
                columns=self.feature_matrix.columns,
                index=self.feature_matrix.index
            )
            logger.info("Features scaled successfully")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def validate_data(self) -> bool:
        """
        Perform validation checks on processed data.
        
        Returns:
            bool: True if validation passes
        """
        logger.info("Validating processed data")
        try:
            # Check for null values
            if self.feature_matrix.isnull().any().any():
                raise ValueError("Feature matrix contains null values")
                
            # Check dimensions
            if len(self.feature_matrix) != len(self.target_vector):
                raise ValueError("Feature matrix and target vector dimensions don't match")
                
            # Check participant array
            if len(self.participants) != len(self.target_vector):
                raise ValueError("Participant array dimension doesn't match target vector")
                
            # Check typing times if present
            if self.typing_times is not None:
                if len(self.typing_times) != len(self.target_vector):
                    raise ValueError("Typing times dimension doesn't match target vector")
            
            logger.info("Data validation passed")
            return True
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise

    def get_processed_data(self) -> ProcessedData:
        """
        Return processed data in a structured format.
        
        Returns:
            ProcessedData object containing all processed data
        """
        if not self.validate_data():
            raise ValueError("Data validation failed")
            
        return ProcessedData(
            bigram_pairs=self.bigram_pairs,
            feature_matrix=self.feature_matrix,
            target_vector=self.target_vector,
            participants=self.participants,
            typing_times=self.typing_times
        )

    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to files.
        
        Args:
            output_path: Base path for saving files
        """
        logger.info(f"Saving processed data to {output_path}")
        try:
            # Save feature matrix
            self.feature_matrix.to_csv(f"{output_path}_features.csv")
            
            # Save other arrays
            np.save(f"{output_path}_target.npy", self.target_vector)
            np.save(f"{output_path}_participants.npy", self.participants)
            if self.typing_times is not None:
                np.save(f"{output_path}_typing_times.npy", self.typing_times)
            
            # Save bigram pairs
            with open(f"{output_path}_bigram_pairs.json", 'w') as f:
                # Convert tuples to lists for JSON serialization
                serializable_pairs = [
                    [[b1[0], b1[1]], [b2[0], b2[1]]]
                    for (b1, b2) in self.bigram_pairs
                ]
                json.dump(serializable_pairs, f)
            
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    @staticmethod
    def load_processed_data(base_path: str) -> ProcessedData:
        """
        Load previously processed data.
        
        Args:
            base_path: Base path where processed data was saved
            
        Returns:
            ProcessedData object containing loaded data
        """
        logger.info(f"Loading processed data from {base_path}")
        try:
            # Load feature matrix
            feature_matrix = pd.read_csv(f"{base_path}_features.csv", index_col=0)
            
            # Load arrays
            target_vector = np.load(f"{base_path}_target.npy")
            participants = np.load(f"{base_path}_participants.npy")
            
            # Try to load typing times
            try:
                typing_times = np.load(f"{base_path}_typing_times.npy")
            except FileNotFoundError:
                typing_times = None
            
            # Load bigram pairs
            with open(f"{base_path}_bigram_pairs.json", 'r') as f:
                pairs = json.load(f)
                # Convert lists back to tuples
                bigram_pairs = [
                    ((p[0][0], p[0][1]), (p[1][0], p[1][1]))
                    for p in pairs
                ]
            
            return ProcessedData(
                bigram_pairs=bigram_pairs,
                feature_matrix=feature_matrix,
                target_vector=target_vector,
                participants=participants,
                typing_times=typing_times
            )
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise

def generate_train_test_splits(processed_data, config):
    """
    Generates and saves train/test participant splits.
    
    Args:
        processed_data (ProcessedData): The processed dataset where each row represents a participant.
        config (dict): Configuration dictionary containing split settings.
    
    Returns:
        None: The splits are saved to the path specified in the config.
    """
    splits_config = config.get('splits', {})
    generate_splits = splits_config.get('generate_splits', False)
    train_ratio = splits_config.get('train_ratio', 0.8)
    splits_file = splits_config.get('splits_file', "data/splits/train_test_indices.npz")
    
    if not generate_splits:
        print("Skipping split generation as per configuration.")
        return
    
    # Determine the number of participants
    participants = processed_data.participants
    n_participants = len(participants)
    
    # Shuffle participants and split indices
    indices = np.arange(n_participants)
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * n_participants)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(splits_file), exist_ok=True)
    
    # Save the splits to an .npz file
    np.savez(splits_file, train_indices=train_indices, test_indices=test_indices)
    print(f"Train/test splits saved to {splits_file}.")

def manage_data_splits(processed_data, config):
    """
    Splits the processed data into training and testing datasets.
    
    Args:
        processed_data (ProcessedData): The processed dataset.
        config (dict): Configuration dictionary.
    
    Returns:
        tuple: (train_data, test_data)
    """
    splits_file = config['splits']['splits_file']
    splits = np.load(splits_file)
    train_indices = splits['train_indices']
    test_indices = splits['test_indices']
    
    # Split participants and other attributes by indices
    train_participants = [processed_data.participants[i] for i in train_indices]
    test_participants = [processed_data.participants[i] for i in test_indices]
    
    train_bigram_pairs = [processed_data.bigram_pairs[i] for i in train_indices]
    test_bigram_pairs = [processed_data.bigram_pairs[i] for i in test_indices]
    
    train_target_vector = [processed_data.target_vector[i] for i in train_indices]
    test_target_vector = [processed_data.target_vector[i] for i in test_indices]
    
    if processed_data.typing_times is not None:
        train_typing_times = [processed_data.typing_times[i] for i in train_indices]
        test_typing_times = [processed_data.typing_times[i] for i in test_indices]
    else:
        train_typing_times = None
        test_typing_times = None
    
    train_feature_matrix = processed_data.feature_matrix.iloc[train_indices, :]
    test_feature_matrix = processed_data.feature_matrix.iloc[test_indices, :]

    # Create new ProcessedData instances for train and test datasets
    train_data = replace(
        processed_data,
        participants=train_participants,
        bigram_pairs=train_bigram_pairs,
        target_vector=train_target_vector,
        typing_times=train_typing_times,
        feature_matrix=train_feature_matrix,
    )
    
    test_data = replace(
        processed_data,
        participants=test_participants,
        bigram_pairs=test_bigram_pairs,
        target_vector=test_target_vector,
        typing_times=test_typing_times,
        feature_matrix=test_feature_matrix,
    )
    
    return train_data, test_data

def validate_features(config: Dict, feature_matrix: pd.DataFrame) -> None:
   """Validate that required features exist and are properly configured."""
   
   # Check control features exist
   for feature in config['features']['groups']['control']:
       if feature not in feature_matrix.columns:
           raise ValueError(f"Control feature {feature} missing from data")
           
   # Check typing_time specifically
   if 'typing_time' in config['features']['groups']['control']:
       if feature_matrix['typing_time'].isnull().any():
           raise ValueError("Missing timing data")
           
   # Check feature definitions match usage
   eval_features = set()
   for combo in config['feature_evaluation']['combinations']:
       eval_features.update(combo)
   
   model_features = set(config['features']['groups']['design'] +
                       config['features']['groups']['control'])
                       
   if not eval_features.issubset(model_features):
       raise ValueError("Evaluated features not subset of model features")
