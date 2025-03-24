# validate_on_model.py
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from bigram_typing_preferences_to_comfort_scores.model import PreferenceModel
from bigram_typing_preferences_to_comfort_scores.data import PreferenceDataset
from bigram_typing_preferences_to_comfort_scores.features.feature_extraction import FeatureExtractor, FeatureConfig
from bigram_typing_preferences_to_comfort_scores.utils.config import Config
from bigram_typing_preferences_to_comfort_scores.features.keymaps import (
    column_map, row_map, finger_map,
    engram_position_values, row_position_values
)
from bigram_typing_preferences_to_comfort_scores.features.features import angles
from bigram_typing_preferences_to_comfort_scores.features.bigram_frequencies import bigrams, bigram_frequencies_array

input_file = "/Users/arno/Downloads/Prolific_studies/Study1_286participants_2x11pairs_2024-09-26/processed_data/tables/processed_bigram_data.csv" 
#input_file = "/Users/arno/Downloads/Prolific_studies/Study2A_30participants_2x35pairs_2024-10-09/processed_data/tables/processed_bigram_data.csv" 
#input_file = "/Users/arno/Downloads/Prolific_studies/Study2B_30participants_2x35pairs_2024-10-09/processed_data/tables/processed_bigram_data.csv" 
#input_file = "/Users/arno/Downloads/Prolific_studies/Study3_30participants_2x35pairs_2024-10-11/processed_data/tables/processed_bigram_data.csv" 
#input_file = "/Users/arno/Downloads/Prolific_studies/Study4_29participants_2x35pairs_2024-10-13/processed_data/tables/processed_bigram_data.csv" 
#input_file = "/Users/arno/Downloads/Prolific_studies/Study5_46participants_2x50pairs_2025-02-06/processed_data/tables/processed_bigram_data.csv" 
#input_file = "/Users/arno/Downloads/Prolific_studies/Study7_25participants_2x50pairs_2025-03-17/processed_data/tables/processed_bigram_data.csv" 

output_file = "participant_metrics_Study1_286participants_2x11pairs_2024-09-26.csv" 
#output_file = "participant_metrics_Study2A_30participants_2x35pairs_2024-10-09.csv" 
#output_file = "participant_metrics_Study2B_30participants_2x35pairs_2024-10-09.csv" 
#output_file = "participant_metrics_Study3_30participants_2x35pairs_2024-10-11.csv" 
#output_file = "participant_metrics_Study4_29participants_2x35pairs_2024-10-13.csv" 
#output_file = "participant_metrics_Study5_46participants_2x50pairs_2025-02-06.csv" 
#output_file = "participant_metrics_Study7_25participants_2x50pairs_2025-03-17.csv" 

# Load configuration
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Convert to Config object
config = Config(**config_dict)  # Convert dict to Config object

# Load trained model
model = PreferenceModel.load(Path("output/data/bigram_score_prediction_model.pkl"))

# Initialize feature extraction (similar to main.py)
feature_config = FeatureConfig(
    column_map=column_map,
    row_map=row_map,
    finger_map=finger_map,
    engram_position_values=engram_position_values,
    row_position_values=row_position_values,
    angles=angles,
    bigrams=bigrams,
    bigram_frequencies_array=bigram_frequencies_array
)
feature_extractor = FeatureExtractor(feature_config)

# Precompute features
all_bigrams, all_bigram_features = feature_extractor.precompute_all_features(
    config.data.layout['chars']  # Use config object here, not dict
)
feature_names = list(next(iter(all_bigram_features.values())).keys())

# Load the new dataset to validate
new_dataset = PreferenceDataset(
    Path(input_file),
    feature_extractor=feature_extractor,
    config=config,  # Pass the Config object, not the dict
    precomputed_features={
        'all_bigrams': all_bigrams,
        'all_bigram_features': all_bigram_features,
        'feature_names': feature_names
    }
)

# Evaluate model on noisy dataset
print("Evaluating model on noisy prolific dataset...")
metrics = model.evaluate(new_dataset)

print("\nTest metrics on noisy data:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")

# Additional analysis - calculate per-participant accuracy
participant_metrics = {}
for participant_id in new_dataset.participants:
    # Create subset with just this participant
    participant_prefs = [p for p in new_dataset.preferences if p.participant_id == participant_id]
    
    predictions = []
    true_labels = []
    for pref in participant_prefs:
        try:
            pred = model.predict_preference(pref.bigram1, pref.bigram2)
            predictions.append(pred.probability > 0.5)
            true_labels.append(pref.preferred)
        except Exception as e:
            print(f"Error predicting for {pref.bigram1}-{pref.bigram2}: {e}")
            continue
    
    if predictions:
        acc = np.mean(np.array(predictions) == np.array(true_labels))
        participant_metrics[participant_id] = {
            'accuracy': acc,
            'n_prefs': len(predictions)
        }

# Save participant metrics
pd.DataFrame.from_dict(participant_metrics, orient='index').to_csv(output_file)

# Analyze distribution of participant accuracies
accuracies = [m['accuracy'] for m in participant_metrics.values()]
print(f"\nParticipant accuracy stats:")
print(f"Mean: {np.mean(accuracies):.3f}")
print(f"Median: {np.median(accuracies):.3f}")
print(f"Min: {np.min(accuracies):.3f}")
print(f"Max: {np.max(accuracies):.3f}")