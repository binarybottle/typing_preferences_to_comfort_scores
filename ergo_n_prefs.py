import numpy as np
import pandas as pd
from itertools import permutations, product
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from scipy.optimize import differential_evolution
import ast
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import choix

#=====================================#
# Keyboard layout and finger mappings #
#=====================================#
qwerty_map = {
    'left': ['q', 'w', 'e', 'r', 't', 
             'a', 's', 'd', 'f', 'g', 
             'z', 'x', 'c', 'v', 'b'],
    'rite': ['y', 'u', 'i', 'o', 'p', 
             'h', 'j', 'k', 'l', ';', 
             'n', 'm', ',', '.', '/']
}

finger_map = {
    'q': 4, 'w': 3, 'e': 2, 'r': 1, 't': 1,
    'a': 4, 's': 3, 'd': 2, 'f': 1, 'g': 1,
    'z': 4, 'x': 3, 'c': 2, 'v': 1, 'b': 1,
    'y': 1, 'u': 1, 'i': 2, 'o': 3, 'p': 4,
    'h': 1, 'j': 1, 'k': 2, 'l': 3, ';': 4, 
    'n': 1, 'm': 1, ',': 2, '.': 3, '/': 4
}

row_map = {
    'q': 3, 'w': 3, 'e': 3, 'r': 3, 't': 3, 
    'a': 2, 's': 2, 'd': 2, 'f': 2, 'g': 2, 
    'z': 1, 'x': 1, 'c': 1, 'v': 1, 'b': 1,
    'y': 3, 'u': 3, 'i': 3, 'o': 3, 'p': 3, 
    'h': 2, 'j': 2, 'k': 2, 'l': 2, ';': 2, 
    'n': 1, 'm': 1, ',': 1, '.': 1, '/': 1
}

column_map = {
    'q': 1, 'w': 2, 'e': 3, 'r': 4, 't': 5, 
    'a': 1, 's': 2, 'd': 3, 'f': 4, 'g': 5, 
    'z': 1, 'x': 2, 'c': 3, 'v': 4, 'b': 5,
    'y': 6, 'u': 7, 'i': 8, 'o': 9, 'p': 10, 
    'h': 6, 'j': 7, 'k': 8, 'l': 9, ';': 10, 
    'n': 6, 'm': 7, ',': 8, '.': 9, '/': 10
}

position_map = {
    'q': (3, 1), 'w': (3, 2), 'e': (3, 3), 'r': (3, 4), 't': (3, 5),
    'a': (2, 1), 's': (2, 2), 'd': (2, 3), 'f': (2, 4), 'g': (2, 5),
    'z': (1, 1), 'x': (1, 2), 'c': (1, 3), 'v': (1, 4), 'b': (1, 5),
    'y': (3, 6), 'u': (3, 7), 'i': (3, 8), 'o': (3, 9), 'p': (3, 10),
    'h': (2, 6), 'j': (2, 7), 'k': (2, 8), 'l': (2, 9), ';': (2, 10),
    'n': (1, 6), 'm': (1, 7), ',': (1, 8), '.': (1, 9), '/': (1, 10)
}

#==================#
# Extract features #
#==================#
def same_hand(char1, char2, column_map):
    """
    Check if the same hand is used to type both keys.
      1: same hand
      0: both hands
    """
    if (column_map[char1] < 6 and column_map[char2] < 6) or \
       (column_map[char1] > 5 and column_map[char2] > 5):
        return 1
    else:
        return 0

def same_finger(char1, char2, column_map, finger_map):
    """
    Check if the same finger on the same hand types both keys.
      1: same finger
      0: different fingers or different hands
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger_map[char1] == finger_map[char2]:
            return 1
        else:
            return 0
    else:
        return 0

def adjacent_fingers(char1, char2, column_map, finger_map):
    """
    Check if adjacent fingers on the same hand type the two keys.
      1: adjacent fingers
      0: non-adjacent fingers
    """
    if same_hand(char1, char2, column_map) == 1:
        finger_difference = abs(finger_map[char2] - finger_map[char1])
        if finger_difference == 1:
            return 1
        else:
            return 0
    else:
        return 0

def finger1(char1, char2, finger_map):
    """
    Check whether finger 1 is used to type either key.
      1: Yes
      0: No
    """
    if finger_map[char1] == 1 or finger_map[char2] == 1:
        return 1
    else:
        return 0

def finger2(char1, char2, finger_map):
    """
    Check whether finger 2 is used to type either key.
      1: Yes
      0: No
    """
    if finger_map[char1] == 2 or finger_map[char2] == 2:
        return 1
    else:
        return 0

def finger3(char1, char2, finger_map):
    """
    Check whether finger 3 is used to type either key.
      1: Yes
      0: No
    """
    if finger_map[char1] == 3 or finger_map[char2] == 3:
        return 1
    else:
        return 0

def finger4(char1, char2, finger_map):
    """
    Check whether finger 4 is used to type either key.
      1: Yes
      0: No
    """
    if finger_map[char1] == 4 or finger_map[char2] == 4:
        return 1
    else:
        return 0

def finger1_above(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger1 types a key on a row above the other key typed by another finger on the same hand.
      1: finger1 above
      0: finger1 not above
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger1(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
            if (finger_map[char1] == 1 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 1 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger2_below(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger2 types a key on a row below the other key typed by another finger on the same hand.
      1: finger2 below
      0: finger2 not below
    """
    if same_hand(char1, char2, column_map) == 1:
        if (finger_map[char1] == 2 or finger_map[char2] == 2) and (finger_map[char1] != finger_map[char2]):
            if (finger_map[char1] == 2 and row_map[char1] < row_map[char2]) or \
               (finger_map[char2] == 2 and row_map[char2] < row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger3_below(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger3 types a key on a row below the other key typed by another finger on the same hand.
      1: finger3 below
      0: finger3 not below
    """
    if same_hand(char1, char2, column_map) == 1:
        if (finger_map[char1] == 3 or finger_map[char2] == 3) and (finger_map[char1] != finger_map[char2]):
            if (finger_map[char1] == 3 and row_map[char1] < row_map[char2]) or \
               (finger_map[char2] == 3 and row_map[char2] < row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger4_above(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger4 types a key on a row above the other key typed by another finger on the same hand.
      1: finger4 above
      0: finger4 not above
    """
    if same_hand(char1, char2, column_map) == 1:
        if (finger_map[char1] == 4 or finger_map[char2] == 4) and (finger_map[char1] != finger_map[char2]):
            if (finger_map[char1] == 4 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 4 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger_pairs(char1, char2, column_map, finger_map):
    """
    Check which finger pairs on the same hand are used to type the two keys.
      12: finger4, finger3
      11: finger3, finger4
      10: finger4, finger2
       9: finger2, finger4
       8: finger3, finger2
       7: finger2, finger3
       6: finger4, finger1
       5: finger1, finger4
       4: finger3, finger1
       3: finger1, finger3
       2: finger2, finger1
       1: finger1, finger2      
       0: repeat keys or different hands
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger_map[char1] == 4 and finger_map[char2] == 3:
            return 12
        elif finger_map[char1] == 3 and finger_map[char2] == 4:
            return 11
        elif finger_map[char1] == 4 and finger_map[char2] == 2:
            return 10
        elif finger_map[char1] == 2 and finger_map[char2] == 4:
            return 9
        elif finger_map[char1] == 3 and finger_map[char2] == 2:
            return 8
        elif finger_map[char1] == 2 and finger_map[char2] == 3:
            return 7
        elif finger_map[char1] == 4 and finger_map[char2] == 1:
            return 6
        elif finger_map[char1] == 1 and finger_map[char2] == 4:
            return 5
        elif finger_map[char1] == 3 and finger_map[char2] == 1:
            return 4
        elif finger_map[char1] == 1 and finger_map[char2] == 3:
            return 3
        elif finger_map[char1] == 2 and finger_map[char2] == 1:
            return 2
        elif finger_map[char1] == 1 and finger_map[char2] == 2:
            return 1
        else: 
            return 0
    else:
        return 0

def rows_apart(char1, char2, column_map, row_map):
    """
    Measure how many rows apart the two characters are (typed by the same hand).
      0: same row
      1: 1 row apart
      2: 2 rows apart
    """
    if same_hand(char1, char2, column_map) == 1:
        return abs(row_map[char2] - row_map[char1])
    else:
        return 0

def columns_apart(char1, char2, column_map):
    """
    Measure how many columns apart the two characters are (typed by the same hand).
      0: same column
      1: 1 column apart
      2: 2 columns apart
      3: 3 columns apart
      4: 4 columns apart
    """
    if same_hand(char1, char2, column_map) == 1: # and middle_columns(char1, char2, column_map) == 0:
        return abs(column_map[char2] - column_map[char1])
    else:
        return 0

def outward_roll(char1, char2, column_map, finger_map):
    """
    Check if the keys were typed in an outward rolling direction with the same hand.
    outward:  right-to-left for the left hand, left-to-right for the right hand
    inward:   left-to-right for the left hand, right-to-left for the right hand
      1: outward
      0: not outward
    """
    if same_hand(char1, char2, column_map) == 1:
        if same_finger(char1, char2, column_map, finger_map) == 0 and \
           middle_columns(char1, char2, column_map) == 0:
            if finger_map[char1] < finger_map[char2]:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
def middle_columns(char1, char2, column_map):
    """
    Check if finger1 types a key in a middle column of the keyboard.
      1: Yes
      0: No
    """
    if column_map[char1] in [5, 6] or column_map[char2] in [5, 6]:
        return 1
    else:
        return 0

def extract_features(char1, char2, column_map, row_map, finger_map):
    return {
        #'same_hand': same_hand(char1, char2, column_map),
        'same_finger': same_finger(char1, char2, column_map, finger_map),
        'adjacent_fingers': adjacent_fingers(char1, char2, column_map, finger_map),
        'finger1': finger1(char1, char2, finger_map),
        'finger2': finger2(char1, char2, finger_map),
        'finger3': finger3(char1, char2, finger_map),
        'finger4': finger4(char1, char2, finger_map),
        'finger1_above': finger1_above(char1, char2, column_map, row_map, finger_map),
        'finger2_below': finger2_below(char1, char2, column_map, row_map, finger_map),
        'finger3_below': finger3_below(char1, char2, column_map, row_map, finger_map),
        'finger4_above': finger4_above(char1, char2, column_map, row_map, finger_map),
        #'finger_pairs': finger_pairs(char1, char2, column_map, finger_map),
        'rows_apart': rows_apart(char1, char2, column_map, row_map), # x7
        'columns_apart': columns_apart(char1, char2, column_map),
        'outward_roll': outward_roll(char1, char2, column_map, finger_map),  # x9
        'middle_columns': middle_columns(char1, char2, column_map)  # x10
    }

#============================================================================#
# Precompute features for all bigrams and their differences in feature space #
#============================================================================#
def precompute_all_bigram_features(layout_chars, column_map, row_map, finger_map):
    """
    Precompute features for all possible bigrams based on the given layout characters.
    
    Parameters:
    - layout_chars: List of all possible characters in the keyboard layout.
    
    Returns:
    - all_bigrams: All possible bigrams, including repetition.
    - all_bigram_features: Dictionary mapping all bigrams to their feature vectors.
    """
    all_bigram_features = {}

    # Generate all possible bigrams (permutations of 2 unique characters)
    #bigrams = list(permutations(layout_chars, 2))  # Permutations give us all bigram pairs (without repetition)

    # Generate all possible bigrams, including repetition
    all_bigrams = list(product(layout_chars, repeat=2))  # Product allows repetition (e.g., ('a', 'a'))

    # Extract features for each bigram
    for char1, char2 in all_bigrams:
        # Extract features for the bigram
        features = extract_features(char1, char2, column_map, row_map, finger_map)
        
        # Convert features to a numpy array
        feature_vector = np.array(list(features.values()))
        
        all_bigram_features[(char1, char2)] = feature_vector
    
    print(f"Extracted {len(feature_vector)} features from each of {len(all_bigrams)} possible bigrams constructed from {len(layout_chars)} characters.")

    return all_bigrams, all_bigram_features

def precompute_bigram_feature_differences(bigram_features):
    """
    Precompute and store feature differences between bigram pairs.
    
    Parameters:
    - bigram_features: Dictionary of precomputed features for each bigram.
    
    Returns:
    - bigram_feature_differences: A dictionary where each key is a tuple of bigrams (bigram1, bigram2),
                                  and the value is the precomputed feature differences.
    """
    bigram_feature_differences = {}
    bigrams_list = list(bigram_features.keys())

    # Loop over all pairs of bigrams
    for i, bigram1 in enumerate(bigrams_list):
        for j, bigram2 in enumerate(bigrams_list):
            if i <= j:  # Only compute differences for unique pairs (skip symmetric pairs)
                # Calculate the feature differences
                abs_feature_diff = np.abs(np.array(bigram_features[bigram1]) - np.array(bigram_features[bigram2]))
                bigram_feature_differences[(bigram1, bigram2)] = abs_feature_diff
                bigram_feature_differences[(bigram2, bigram1)] = abs_feature_diff  # Symmetric pair

    print(f"Calculated all {len(bigram_feature_differences)} bigram-bigram feature differences.")
      
    return bigram_feature_differences

def prepare_feature_matrix_target_vector(bigram_data, bigram_feature_differences):
    """
    Prepare the feature matrix by looking up precomputed feature differences between bigram pairs.
    
    Parameters:
    - bigram_data: DataFrame containing bigram pairs and their preference scores.
    - bigram_feature_differences: Dictionary of precomputed feature differences for each bigram pair.
    
    Returns:
    - feature_matrix: Feature matrix (precomputed feature differences between bigram pairs).
    - target_vector: Target vector of preference scores.
    """
    # Convert bigram_pair strings to actual tuples using ast.literal_eval
    bigram_pairs = [ast.literal_eval(bigram) for bigram in bigram_data['bigram_pair']]

    # Split each bigram in the pair into its individual characters
    bigram_pairs = [((bigram1[0], bigram1[1]), (bigram2[0], bigram2[1])) for bigram1, bigram2 in bigram_pairs]

    # Filter out bigram pairs where either bigram is not in the precomputed differences
    filtered_bigram_pairs = [bigram for bigram in bigram_pairs if bigram in bigram_feature_differences]

    # Use the precomputed feature differences for the filtered bigram pairs
    feature_matrix = np.array([bigram_feature_differences[(bigram1, bigram2)] for (bigram1, bigram2) in filtered_bigram_pairs])

    # Filter the target vector accordingly (only include scores for valid bigram pairs)
    filtered_target_vector = [
        bigram_data['abs_sliderValue'].iloc[idx] for idx, bigram in enumerate(bigram_pairs)
        if bigram in bigram_feature_differences
    ]

    return feature_matrix, np.array(filtered_target_vector), filtered_bigram_pairs

#==========================#
# Train and validate model #
#==========================#
def train_glmm(feature_matrix, target_vector, participants, study_groups):
    """
    Train a GLMM model to handle continuous preference scores per participant per bigram pair.
    
    Parameters:
    - feature_matrix: The feature matrix (precomputed bigram pair feature differences).
    - target_vector: The target vector (continuous preference scores for each trial).
    - participants: Participant labels for random effects.
    - study_groups: Labels for random effects per study group.
    
    Returns:
    - cleaned_data: Cleaned preference data (fitted values from the GLMM).
    """
    # Add constant (intercept) term to the feature matrix
    feature_matrix = sm.add_constant(feature_matrix)

    # Fit a mixed linear model with random effects for participants and study groups
    model = MixedLM(target_vector, feature_matrix, groups=participants) #, exog_re=study_groups)
    glmm_result = model.fit()

    # Output the model summary for validation
    print(glmm_result.summary())

    # Extract the cleaned preferences (fitted values)
    cleaned_data = glmm_result.fittedvalues

    return cleaned_data

def fit_bradley_terry_model(cleaned_data, bigram_pairs):
    """
    Fit a Bradley-Terry model using choix to assign latent comfort scores.
    
    Parameters:
    - cleaned_data: Cleaned preference data from GLMM (continuous scores indicating strength of preference).
    - bigram_pairs: List of bigram pairs used in the comparisons.
    
    Returns:
    - A dictionary mapping each bigram to its latent comfort score.
    """
    # Assign indices to unique bigrams for choix compatibility
    bigram_index = {}
    current_index = 0
    for bigram1, bigram2 in bigram_pairs:
        if bigram1 not in bigram_index:
            bigram_index[bigram1] = current_index
            current_index += 1
        if bigram2 not in bigram_index:
            bigram_index[bigram2] = current_index
            current_index += 1

    # Create data for choix's pairwise fitting
    comparisons = []
    for idx, (bigram1, bigram2) in enumerate(bigram_pairs):
        if cleaned_data[idx] > 0:
            comparisons.append((bigram_index[bigram1], bigram_index[bigram2]))  # bigram1 preferred over bigram2
        else:
            comparisons.append((bigram_index[bigram2], bigram_index[bigram1]))  # bigram2 preferred over bigram1

    # Fit Bradley-Terry model using choix
    num_bigrams = len(bigram_index)
    scores = choix.ilsr_pairwise(num_bigrams, comparisons)

    # Map latent comfort scores back to the bigrams
    bigram_comfort_scores = {bigram: scores[index] for bigram, index in bigram_index.items()}

    return bigram_comfort_scores

def validate_model(model, feature_matrix, target_vector, use_loo=False):
    """
    Perform cross-validation on the model.
    
    Parameters:
    - model: The trained Ridge regression model.
    - feature_matrix: Feature matrix.
    - target_vector: Target vector.
    - use_loo: If True, use Leave-One-Out Cross-Validation; otherwise, use 5-fold cross-validation.
    
    Returns:
    - cv_scores: Cross-validation scores.
    """
    if use_loo:
        # Perform Leave-One-Out Cross-Validation
        loo = LeaveOneOut()
        cv_scores = cross_val_score(model, feature_matrix, target_vector, cv=loo, scoring='r2')
    else:
        # Perform 5-fold Cross-Validation
        cv_scores = cross_val_score(model, feature_matrix, target_vector, cv=5, scoring='r2')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores)}")

    return cv_scores

#=================#
# Optimize layout #
#=================#
def is_unique(layout):
    """ Check if all characters in the layout are unique. """
    return len(layout) == len(set(layout))

def evaluate_layout(layout, bigram_data, model, bigram_features):
    """
    Evaluate the layout by calculating its score using precomputed bigram features.
    
    Parameters:
    - layout: A string representing the keyboard layout.
    - bigram_data: Data containing bigram preferences.
    - model: Trained Ridge regression model.
    - bigram_features: Dictionary of precomputed bigram features.
    
    Returns:
    - score: The calculated score for the layout.
    """
    total_score = 0
    layout_map = {char: idx for idx, char in enumerate(layout)}

    # Iterate through each bigram in the data and calculate its score
    for _, row in bigram_data.iterrows():
        char1, char2 = row['bigram1'][0], row['bigram1'][1]  # Split the bigram into individual characters
        
        if char1 in layout_map and char2 in layout_map:
            bigram = (char1, char2)

            if bigram in bigram_features:
                # Use precomputed features
                feature_vector = bigram_features[bigram].reshape(1, -1)

                # Predict the score for this bigram using the model
                predicted_score = model.predict(feature_vector)

                # Accumulate the score (e.g., sum predicted preferences)
                total_score += predicted_score[0]
    
    return total_score

# Optimize the layout using differential evolution
def optimize_layout(initial_layout, bigram_data, model, bigram_features):
    """
    Optimize the keyboard layout using differential evolution and precomputed bigram features.
    
    Parameters:
    - initial_layout: The starting layout.
    - bigram_data: Bigram preference data.
    - model: Trained Ridge regression model.
    - bigram_features: Dictionary of precomputed bigram features.
    
    Returns:
    - optimized_layout: The optimized keyboard layout.
    - improvement: The score improvement.
    """
    layout_chars = list(initial_layout)

    def score_layout(layout):
        layout_str = ''.join(layout.astype(str))  # Convert the array back to string format
        
        if not is_unique(layout_str):  # Check if layout has repeated characters
            return np.inf  # Penalize layouts with repeated characters
        
        # Use the updated evaluate_layout function with precomputed features
        layout_score = evaluate_layout(layout_str, bigram_data, model, bigram_features)
        return -layout_score  # Minimize the negative score for differential evolution

    bounds = [(0, len(layout_chars) - 1)] * len(layout_chars)
    
    result = differential_evolution(score_layout, bounds, maxiter=1000, popsize=15, tol=1e-6)
    
    optimized_layout = ''.join([layout_chars[int(round(i))] for i in result.x])  # Round and convert to indices
    improvement = -result.fun
    
    return optimized_layout, improvement

#==============#
# Run the code #
#==============#
if __name__ == "__main__":

    layout_chars = list("qwertasdfgzxcvbyuiophjkl;nm,./")

    # Precompute all bigram features and differences between the features of every pair of bigrams
    all_bigrams, all_bigram_features = precompute_all_bigram_features(layout_chars, column_map, row_map, finger_map)
    all_bigram_feature_differences = precompute_bigram_feature_differences(all_bigram_features)

    # Load the CSV file into a pandas DataFrame
    csv_file_path = "/Users/arno.klein/Downloads/osf/output/tables/processed_bigram_data.csv"
    bigram_data = pd.read_csv(csv_file_path)  # print(bigram_data.columns)

    # Prepare the feature matrix and target vector
    feature_matrix, target_vector, bigram_pairs = prepare_feature_matrix_target_vector(bigram_data, 
                                                                                       all_bigram_feature_differences)
    feature_matrix = np.column_stack([feature_matrix, feature_matrix[:, 11] * feature_matrix[:, 12]])
    """ 'same_finger': same_finger(char1, char2, column_map, finger_map),
        'adjacent_fingers': adjacent_fingers(char1, char2, column_map, finger_map),
        'finger1': finger1(char1, char2, finger_map),
        'finger2': finger2(char1, char2, finger_map),
        'finger3': finger3(char1, char2, finger_map),
        'finger4': finger4(char1, char2, finger_map),
        'finger1_above': finger1_above(char1, char2, column_map, row_map, finger_map),
        'finger2_below': finger2_below(char1, char2, column_map, row_map, finger_map),
        'finger3_below': finger3_below(char1, char2, column_map, row_map, finger_map),
        'finger4_above': finger4_above(char1, char2, column_map, row_map, finger_map),
        #'finger_pairs': finger_pairs(char1, char2, column_map, finger_map),
        'rows_apart': rows_apart(char1, char2, column_map, row_map), # x7
        'columns_apart': columns_apart(char1, char2, column_map),
        'outward_roll': outward_roll(char1, char2, column_map, finger_map),  # x9
        'middle_columns': middle_columns(char1, char2, column_map)  # x10
    """
    feature_matrix['bigram_finger1_above_2'] = feature_matrix['finger1_above'] * feature_matrix['finger2_below']


    # Train the GLMM
    group_ids = bigram_data['group_id']  # Group labels for random effects
    participant_ids = bigram_data['user_id']  # Participant labels for random effects
    study_groups = pd.Categorical(group_ids).codes  # Convert labels to numeric codes
    participants = pd.Categorical(participant_ids).codes  # Convert labels to numeric codes
    cleaned_data = train_glmm(feature_matrix, target_vector, participants, study_groups)

    # Assuming cleaned_data and bigram_pairs are already defined from GLMM
    #bigram_comfort_scores = fit_bradley_terry_model(cleaned_data, bigram_pairs)

    # Example: Print comfort score for a specific bigram
    #print(bigram_comfort_scores[('a', 'e')])

    #validate_model(model, feature_matrix, target_vector, use_loo=False)

    """
    # Initial layout
    initial_layout = layout_chars

    # Optimize the layout
    optimized_layout, improvement = optimize_layout(initial_layout, scored_bigram_data_df, model, bigram_features)

    # Print results
    print(f"Initial layout: {initial_layout}")
    print(f"Optimized layout: {optimized_layout}")
    print(f"Score improvement: {improvement}")
    """
    

