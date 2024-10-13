import numpy as np
import pandas as pd
from itertools import permutations, product
import matplotlib.pyplot as plt
import ast
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor
import choix
from sklearn.base import BaseEstimator
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut, GroupKFold, cross_val_score, GridSearchCV
from scipy.optimize import differential_evolution, minimize
import networkx as nx
import pymc as pm
import arviz as az

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

#==========#
# Features #
#==========#
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
    features = {
        #'same_hand': same_hand(char1, char2, column_map),
        'same_finger': same_finger(char1, char2, column_map, finger_map),
        'adjacent_fingers': adjacent_fingers(char1, char2, column_map, finger_map),
        'finger1': finger1(char1, char2, finger_map),
        'finger2': finger2(char1, char2, finger_map),
        'finger3': finger3(char1, char2, finger_map),
        'finger4': finger4(char1, char2, finger_map),
        'finger1above': finger1_above(char1, char2, column_map, row_map, finger_map),
        'finger2below': finger2_below(char1, char2, column_map, row_map, finger_map),
        'finger3below': finger3_below(char1, char2, column_map, row_map, finger_map),
        'finger4above': finger4_above(char1, char2, column_map, row_map, finger_map),
        'finger_pairs': finger_pairs(char1, char2, column_map, finger_map),
        'rows_apart': rows_apart(char1, char2, column_map, row_map), # x7
        'columns_apart': columns_apart(char1, char2, column_map),
        'outward_roll': outward_roll(char1, char2, column_map, finger_map),  # x9
        'middle_columns': middle_columns(char1, char2, column_map)  # x10
    }

    #print(f"Extracted features for {char1}, {char2}: {features}")

    feature_names = list(features.keys())
    
    return features, feature_names

#=================================================================#
# Features for all bigrams and their differences in feature space #
#=================================================================#
def precompute_all_bigram_features(layout_chars, column_map, row_map, finger_map):
    """
    Precompute features for all possible bigrams based on the given layout characters.
    
    Parameters:
    - layout_chars: List of all possible characters in the keyboard layout.
    
    Returns:
    - all_bigrams: All possible bigrams, including repetition.
    - all_bigram_features: DataFrame mapping all bigrams to their feature vectors (with named columns).
    - feature_names: List of feature names.   
    """
    # Generate all possible bigrams (permutations of 2 unique characters)
    #bigrams = list(permutations(layout_chars, 2))  # Permutations give us all bigram pairs (without repetition)

    # Generate all possible bigrams, including repetition
    all_bigrams = list(product(layout_chars, repeat=2))  # Product allows repetition (e.g., ('a', 'a'))

    # Extract features for each bigram
    feature_vectors = []
    feature_names = None

    for char1, char2 in all_bigrams:
        # Extract features for the bigram
        features, feature_names = extract_features(char1, char2, column_map, row_map, finger_map)

        # Convert features to a list
        feature_vector = list(features.values())
        feature_vectors.append(feature_vector)
    
    # Convert to DataFrame with feature names
    all_bigram_features = pd.DataFrame(feature_vectors, columns=feature_names, index=all_bigrams)
    all_bigram_features.index = pd.MultiIndex.from_tuples(all_bigram_features.index)

    print(f"Extracted {len(all_bigram_features.columns)} features from each of {len(all_bigrams)} possible bigrams.")

    return all_bigrams, all_bigram_features, feature_names

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
    bigrams_list = list(bigram_features.index)

    # Loop over all pairs of bigrams
    for i, bigram1 in enumerate(bigrams_list):
        for j, bigram2 in enumerate(bigrams_list):
            if i <= j:  # Only compute differences for unique pairs (skip symmetric pairs)
                # Calculate the feature differences
                abs_feature_diff = np.abs(bigram_features.loc[bigram1].values - bigram_features.loc[bigram2].values)
                bigram_feature_differences[(bigram1, bigram2)] = abs_feature_diff
                bigram_feature_differences[(bigram2, bigram1)] = abs_feature_diff  # Symmetric pair

    print(f"Calculated all {len(bigram_feature_differences)} bigram-bigram feature differences.")
      
    return bigram_feature_differences

def plot_bigram_graph(bigram_pairs):
    """
    Plot a graph of all bigrams as nodes with edges connecting bigrams that are in pairs.
    
    Parameters:
    - bigram_pairs: List of bigram pairs (e.g., [(('a', 'r'), ('s', 't')), ...])
    """
    # Create a graph
    G = nx.Graph()

    # Create a mapping of tuple bigrams to string representations
    for bigram1, bigram2 in bigram_pairs:
        bigram1_str = ''.join(bigram1)  # Convert tuple ('a', 'r') to string "ar"
        bigram2_str = ''.join(bigram2)  # Convert tuple ('s', 't') to string "st"
        
        # Add edges between bigram string representations
        G.add_edge(bigram1_str, bigram2_str)

    # Get all connected components (subgraphs)
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    # Initialize figure
    plt.figure(figsize=(14, 14))

    # Layout positioning of all components
    pos = {}
    grid_size = int(np.ceil(np.sqrt(len(components))))  # Arrange components in a grid
    spacing = 5.0  # Adjust this value for more space between clusters

    # Iterate over each component and apply a layout
    for i, component in enumerate(components):
        # Apply spring layout to the current component
        component_pos = nx.spring_layout(component, k=1.0, seed=i)  # Use different seed for each component
        
        # Determine the grid position for this component
        x_offset = (i % grid_size) * spacing  # X-axis grid position
        y_offset = (i // grid_size) * spacing  # Y-axis grid position

        # Shift the component positions to avoid overlap
        for node in component_pos:
            component_pos[node][0] += x_offset
            component_pos[node][1] += y_offset

        # Update the global position dictionary
        pos.update(component_pos)
    
    # Draw the entire graph with the adjusted positions
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_weight='bold', 
            node_size=500, font_size=14, edge_color='gray', linewidths=1.5, width=2.0)

    # Display the graph
    plt.title("Bigram Connectivity Graph", fontsize=20)
    plt.show()

def check_multicollinearity(feature_matrix):
    """
    Check for multicollinearity.
    Multicollinearity occurs when two or more predictor variables are highly correlated, 
    which can inflate standard errors and lead to unreliable p-values. 
    You can check for multicollinearity using the Variance Inflation Factor (VIF), 
    a common metric that quantifies how much the variance of a regression coefficient 
    is inflated due to collinearity with other predictors.

    VIF ≈ 1: No correlation between a feature and the other features.
        •	1 < VIF < 5: Moderate correlation, but acceptable.
        •	VIF > 5: High correlation; consider removing or transforming the feature.
        •	VIF > 10: Serious multicollinearity issue.

    Parameters:
    - feature_matrix: DataFrame containing the features
    """
    print("\nVariance Inflation Factor to check for multicollinearity")
    print("    1 < VIF < 5: moderate correlation, but acceptable")
    # Add a constant column for intercept
    X = sm.add_constant(feature_matrix)

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Display the VIF for each feature
    print(vif_data)

#============================================================#
# Bayesian GLMM: priors and prior sensitivity, GLMM training #
#============================================================#
def train_bayesian_glmm(feature_matrix, target_vector, participants, 
                        selected_feature_names=None, typing_time_prior_func=None,
                        typing_times=None, inference_method="mcmc", 
                        num_samples=1000, chains=4):
    """
    Train a Bayesian GLMM with hierarchical modeling using custom and feature-based priors.
    """
    # Ensure that selected features are provided, or default to all features
    if selected_feature_names is None:
        selected_feature_names = feature_matrix.columns.tolist()

    num_participants = len(np.unique(participants))
    selected_features_matrix = feature_matrix[selected_feature_names]

    # Define the model context block to avoid the context stack error
    with pm.Model() as model:
        # Create feature-based priors dynamically
        feature_priors = {}
        for feature_name in selected_feature_names:
            feature_priors[feature_name] = pm.Normal(
                feature_name, 
                mu=np.mean(selected_features_matrix[feature_name]), 
                sigma=np.std(selected_features_matrix[feature_name])
            )

        # Define custom priors
        if typing_times is not None:
            if typing_time_prior_func is None:
                typing_time_prior = pm.Normal('typing_time', 
                                              mu=np.mean(typing_times), 
                                              sigma=np.std(typing_times))
            else:
                typing_time_prior = typing_time_prior_func()
        #adjacent_fingers_prior = pm.Normal('adjacent_fingers', mu=0, sigma=2)

        # Combine all priors (feature-based + custom)
        all_priors = list(feature_priors.values())
        if typing_times is not None:
            all_priors.append(typing_time_prior)
        #all_priors.append(adjacent_fingers_prior)

        # Random effects: Participant-specific intercepts
        participant_intercept = pm.Normal('participant_intercept', mu=0, sigma=1, 
                                          shape=num_participants)

        # Define the linear predictor (fixed + random effects)
        fixed_effects = pm.math.dot(selected_features_matrix, pm.math.stack([feature_priors[name] for name in selected_feature_names]))
        if typing_times is not None:
            fixed_effects += typing_time_prior
        #fixed_effects += adjacent_fingers_prior * selected_features_matrix['adjacent_fingers']

        mu = fixed_effects + participant_intercept[participants]

        # Likelihood: Observed target vector
        sigma = pm.HalfNormal('sigma', sigma=10)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=target_vector)

        # Choose the inference method
        if inference_method == "mcmc":
            trace = pm.sample(num_samples, chains=chains, return_inferencedata=True)
        elif inference_method == "variational":
            approx = pm.fit(method="advi", n=num_samples)
            trace = approx.sample(num_samples)
        else:
            raise ValueError("Inference method must be 'mcmc' or 'variational'.")

    return trace, model, all_priors

def perform_sensitivity_analysis(feature_matrix, target_vector, participants, typing_times, 
                                 prior_means=[0, 50, 100], prior_stds=[1, 10, 100]):
    """
    Perform sensitivity analysis by varying the prior on typing_time.
    
    Args:
    feature_matrix, target_vector, participants: As in the original model
    typing_times: The typing time data
    prior_means, prior_stds: Lists of means and standard deviations to try for the typing_time prior
    
    Returns:
    A dictionary of results for each prior configuration
    """
    results = {}
    
    for mean in prior_means:
        for std in prior_stds:
            print(f"Running model with typing_time prior: N({mean}, {std})")
            
            def custom_typing_time_prior():
                return pm.Normal('typing_time', mu=mean, sigma=std)
            
            trace, model, priors = train_bayesian_glmm(
                feature_matrix=feature_matrix,
                target_vector=target_vector,
                participants=participants,
                selected_feature_names=['same_finger', 'outward_roll', 'adjacent_fingers'],
                typing_times=typing_times,
                typing_time_prior_func=custom_typing_time_prior,
                inference_method="mcmc",
                num_samples=500,
                chains=2
            )
            
            summary = az.summary(trace)
            results[f"prior_N({mean},{std})"] = summary
            
            print(summary)
            print("\n" + "="*50 + "\n")
    
    return results

def calculate_bayesian_comfort_scores(trace, bigram_pairs, feature_matrix, params=None):
    """
    Generate latent comfort scores using the full posterior distributions from a Bayesian GLMM.
    
    Parameters:
    - trace: The trace object from the Bayesian GLMM (contains posterior samples).
    - bigram_pairs: List of bigram pairs used in the comparisons.
    - feature_matrix: The feature matrix used in the GLMM, indexed by bigrams.
    - params: List of parameter names to use from the trace. If None, all parameters in the trace will be used.
   
    Returns:
    - bigram_comfort_scores: Dictionary mapping each bigram to its latent comfort score.
    """
    # If params is not provided, use all parameters in the trace
    if params is None:
        params = [var for var in trace.posterior.variables() if var != 'participant_intercept' and var != 'sigma']
    
    # Extract posterior samples for the specified parameters
    posterior_samples = {param: trace.posterior[param].values.flatten() for param in params}
    
    # Create a set of all unique bigrams
    all_bigrams = set(bigram for pair in bigram_pairs for bigram in pair)
    
    # Calculate comfort scores for each bigram
    bigram_scores = {}
    for bigram in all_bigrams:
        # Extract feature values for this bigram
        bigram_features = feature_matrix.loc[bigram]
        
        # Calculate score using all posterior samples
        scores = np.zeros(len(posterior_samples[params[0]]))
        for param in params:
            if param in bigram_features:
                scores += posterior_samples[param] * bigram_features[param]
        
        # Use the mean score across all posterior samples
        bigram_scores[bigram] = np.mean(scores)
    
    # Normalize scores to 0-1 range
    min_score = min(bigram_scores.values())
    max_score = max(bigram_scores.values())
    bigram_comfort_scores = {bigram: (score - min_score) / (max_score - min_score) 
                             for bigram, score in bigram_scores.items()}
    
    return bigram_comfort_scores

#=========================#
# Nested cross-validation #
#=========================#
def scoring_function(y_true, y_pred):
    """
    Custom scoring function: R-squared.
    Ensures that only aligned data points are used for scoring.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure the two arrays are aligned in length
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]

    # Calculate R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def nested_cv_full_pipeline(glmm_model_func, bradley_terry_model_func, feature_matrix, 
                            target_vector, participants, bigram_pairs):
    """
    Perform nested cross-validation for both GLMM and Bradley-Terry models 
    for model validation and tuning.
    
    Parameters:
    - glmm_model: GLMM model to clean the data.
    - bradley_terry_model: Bradley-Terry model for pairwise comparisons.
    - feature_matrix: Feature matrix (X).
    - target_vector: Target vector (y).
    - participants: Grouping variable for random effects.
    
    Returns:
    - nested_scores: Scores from nested cross-validation for the full pipeline.
    """
    # Outer loop for evaluating the full pipeline
    outer_cv = GroupKFold(n_splits=5)
    
    cv_scores = []
    
    for train_idx, test_idx in outer_cv.split(feature_matrix, target_vector, groups=participants):
        X_train, X_test = feature_matrix.iloc[train_idx], feature_matrix.iloc[test_idx]
        y_train, y_test = target_vector[train_idx], target_vector[test_idx]
        groups_train = participants[train_idx]
        bigram_pairs_train = [bigram_pairs[i] for i in train_idx]
        bigram_pairs_test = [bigram_pairs[i] for i in test_idx]
        
        # Make sure bigram_pairs also align with the train/test split
        bigram_pairs_train = [bigram_pairs[i] for i in train_idx]  # Ensure the bigram pairs are correctly split
        
        # Step 1: Fit the GLMM on the training set
        glmm_result, cleaned_data_train = glmm_model_func(X_train, y_train, groups_train)
        
        # Step 2: Fit the Bradley-Terry model on the cleaned data and corresponding bigram pairs
        bigram_comfort_scores_train = bradley_terry_model_func(cleaned_data_train, bigram_pairs_train)
        
        # Step 3: Predict latent scores for the test set
        #latent_scores_test = [bigram_comfort_scores_train.get(bigram, 0) for bigram in bigram_pairs]
        latent_scores_test = [bigram_comfort_scores_train.get(bigram, 0) for bigram in bigram_pairs_test]

        # Step 4: Evaluate performance (e.g., using custom scoring function)
        print(f"y_test shape: {y_test.shape}, latent_scores_test shape: {len(latent_scores_test)}")
        score = scoring_function(y_test, latent_scores_test)
        cv_scores.append(score)
    
        print(f"Nested CV Scores: {cv_scores}")
        print(f"Mean CV Score: {np.mean(cv_scores)}")
        
        return cv_scores
    
#=====================#
# Layout optimization #
#=====================#
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

    plot_bigram_pair_graph = False
    run_sensitivity_analysis = True
    run_glmm = False
    run_nested_cross_validation = False
    run_optimize_layout = False

    #=====================================#
    # Load, prepare, and analyze features #
    #=====================================#
    layout_chars = list("qwertasdfgzxcvbyuiophjkl;nm,./")

    # Precompute all bigram features and differences between the features of every pair of bigrams
    all_bigrams, all_bigram_features, feature_names = precompute_all_bigram_features(layout_chars, column_map, row_map, finger_map)
    all_feature_differences = precompute_bigram_feature_differences(all_bigram_features)

    # Load the CSV file into a pandas DataFrame
    #csv_file_path = "/Users/arno.klein/Downloads/osf/output_all4studies_274of377participants_0improbable/tables/filtered_bigram_data.csv"
    csv_file_path = "/Users/arno.klein/Downloads/osf/output_all4studies_377participants/tables/filtered_bigram_data.csv"
    bigram_data = pd.read_csv(csv_file_path)  # print(bigram_data.columns)

    # Prepare bigram data (format, including strings to actual tuples, conversion to numeric codes)
    bigram_pairs = [ast.literal_eval(bigram_pair) for bigram_pair in bigram_data['bigram_pair']]
    bigram_pairs = [((bigram1[0], bigram1[1]), (bigram2[0], bigram2[1])) for bigram1, bigram2 in bigram_pairs]     # Split each bigram in the pair into its individual characters
    slider_values = bigram_data['abs_sliderValue']
    typing_times = bigram_data['chosen_bigram_time']
    participants = pd.Categorical(bigram_data['user_id']).codes  
    
    # Filter out bigram pairs where either bigram is not in the precomputed differences
    bigram_pairs = [bigram for bigram in bigram_pairs if bigram in all_feature_differences]
    target_vector = np.array([slider_values.iloc[idx] for idx, bigram in enumerate(bigram_pairs)])

    # Create a DataFrame for the feature matrix, using precomputed feature differences for the bigram pairs
    feature_matrix_data = [all_feature_differences[(bigram1, bigram2)] for (bigram1, bigram2) in bigram_pairs] 
    feature_matrix = pd.DataFrame(feature_matrix_data, columns=feature_names, index=bigram_pairs)

    # Add feature interactions
    feature_matrix['finger1above2'] = feature_matrix['finger1above'] * feature_matrix['finger2below']

    # Convert all columns in the feature matrix to numeric (coerce invalid values to NaN if any)
    feature_matrix = feature_matrix.apply(pd.to_numeric, errors='coerce')

    # Check multicollinearity: Run VIF on the feature matrix to identify highly correlated features
    check_multicollinearity(feature_matrix)

    # Plot a graph of all bigram pairs to make sure they are all connected for Bradley-Terry training
    if plot_bigram_pair_graph:
        plot_bigram_graph(bigram_pairs)

    #=======================================================================================#
    # Define and analyze priors, train a Bayesian GLMM, and score using Bayesian posteriors #
    #=======================================================================================#
    # Sensitivity analysis to determine how much each prior influences the results
    if run_sensitivity_analysis:
        sensitivity_results = perform_sensitivity_analysis(feature_matrix, target_vector, participants, 
                                                           typing_times, prior_means=[0, 50, 100], 
                                                           prior_stds=[1, 10, 100])
    if run_glmm:
        trace, model, priors = train_bayesian_glmm(feature_matrix, target_vector, participants, 
            selected_feature_names=['same_finger', 'outward_roll', 'adjacent_fingers'],  
            typing_times=typing_times,
            inference_method="mcmc", 
            num_samples=1000, 
            chains=4)

        # Plot the posterior summary
        print(az.summary(trace))  
        az.plot_trace(trace)

        # Generate bigram typing comfort scores using the Bayesian posteriors
        params = None #['same_finger', 'adjacent_fingers']
        bigram_comfort_scores = calculate_bayesian_comfort_scores(trace, bigram_pairs, feature_matrix, params)

        # Print the comfort score for a specific bigram
        print(bigram_comfort_scores[('a', 'e')])

    #=============================#
    # Run nested cross-validation #
    #=============================#
    if run_nested_cross_validation:
        # 1. Train the GLMM to analyze fixed and random effects to clean and stabilize data.
        # 2. Train Bradley-Terry model to estimate latent bigram typing comfort values.
        # 3. Nested cross-validation for model validation and tuning:
        participants = pd.Categorical(bigram_data['user_id']).codes  # Convert participant labels for random effects to numeric codes
        cv_scores = nested_scores = nested_cv_full_pipeline(
                        glmm_model_func=train_glmm,                     # The GLMM function to clean the data
                        bradley_terry_model_func=fit_bradley_terry_model,  # The Bradley-Terry model function
                        feature_matrix=feature_matrix,                   # Feature matrix (X)
                        target_vector=target_vector,                     # Target vector (y)
                        participants=participants,                       # Grouping variable (e.g., participants)
                        bigram_pairs=bigram_pairs                        # Bigram pairs for pairwise comparisons
                    )
    
    #==============================#
    # Optimization keyboard layout #
    #==============================#
    if run_optimize_layout:
        # Initial layout
        initial_layout = layout_chars

        # Optimize the layout
        #optimized_layout, improvement = optimize_layout(initial_layout, scored_bigram_data_df, model, bigram_features)

        # Print results
        #print(f"Initial layout: {initial_layout}")
        #print(f"Optimized layout: {optimized_layout}")
        #print(f"Score improvement: {improvement}")

