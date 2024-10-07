import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.optimize import differential_evolution

#=====================================#
# Keyboard layout and finger mappings #
#=====================================#
qwerty = {
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

#====================#
# Feature Extraction #
#====================#
def same_finger(char1, char2, finger_map):
    # Check if the characters are typed with the same finger.
    return finger_map[char1] == finger_map[char2]

def lateral_stretch(char1, char2, column_map):
    # Check if the first finger is stretched laterally.
    if column_map[char1] in [5, 6] or column_map[char2] in [5, 6]:
        return True
    else:
        return False

def off_home(char1, char2, row_map):
    # Return True if either character is not on the home row.
    y1 = row_map[char1]
    y2 = row_map[char2]
    if y1 != 2 or y2 != 2:
        return True
    else:
        return False

def row_change(char1, char2, row_map):
    # Check if the characters are on different rows.
    return row_map[char1] != row_map[char2]

def nrows_change(char1, char2, row_map):
    # Measure how many rows apart are the characters.
    return abs(row_map[char1] - row_map[char2])

def ncolumns_change(char1, char2, column_map):
    # Measure how many columns apart are the characters.
    return abs(column_map[char1] - column_map[char2])

def hand_alternation(char1, char2, qwerty):
    # Check if characters are typed with different hands.
    return (char1 in qwerty['left'] and char2 in qwerty['rite']) or \
           (char1 in qwerty['rite'] and char2 in qwerty['left'])

def outward_roll(char1, char2, qwerty, finger_map):
    # Return True if outward roll. 
    #   - outward:  right-to-left for the left hand, left-to-right for the right hand
    #   - inward:   left-to-right for the left hand, right-to-left for the right hand
    if hand_alternation(char1, char2, qwerty) or lateral_stretch(char1, char2, column_map):
        return False
    else:
        f1 = finger_map[char1]
        f2 = finger_map[char2]
        if f2 > f1:
            return True
        else:
            return False

def key_distance(char1, char2, position_map):
    # Measure the distance between two characters based on keyboard positions.
    x1, y1 = position_map[char1]
    x2, y2 = position_map[char2]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_features(char1, char2, qwerty, finger_map, row_map, column_map, position_map):
    return {
        'same_finger': same_finger(char1, char2, finger_map),
        'lateral_stretch': lateral_stretch(char1, char2, column_map),
        'off_home': off_home(char1, char2, row_map),
        'row_change': row_change(char1, char2, row_map),
        'nrows_change': nrows_change(char1, char2, row_map),
        'ncolumns_change': ncolumns_change(char1, char2, column_map),
        'hand_alternation': hand_alternation(char1, char2, qwerty),
        'outward_roll': outward_roll(char1, char2, qwerty, finger_map),
        'key_distance': key_distance(char1, char2, position_map)
    }

#=================#
# Data processing #
#=================#
def prep_data(df, filter_inconsistent=True):
    """
    Prepare the consolidated data for analysis, splitting two-letter bigrams into individual characters.
    - df: The dataframe containing consolidated bigram data per user.
    - filter_inconsistent: If True, filter out rows where 'is_consistent' is False.
    """
    # Optionally filter out inconsistent users
    if filter_inconsistent:
        df = df[df['is_consistent'] == True]
    
    X = []  # Feature matrix
    y = []  # Target vector (preference scores)

    for _, row in df.iterrows():
        char1_bigram = row['bigram1']
        char2_bigram = row['bigram2']
        
        # Split the two-letter bigrams into individual characters
        char1_1, char1_2 = char1_bigram[0], char1_bigram[1]
        char2_1, char2_2 = char2_bigram[0], char2_bigram[1]

        # Skip if characters are repeated or if either character is not valid
        if char1_1 == char1_2 or char2_1 == char2_2 or char1_1 not in finger_map or char2_1 not in finger_map:
            continue
        
        # Extract features for valid bigram pairs
        try:
            features = extract_features(char1_1, char2_1, qwerty, finger_map, row_map, column_map, position_map)
            X.append(list(features.values()))  # Convert feature dictionary to list
            # Target: the consolidated score for this user and bigram pair
            y.append(row['score'])

        except KeyError as e:
            print(f"Error extracting features for characters: {char1_1}, {char2_1}: {e}")
            continue
    
    return np.array(X), np.array(y)

# Train a Ridge regression model
def train_ridge_regression(X, y, alpha=1.0):
    """
    Train a Ridge regression model on the given data.
    
    Parameters:
    - X: The feature matrix.
    - y: The target vector.
    - alpha: The regularization strength for Ridge regression (default is 1.0).
    
    Returns:
    - The trained Ridge regression model.
    """
    # Split the data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Ridge regression model
    ridge_model = Ridge(alpha=alpha)

    # Train the model on the training data
    ridge_model.fit(X_train, y_train)

    # Evaluate the model on the training set
    train_score = ridge_model.score(X_train, y_train)
    print(f"Training R^2 score: {train_score}")

    # Evaluate the model on the test set
    test_score = ridge_model.score(X_test, y_test)
    print(f"Test R^2 score: {test_score}")

# Objective function for layout optimization
def objective_function(layout_indices, initial_layout, bigram_data, model, qwerty, finger_map, row_map, column_map, position_map):
    layout_indices = np.round(layout_indices).astype(int)
    layout_str = ''.join([initial_layout[i] for i in layout_indices])

    total_score = 0
    valid_pairs = 0

    # Loop through all pairs of characters in the layout
    for i in range(len(layout_str)):
        for j in range(i + 1, len(layout_str)):
            char1 = layout_str[i]
            char2 = layout_str[j]
            features = extract_features(char1, char2, qwerty, finger_map, row_map, column_map, position_map)
            score = model.predict([list(features.values())])[0]

            total_score += score
            valid_pairs += 1

    # Return the negative score (since we minimize by default)
    return -total_score / valid_pairs if valid_pairs > 0 else 0

def validate_model(model, X, y):
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores)}")

    # Leave-One-Out Cross-Validation (LOO-CV)
    loo = LeaveOneOut()
    cvloo_scores = cross_val_score(model, X, y, cv=loo, scoring='r2')
    print(f"Leave-One-Out cross-validation scores: {cvloo_scores}")
    print(f"Mean LOO cross-validation score: {np.mean(cvloo_scores)}")

# Optimize the layout using differential evolution
def optimize_layout(initial_layout, bigram_data, model, qwerty, finger_map, row_map, column_map, position_map):
    n = len(initial_layout)
    bounds = [(0, n-1)] * n

    initial_score = -objective_function(np.arange(n), initial_layout, bigram_data, model, qwerty, finger_map, row_map, column_map, position_map)
    print(f"Initial layout score: {initial_score}")

    result = differential_evolution(
        lambda x: objective_function(x, initial_layout, bigram_data, model, qwerty, finger_map, row_map, column_map, position_map),
        bounds,
        maxiter=1000,
        popsize=20,
        mutation=(0.5, 1),
        recombination=0.7
    )

    optimized_indices = np.round(result.x).astype(int)
    final_layout = ''.join([initial_layout[i] for i in optimized_indices])
    final_score = -objective_function(optimized_indices, initial_layout, bigram_data, model, qwerty, finger_map, row_map, column_map, position_map)

    print(f"Final layout score: {final_score}")
    return final_layout, final_score - initial_score

# Main Execution
if __name__ == "__main__":
    # Load the CSV file into a pandas DataFrame
    csv_file_path = "/Users/arno.klein/Downloads/osf/output/output_99filtered-users/tables/scored_bigram_data.csv" 
    scored_bigram_data_df = pd.read_csv(csv_file_path)

    # Prepare the data with filtering for repeated characters and inconsistent users
    X_consolidated, y_consolidated = prep_data(scored_bigram_data_df, filter_inconsistent=True)

    # Check the shape of the feature matrix and target vector
    print(f"Feature matrix shape: {X_consolidated.shape}")
    print(f"Target vector shape: {y_consolidated.shape}")

    # Train the Ridge regression model using the prepared data
    model = train_ridge_regression(X_consolidated, y_consolidated)

    # Validate modal with cross-validation
    validate_model(model, X_consolidated, y_consolidated)

    # Initial layout (replace with your own layout if needed)
    initial_layout = 'qwertasdfgzxcvbyuiophjklmn,'

    # Optimize the layout
    optimized_layout, improvement = optimize_layout(initial_layout, scored_bigram_data_df, model, 
                                                    qwerty, finger_map, row_map, column_map, position_map)

    # Print results
    print(f"Initial layout: {initial_layout}")
    print(f"Optimized layout: {optimized_layout}")
    print(f"Score improvement: {improvement}")