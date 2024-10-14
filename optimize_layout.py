import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

#=========================================#
# Functions to optimizge keyboard layouts #
#=========================================#
def precompute_valid_bigrams(keyboard_layout, bigram_frequencies, comfort_scores):
    """
    Optimizes the computation of valid bigrams by caching character positions.
    """
    character_set = set(keyboard_layout)
    valid_bigrams = {}

    # Normalize comfort scores and bigram frequencies
    comfort_scores['normalized_comfort'] = (
        (comfort_scores['comfort_score'] - comfort_scores['comfort_score'].min()) /
        (comfort_scores['comfort_score'].max() - comfort_scores['comfort_score'].min())
    )
    freq_values = np.array(list(bigram_frequencies.values()))
    min_freq, max_freq = freq_values.min(), freq_values.max()

    # Cache the index positions of characters
    char_positions = {char: idx for idx, char in enumerate(keyboard_layout)}

    # Iterate through all valid bigrams efficiently using cached positions
    for char1, char2 in itertools.permutations(character_set, 2):
        bigram = char1 + char2
        if bigram in bigram_frequencies:
            normalized_freq = (bigram_frequencies[bigram] - min_freq) / (max_freq - min_freq)

            # Construct the keyboard bigram from cached positions
            keyboard_bigram = keyboard_layout[char_positions[char1]] + keyboard_layout[char_positions[char2]]

            # Assign a normalized comfort score or the maximum of 1.0 for all valid pairs
            comfort = comfort_scores.get(keyboard_bigram, {}).get('normalized_comfort', 1.0)
            valid_bigrams[bigram] = normalized_freq * comfort

    return valid_bigrams

def optimize_subset(keys_to_replace, chars_to_arrange, valid_bigrams, keyboard_layout):
    """
    Optimizes the arrangement of a subset of keys in the layout with a progress bar.
    """
    best_layout = None
    best_score = -math.inf

    # Calculate the total number of permutations
    total_permutations = math.factorial(len(chars_to_arrange))

    # Iterate over all permutations with tqdm progress bar
    for permutation in tqdm(itertools.permutations(chars_to_arrange), total=total_permutations, desc="Optimizing layout"):
        # Create a new layout with the current permutation
        temp_layout = list(keyboard_layout)
        for key, char in zip(keys_to_replace, permutation):
            pos = temp_layout.index(key)
            temp_layout[pos] = char

        # Calculate the layout score (inlined for performance)
        score = 0
        for char1, char2 in itertools.permutations(temp_layout, 2):
            bigram = char1 + char2
            if bigram in valid_bigrams:
                score += valid_bigrams[bigram]

        # Track the best layout and score
        if score > best_score:
            best_score = score
            best_layout = "".join(temp_layout)

    return best_layout, best_score

def display_keyboard(layout, keys_to_replace, chars_to_arrange):
    """
    Displays the keyboard layout by splitting it into 3 rows of 10 keys.
    Uses '-' placeholders for all keys except the new characters.
    """
    # Initialize the layout with '-' placeholders
    filled_layout = ["-"] * 30

    # Place new characters at the correct positions
    key_positions = [keyboard_layout.index(key) for key in keys_to_replace]
    for pos, new_char in zip(key_positions, chars_to_arrange):
        filled_layout[pos] = new_char

    # Split the layout into 3 rows of 10 keys each
    rows = [
        filled_layout[:10],   # First row
        filled_layout[10:20], # Second row
        filled_layout[20:]    # Third row
    ]

    # Build the display with proper left-right spacing
    result = []
    for row in rows:
        left = " ".join(row[:5])   # First 5 keys (left side)
        right = " ".join(row[5:])  # Last 5 keys (right side)
        result.append(f"{left}     {right}")  # Add space between halves

    # Join the rows into the final string
    return "\n".join(result)

#==============#
# Run the code #
#==============#
if __name__ == "__main__":

    #------------------------------------------------------------#
    # Load, normalize, and combine bigram scores and frequencies #
    #------------------------------------------------------------#
    keyboard_layout = "qwertyuiopasdfghjkl;zxcvbnm,./"

    # File with bigram comfort scores for left and right sides of a computer keyboard
    bigram_scores_file = "output/all_bigram_comfort_scores.csv"
    comfort_scores = pd.read_csv(f"{bigram_scores_file}", index_col='bigram')

    from ngram_frequencies import onegram_frequencies, bigram_frequencies

    valid_bigrams = precompute_valid_bigrams(keyboard_layout, bigram_frequencies, comfort_scores)

    #--------------------------------------------------------------#
    # Stage 1: select keys and characters to arrange in those keys #
    #--------------------------------------------------------------#
    keys_to_replace = "asdf"
    chars_to_arrange = "etao"

    print(f"Keys to replace: {keys_to_replace}")
    print(f"Characters to arrange in the keys: {chars_to_arrange}")
    
    optimized_layout, score = optimize_subset(keys_to_replace, chars_to_arrange, valid_bigrams, keyboard_layout)
    
    # Print the original layout with new characters in their respective positions
    print("Original positions of keys to arrange:")
    print(display_keyboard(keyboard_layout, keys_to_replace, chars_to_arrange))

    # Print the optimized layout, assuming optimized_layout is properly computed
    print("\nOptimized Layout:")
    print(display_keyboard(optimized_layout, keys_to_replace, chars_to_arrange))

