import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from collections import defaultdict

#====================================#
# Functions to prepare bigram scores #
#====================================#
def precompute_valid_bigrams(qwerty_layout, qwerty_comfort_scores, bigram_frequencies, 
                             scaling_strategy="sqrt", offset=0.01, plot=True, save_plot=None):
    """
    Precomputes normalized bigram frequencies and QWERTY comfort scores.
    Also, optionally plots their distributions.
    """
    # Define left and right keys on the keyboard
    left_keys = set("qwertasdfgzxcvb")
    right_keys = set("yuiophjkl;nm,./")

    # Ensure qwerty_comfort_scores is a DataFrame
    if isinstance(qwerty_comfort_scores, dict):
        qwerty_comfort_scores = pd.DataFrame.from_dict(qwerty_comfort_scores, 
                                                       orient='index', 
                                                       columns=['comfort_score'])
    # Ensure that bigram_frequencies is a 1D dictionary
    if isinstance(bigram_frequencies, pd.DataFrame):
        bigram_frequencies = bigram_frequencies.squeeze().to_dict()
    elif isinstance(bigram_frequencies, np.ndarray):
        bigram_frequencies = dict(enumerate(bigram_frequencies.flatten()))

    # Assign the max comfort score to missing left & right bigrams
    max_comfort = qwerty_comfort_scores['comfort_score'].max()
    missing_bigrams = []

    for char1, char2 in itertools.permutations(qwerty_layout, 2):
        bigram = char1 + char2
        if bigram not in qwerty_comfort_scores.index:
            # Check if the bigram crosses left and right sides
            if (char1 in left_keys and char2 in right_keys) or \
               (char1 in right_keys and char2 in left_keys):
                # Assign max comfort value to missing bigrams
                qwerty_comfort_scores.loc[bigram] = max_comfort
            else:
                # Collect non-left-right bigrams for debugging
                missing_bigrams.append(bigram)

    # Raise a warning if unexpected missing bigrams are found
    if missing_bigrams:
        print(f"Warning: Unexpected missing bigrams detected: {missing_bigrams}")

    # Normalize bigram frequencies
    bigram_freq_series = pd.Series(bigram_frequencies)
    normalized_frequencies = normalize(bigram_freq_series, scaling_strategy, offset).to_dict()

    # Normalize comfort scores for QWERTY key pairs
    qwerty_comfort_scores['normalized_comfort'] = normalize(qwerty_comfort_scores['comfort_score'], 
                                                            scaling_strategy, offset)

    # Create a lookup dictionary for QWERTY key pair comfort scores
    qwerty_comfort = {
        char1 + char2: qwerty_comfort_scores.loc[char1 + char2, 'normalized_comfort']
        for char1, char2 in itertools.permutations(qwerty_layout, 2)
    }

    # Optional: Plot raw and normalized distributions
    if plot:
        plot_comparison_histograms(bigram_frequencies, normalized_frequencies, qwerty_comfort_scores, save_plot)

    return normalized_frequencies, qwerty_comfort

def normalize(series, strategy="sqrt", offset=0.01):
    """Applies the selected normalization strategy to a Pandas Series."""
    normalized = (series - series.min()) / (series.max() - series.min())

    if strategy == "sqrt":
        normalized = np.sqrt(normalized)
    elif strategy == "log":
        normalized = np.log1p(normalized)  # log(1 + x) to handle zero values

    # Apply the offset once, at the end of normalization
    return normalized + offset

def plot_comparison_histograms(bigram_frequencies, normalized_frequencies, qwerty_comfort_scores, save_plot):
    """Plots histograms comparing raw and normalized distributions of frequencies and comfort scores."""
    # Prepare data for plotting
    raw_frequencies = list(bigram_frequencies.values())
    normalized_freq_values = list(normalized_frequencies.values())
    raw_comforts = qwerty_comfort_scores['comfort_score'].values
    normalized_comforts = qwerty_comfort_scores['normalized_comfort'].values

    # Create a figure with two rows: Raw and Normalized histograms
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot raw frequencies
    axes[0, 0].hist(raw_frequencies, bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Raw Frequencies')
    axes[0, 0].set_xlabel('Frequency')
    axes[0, 0].set_ylabel('Count')

    # Plot raw comfort scores
    axes[0, 1].hist(raw_comforts, bins=30, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Raw Comfort Scores')
    axes[0, 1].set_xlabel('Comfort Score')
    axes[0, 1].set_ylabel('Count')

    # Plot normalized frequencies
    axes[1, 0].hist(normalized_freq_values, bins=30, alpha=0.7, color='blue')
    axes[1, 0].set_title('Normalized Frequencies')
    axes[1, 0].set_xlabel('Normalized Frequency')
    axes[1, 0].set_ylabel('Count')

    # Plot normalized comfort scores
    axes[1, 1].hist(normalized_comforts, bins=30, alpha=0.7, color='green')
    axes[1, 1].set_title('Normalized Comfort Scores')
    axes[1, 1].set_xlabel('Normalized Comfort Score')
    axes[1, 1].set_ylabel('Count')

    # Adjust layout and save the plot
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot, dpi=300)
    plt.show()

#========================================#
# Functions to optimize keyboard layouts #
#========================================#
def optimize_subset(qwerty_keys, chars_to_arrange, normalized_frequencies, normalized_qwerty_comfort, alpha=0.5):
    """
    Optimizes a subset of the layout using precomputed frequencies and QWERTY comfort scores.
    """
    best_layout = ''.join(chars_to_arrange)
    best_score = -np.inf

    # Iterate over permutations of the chars_to_arrange
    for permutation in itertools.permutations(chars_to_arrange):
        layout = ''.join(permutation)
        total_score = 0

        # Calculate total score for this layout permutation
        for i, char1 in enumerate(layout):
            for j, char2 in enumerate(layout):
                if i != j:  # Avoid same key pairs
                    character_bigram = char1 + char2
                    qwerty_bigram = qwerty_keys[i] + qwerty_keys[j]

                    # Retrieve scores
                    freq_score = normalized_frequencies.get(character_bigram, 0)
                    comfort_score = normalized_qwerty_comfort.get(qwerty_bigram, 0)

                    # Accumulate score
                    total_score += alpha * freq_score + (1 - alpha) * comfort_score

        # Update best layout if the score is better
        if total_score > best_score:
            best_score = total_score
            best_layout = layout

    return best_layout, best_score

def update_layout_display(original_layout, subset_positions, optimized_subset):
    """
    Updates the original layout with the optimized subset of characters.
    """
    # Convert the original layout to a list for easy modification
    layout = list(original_layout)

    # Replace the original characters with the optimized ones
    for pos, char in zip(subset_positions, optimized_subset):
        layout[pos] = char

    return ''.join(layout)

def display_keyboard(layout, qwerty_keys, optimized_chars):
    """
    Displays the keyboard layout by splitting it into 3 rows of 10 keys.
    Uses '-' placeholders for all keys except the new characters.
    """
    # Initialize the layout with '-' placeholders
    filled_layout = ["-"] * 30

    # Find the positions of the QWERTY keys in the original layout
    key_positions = [layout.index(key) for key in qwerty_keys]

    # Place the optimized characters at the correct positions
    for pos, new_char in zip(key_positions, optimized_chars):
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

    #---------------------------------------------------------------#
    # Normalize Qwerty bigram frequencies and bigram comfort scores #
    #---------------------------------------------------------------#
    # Qwerty bigram comfort scores for left and right sides of a computer keyboard
    qwerty_layout = "qwertyuiopasdfghjkl;zxcvbnm,./"
    qwerty_comfort_scores_file = "output_bayes_bigram_scoring/all_bigram_comfort_scores.csv"
    qwerty_comfort_scores = pd.read_csv(f"{qwerty_comfort_scores_file}", index_col='bigram')

    # bigram frequencies
    from ngram_frequencies import bigram_frequencies

    # Precompute bigram comfort scores and qwerty bigram frequencies with square root scaling and offset
    normalized_frequencies, normalized_qwerty_comfort = precompute_valid_bigrams(qwerty_layout, 
                                                            qwerty_comfort_scores, 
                                                            bigram_frequencies, 
                                                            scaling_strategy="sqrt", 
                                                            offset=0.01, 
                                                            plot=False, 
                                                            save_plot="scores_histogram.png")

    #--------------------------------------------------------------#
    # Stage 1: select keys and characters to arrange in those keys #
    #--------------------------------------------------------------#
    qwerty_keys = "dfqz"  # Subset of keys to replace
    chars_to_arrange = "qzea"  # Characters to rearrange
    print(f"Qwertyu key positions: {qwerty_keys}")
    print(f"Characters to arrange: {chars_to_arrange}")
    
    # Step 1: Optimize a subset of the layout
    alpha = 0.5  # alpha * normalized_freq + (1 - alpha) * comfort
    chars_rearranged, best_score = optimize_subset(qwerty_keys, 
                                                   chars_to_arrange,
                                                   normalized_frequencies, 
                                                   normalized_qwerty_comfort, 
                                                   alpha)

    # Step 2: Determine the new order of QWERTY keys
    char_to_qwerty = dict(zip(chars_to_arrange, qwerty_keys))
    keys_rearranged = ''.join(char_to_qwerty[char] for char in chars_rearranged)

    # Step 3: Update the layout with the optimized arrangement
    qwerty_indices = [qwerty_layout.index(key) for key in qwerty_keys]
    new_qwerty_layout = update_layout_display(qwerty_layout, qwerty_indices, chars_rearranged)

    print(f"Layout score: {best_score}")
    print(f"Keys: {qwerty_keys} => {keys_rearranged}")
    print(f"Char: {chars_to_arrange} => {chars_rearranged}")
    print("Old:")
    print(display_keyboard(qwerty_layout, qwerty_keys, chars_to_arrange))    
    print("New:")
    print(display_keyboard(qwerty_layout, qwerty_keys, chars_rearranged))
