import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from collections import defaultdict

#=========================================#
# Functions to optimizge keyboard layouts #
#=========================================#
def precompute_valid_bigrams(keyboard_layout, bigram_frequencies, comfort_scores, alpha=0.5, 
                             scaling_strategy="sqrt", offset=0.01, plot_filename=None):
    """
    Optimizes the computation of valid bigrams by caching character positions.
    Provides three scaling options: square root, offset, or logarithmic scaling.
    """
    character_set = set(keyboard_layout)
    valid_bigrams = {}

    # Define left and right sides of the keyboard
    left_keys = set("qwertasdfgzxcvb")
    right_keys = set("yuiophjkl;nm,./")

    # Set max comfort value for missing left-right bigrams
    max_comfort = comfort_scores['comfort_score'].max()
    for char1 in left_keys:
        for char2 in right_keys:
            bigram = char1 + char2
            if bigram not in comfort_scores.index:
                comfort_scores.loc[bigram] = max_comfort
            #if bigram in comfort_scores.index:
            #    print(f"{bigram}: {comfort_scores.loc[bigram, 'comfort_score']}")

    # Normalize comfort scores with the chosen scaling strategy and offset
    comfort_scores['normalized_comfort'] = normalize(
        comfort_scores['comfort_score'], strategy=scaling_strategy, offset=offset
    )
    #print("Normalized comfort scores:")
    #print(comfort_scores['normalized_comfort'].describe())

    # Convert bigram frequencies into a Pandas Series for normalization
    bigram_freq_series = pd.Series(bigram_frequencies)

    # Normalize bigram frequencies with the same strategy and offset
    normalized_frequencies = normalize(
        bigram_freq_series, strategy=scaling_strategy, offset=offset
    ).to_dict()

    # Cache character positions for faster lookup
    char_positions = {char: idx for idx, char in enumerate(keyboard_layout)}

    # Iterate through all valid bigrams using cached positions
    for char1, char2 in itertools.permutations(character_set, 2):
        bigram = char1 + char2
        if bigram in bigram_frequencies:
            normalized_freq = normalized_frequencies[bigram]

            # Construct the keyboard bigram from cached positions
            keyboard_bigram = (
                keyboard_layout[char_positions[char1]] +
                keyboard_layout[char_positions[char2]]
            )

            # Retrieve the normalized comfort score
            comfort = comfort_scores['normalized_comfort'].get(keyboard_bigram, 1.0)

            # Store the valid bigram score
            valid_bigrams[bigram] = alpha * normalized_freq + (1 - alpha) * comfort

    # Optionally plot the bigram scores
    if plot_filename is not None:
        plot_comparison_histograms(bigram_frequencies, comfort_scores, normalized_frequencies, "compare_histogram_"+plot_filename)
        plot_bigram_scores_histogram(valid_bigrams, normalized_frequencies, comfort_scores, "histogram_"+plot_filename)
        plot_bigram_scores(valid_bigrams, normalized_frequencies, comfort_scores, plot_filename)
    return valid_bigrams

def normalize(series, strategy="sqrt", offset=0.01):
    """Applies the selected normalization strategy to a Pandas Series."""
    normalized = (series - series.min()) / (series.max() - series.min())

    if strategy == "sqrt":
        normalized = np.sqrt(normalized)
    elif strategy == "log":
        normalized = np.log1p(normalized)  # log(1 + x) to handle zero values

    # Apply the offset once, at the end of normalization
    return normalized + offset

def plot_comparison_histograms(bigram_frequencies, comfort_scores, normalized_frequencies, plot_filename):
    """Plots histograms comparing raw and normalized distributions of frequencies and comfort scores."""
    # Prepare data for plotting
    raw_frequencies = list(bigram_frequencies.values())
    raw_comforts = comfort_scores['comfort_score'].values
    normalized_comforts = comfort_scores['normalized_comfort'].values
    normalized_freq_values = list(normalized_frequencies.values())

    # Create a figure with two rows: Raw and Normalized histograms
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot raw frequencies
    axes[0, 0].hist(raw_frequencies, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Raw Frequencies')
    axes[0, 0].set_xlabel('Frequency')
    axes[0, 0].set_ylabel('Count')

    # Plot raw comfort scores
    axes[0, 1].hist(raw_comforts, bins=30, alpha=0.7, color='orange')
    axes[0, 1].set_title('Raw Comfort Scores')
    axes[0, 1].set_xlabel('Comfort Score')
    axes[0, 1].set_ylabel('Count')

    # Plot normalized frequencies
    axes[1, 0].hist(normalized_freq_values, bins=30, alpha=0.7, color='blue')
    axes[1, 0].set_title('Normalized Frequencies')
    axes[1, 0].set_xlabel('Normalized Frequency')
    axes[1, 0].set_ylabel('Count')

    # Plot normalized comfort scores
    axes[1, 1].hist(normalized_comforts, bins=30, alpha=0.7, color='orange')
    axes[1, 1].set_title('Normalized Comfort Scores')
    axes[1, 1].set_xlabel('Normalized Comfort Score')
    axes[1, 1].set_ylabel('Count')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
   
def plot_bigram_scores_histogram(valid_bigrams, normalized_frequencies, comfort_scores, plot_filename):
    """Plots the distributions of frequencies, comfort scores, and combined scores as histograms."""
    # Prepare data for plotting
    combined_scores = list(valid_bigrams.values())
    frequencies = list(normalized_frequencies.values())
    comforts = comfort_scores['normalized_comfort'].values

    # Create a figure with three histograms side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Plot the histogram for normalized frequencies
    axes[0].hist(frequencies, bins=30, alpha=0.7, color='blue')
    axes[0].set_title('Normalized Frequencies')
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Count')

    # Plot the histogram for normalized comforts
    axes[1].hist(comforts, bins=30, alpha=0.7, color='orange')
    axes[1].set_title('Normalized Comforts')
    axes[1].set_xlabel('Score')

    # Plot the histogram for combined scores (Freq * Comfort)
    axes[2].hist(combined_scores, bins=30, alpha=0.7, color='black')
    axes[2].set_title('Combined Scores (Freq * Comfort)')
    axes[2].set_xlabel('Score')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    
def plot_bigram_scores(valid_bigrams, normalized_frequencies, comfort_scores, plot_filename):
    """Plots the bigram scores, frequencies, and comforts as scatter plots."""
    bigrams = list(valid_bigrams.keys())
    combined_scores = list(valid_bigrams.values())
    frequencies = [normalized_frequencies[bigram] for bigram in bigrams]
    comforts = [
        comfort_scores.loc[bigram, 'normalized_comfort'] if bigram in comfort_scores.index else 1.0
        for bigram in bigrams
    ]

    # Plot the distributions as scatter plots
    plt.figure(figsize=(14, 7))

    # Scatter plot for normalized frequencies
    plt.scatter(bigrams, frequencies, label='Normalized Frequencies', color="blue", alpha=0.7)

    # Scatter plot for normalized comforts
    plt.scatter(bigrams, comforts, label='Normalized Comforts', color="orange", alpha=0.7)

    # Scatter plot for combined scores (Freq * Comfort)
    plt.scatter(bigrams, combined_scores, label='Combined Scores (Freq * Comfort)', color="black", alpha=0.7)

    # Configure plot appearance
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.xlabel('Bigrams')
    plt.ylabel('Scores')
    plt.title('Comparison of Bigram Scores, Frequencies, and Comforts')
    plt.legend()
    plt.tight_layout()

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)

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

        # Identify positions of the replaced keys in the layout
        key_positions = [keyboard_layout.index(key) for key in keys_to_replace]

        # Calculate the layout score (inlined for performance)
        score = 0
        for i in key_positions:
            for j in key_positions:  # Only compare with other relevant keys
                if i != j:
                    bigram = temp_layout[i] + temp_layout[j]
                    bigram_score = valid_bigrams.get(bigram, 0)
                    score += bigram_score

                    # Debugging print to confirm only relevant bigrams are evaluated
                    print(f"Evaluating relevant bigram: {bigram}, Score: {bigram_score}")
        print(f"Temporary Layout: {''.join(temp_layout)} score: {score}")

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
    bigram_scores_file = "output_bayes_bigram_scoring/all_bigram_comfort_scores.csv"
    comfort_scores = pd.read_csv(f"{bigram_scores_file}", index_col='bigram')
    #print("Min comfort score:", comfort_scores['comfort_score'].min())
    #print("Max comfort score:", comfort_scores['comfort_score'].max())

    from ngram_frequencies import onegram_frequencies, bigram_frequencies

    alpha = 0.5
    scaling_strategy = "sqrt" 
    baseline_offset = 0.1
    plot_filename = "scores.png"
    valid_bigrams = precompute_valid_bigrams(keyboard_layout, bigram_frequencies, 
                        comfort_scores, alpha, scaling_strategy, baseline_offset, plot_filename)

    # Call the function with square root scaling

    #--------------------------------------------------------------#
    # Stage 1: select keys and characters to arrange in those keys #
    #--------------------------------------------------------------#
    keys_to_replace = "asd"
    chars_to_arrange = "eic"

    print(f"Keys to replace: {keys_to_replace}")
    print(f"Characters to arrange in the keys: {chars_to_arrange}")
    
    optimized_layout, score = optimize_subset(keys_to_replace, chars_to_arrange, valid_bigrams, keyboard_layout)
    
    # Print the original layout with new characters in their respective positions
    print("Original positions of keys to arrange:")
    print(display_keyboard(keyboard_layout, keys_to_replace, chars_to_arrange))

    # Print the optimized layout, assuming optimized_layout is properly computed
    print("\nOptimized Layout:")
    print(display_keyboard(optimized_layout, keys_to_replace, chars_to_arrange))

