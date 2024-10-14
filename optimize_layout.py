import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

#=========================================#
# Functions to optimizge keyboard layouts #
#=========================================#
def precompute_valid_bigrams(layout, bigram_frequencies, comfort_scores):
    layout_chars = set(layout)
    valid_bigrams = {}
    
    # Normalize comfort scores to [0, 1] range
    min_comfort = comfort_scores['comfort_score'].min()
    max_comfort = comfort_scores['comfort_score'].max()
    comfort_scores['normalized_comfort'] = (comfort_scores['comfort_score'] - min_comfort) / (max_comfort - min_comfort)
    
    # Normalize bigram frequencies to [0, 1] range
    freq_values = np.array(list(bigram_frequencies.values()))
    min_freq = freq_values.min()
    max_freq = freq_values.max()
    
    for char1, char2 in itertools.permutations(layout_chars, 2): 
        bigram = char1 + char2
        if bigram in bigram_frequencies:
            keyboard_bigram = layout[layout.index(char1)] + layout[layout.index(char2)]
            
            # Normalize frequency
            normalized_freq = (bigram_frequencies[bigram] - min_freq) / (max_freq - min_freq)
            
            if keyboard_bigram in comfort_scores.index:
                comfort = comfort_scores.loc[keyboard_bigram, 'normalized_comfort']
            elif (char1 in left_keys and char2 in right_keys) or (char1 in right_keys and char2 in left_keys):
                comfort = 1.0  # Set left-right bigrams to maximum normalized comfort
            else:
                continue  # Skip bigrams we don't have comfort data for
            
            # Multiply normalized frequency and normalized comfort score
            valid_bigrams[bigram] = normalized_freq * comfort

    return valid_bigrams

def calculate_layout_score(layout, valid_bigrams):
    score = 0
    for i, char1 in enumerate(layout):
        for j, char2 in enumerate(layout):
            if i != j:  # Avoid repeat-key bigrams
                bigram = char1 + char2
                if bigram in valid_bigrams:
                    score += valid_bigrams[bigram]
    return score

def optimize_subset(keys_to_arrange, new_chars, valid_bigrams, keyboard_layout):
    best_score = float('-inf')
    best_layout = None
    
    # Create a layout template with placeholders for the keys to arrange
    layout_template = list(keyboard_layout)
    indices_to_change = [keyboard_layout.index(key) for key in keys_to_arrange]
    
    # Calculate total number of permutations for the progress bar
    total_permutations = math.factorial(len(new_chars))
    
    # Generate all permutations of the new characters
    with tqdm(total=total_permutations, desc="Optimizing layout") as pbar:
        for perm in itertools.permutations(new_chars):
            # Create a new layout by placing the permuted characters in the template
            new_layout = layout_template.copy()
            for i, index in enumerate(indices_to_change):
                new_layout[index] = perm[i]
            
            new_layout = ''.join(new_layout)
            
            # Calculate the score for this layout
            score = calculate_layout_score(new_layout, valid_bigrams)
            
            if score > best_score:
                best_score = score
                best_layout = new_layout
            
            # Update the progress bar
            pbar.update(1)
    
    print(f"Best layout found: {best_layout}")  # Add this line for debugging
    return best_layout, best_score

def display_keyboard(layout, keys_to_arrange, new_chars):
    keyboard = [
        "┌───┬───┬───┬───┬───┐ ┌───┬───┬───┬───┬───┐",
        "│   │   │   │   │   │ │   │   │   │   │   │",
        "├───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┤",
        "│   │   │   │   │   │ │   │   │   │   │   │",
        "├───┼───┼───┼───┼───┤ ├───┼───┼───┼───┼───┤",
        "│   │   │   │   │   │ │   │   │   │   │   │",
        "└───┴───┴───┴───┴───┘ └───┴───┴───┴───┴───┘"
    ]
    
    layout_chars = list(layout)
    keys_to_show = set(keys_to_arrange)
    
    for i, row in enumerate([1, 3, 5]):
        for j in range(10):
            char = layout_chars[i*10 + j]
            if char in keys_to_show:
                if j < 5:
                    keyboard[row] = keyboard[row][:4*j+2] + f" {char} " + keyboard[row][4*j+4:]
                else:
                    keyboard[row] = keyboard[row][:4*(j-5)+23] + f" {char} " + keyboard[row][4*(j-5)+25:]
    
    return "\n".join(keyboard)

#==============#
# Run the code #
#==============#
if __name__ == "__main__":

    #----------------------------------------------#
    # Define keyboard layout and new character set #
    #----------------------------------------------#
    keyboard_layout = "qwertyuiopasdfghjkl;zxcvbnm,./"
    left_keys = set("qwertasdfgzxcvb")
    right_keys = set("yuiophjkl;nm,./")
    new_chars = "abcdefghijklmnopqrstuvwxyz;,./"

    #------------------------------------#
    # Load bigram scores and frequencies #
    #------------------------------------#
    # File with bigram comfort scores for left and right sides of a computer keyboard
    bigram_scores_file = "output/all_bigram_comfort_scores.csv"
    comfort_scores = pd.read_csv(f"{bigram_scores_file}", index_col='bigram')

    from ngram_frequencies import onegram_frequencies, bigram_frequencies

    #--------------------------------------------------------------#
    # Stage 1: select keys and characters to arrange in those keys #
    #--------------------------------------------------------------#
    keys_to_arrange = "asdf"  # Home row keys
    new_chars = "etao"  # Most common letters in English

    print(f"Keys to arrange: {keys_to_arrange}")
    print(f"New characters: {new_chars}")

    valid_bigrams = precompute_valid_bigrams(keyboard_layout, bigram_frequencies, comfort_scores)
    
    optimized_layout, score = optimize_subset(keys_to_arrange, new_chars, valid_bigrams, keyboard_layout)
    
    print("\nOriginal positions of keys to arrange:")
    print(display_keyboard(keyboard_layout, keys_to_arrange, new_chars))
    print("\nOptimized Layout:")
    print(display_keyboard(optimized_layout, keys_to_arrange, new_chars))
    print(f"\nLayout score: {score}")