import numpy as np
from itertools import permutations
import heapq
from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict
from features.bigram_frequencies import (
    bigrams, bigram_frequencies,
    onegrams, onegram_frequencies_array
)
from features.features import (
    same_finger, sum_finger_values, adj_finger_diff_row,
    rows_apart, angle_apart, sum_engram_position_values, sum_row_position_values,
    finger_map, row_map, column_map, angles,
    engram_position_values, row_position_values
)

# Fixed positions
RIGHT_POSITIONS = ['u', 'i', 'o', 'p', 'j', 'k', 'l', ';', 'm', ',', '.', '/']
LEFT_POSITIONS = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']

# Position mapping for mirrored comfort scores (left to right)
POSITION_MAP = {
    'r': 'u', 'e': 'i', 'w': 'o', 'q': 'p',
    'f': 'j', 'd': 'k', 's': 'l', 'a': ';',
    'v': 'm', 'c': ',', 'x': '.', 'z': '/'
}

def get_bigram_frequencies() -> Dict[str, float]:
    """Return dictionary of bigram frequencies."""
    return bigram_frequencies

def get_comfort_scores() -> Dict[Tuple[str, str], float]:
    """Load comfort scores from CSV."""
    df = pd.read_csv('estimated_bigram_scores.csv')
    scores = {}
    for _, row in df.iterrows():
        scores[(row['first_char'], row['second_char'])] = row['comfort_score']
    return scores

def get_top_consonants(n: int) -> str:
    """Get n most frequent consonants from onegrams data."""
    consonants = []
    vowels = set('aeiou')
    
    for letter in onegrams:
        if letter not in vowels:
            consonants.append(letter)
            if len(consonants) == n:
                break
    
    return ''.join(consonants)

def get_bigram_score(bigram: str, positions: Dict[str, str], 
                    bigram_freqs: Dict[str, float], 
                    comfort_scores: Dict[Tuple[str, str], float],
                    is_right_side: bool = False) -> float:
    """
    Get frequency-weighted comfort score for a bigram in given positions.
    For right side bigrams, maps positions to equivalent left side positions.
    For left side bigrams, uses positions directly.
    """
    char1, char2 = bigram
    pos1, pos2 = positions[char1], positions[char2]
    
    # For right side positions, map their left-side equivalents
    if is_right_side:
        # Find which left-side positions map to these right-side positions
        pos1 = [left for left, right in POSITION_MAP.items() if right == pos1][0]
        pos2 = [left for left, right in POSITION_MAP.items() if right == pos2][0]
    
    freq = bigram_freqs.get(bigram, 0)
    comfort = comfort_scores.get((pos1, pos2), float('-inf'))
    
    # Debug print
    if freq > 0.001:  # Only print significant bigrams
        print(f"Bigram {bigram}: {char1}({pos1}) -> {char2}({pos2}), freq={freq}, comfort={comfort}")
    
    return freq * comfort

def score_placement(letters: str, positions: List[str], 
                   bigram_freqs: Dict[str, float],
                   comfort_scores: Dict[Tuple[str, str], float],
                   is_right_side: bool = False) -> float:
    """Score a placement of letters in positions."""
    position_map = dict(zip(letters, positions))
    total_score = 0
    
    # Score all possible bigrams
    for i, c1 in enumerate(letters):
        for c2 in letters[i+1:]:
            bigram = c1 + c2
            bigram_rev = c2 + c1
            
            # Add scores for both directions
            total_score += get_bigram_score(bigram, position_map, bigram_freqs, comfort_scores, is_right_side)
            total_score += get_bigram_score(bigram_rev, position_map, bigram_freqs, comfort_scores, is_right_side)
            
    return total_score

def find_best_placements(letters: str, positions: List[str],
                        bigram_freqs: Dict[str, float],
                        comfort_scores: Dict[Tuple[str, str], float],
                        is_right_side: bool = False,
                        top_n: int = 10) -> List[Tuple[float, Tuple[str, ...]]]:
    """
    Find top N best placements for given letters.
    Returns list of (score, positions) tuples sorted by descending score.
    """
    # Use heap to maintain top N placements
    best_placements = []
    count = 0
    
    # Try all possible positions
    for pos_perm in permutations(positions, len(letters)):
        score = score_placement(letters, pos_perm, bigram_freqs, comfort_scores, is_right_side)
        
        # Push negative score for min-heap behavior (keeps highest scores)
        if len(best_placements) < top_n:
            heapq.heappush(best_placements, (-score, pos_perm))
        elif -score > best_placements[0][0]:
            heapq.heapreplace(best_placements, (-score, pos_perm))
            
        count += 1
        if count % 100000 == 0:
            print(f"Evaluated {count} permutations...")
    
    # Convert back to positive scores and sort descending
    return [((-score), perm) for score, perm in sorted(best_placements)]

def optimize_layout(num_consonants: int, top_n: int = 10):
    """
    Find optimal placements for vowels and specified number of most frequent consonants.
    Returns (vowel_placements, consonant_placements).
    """
    # Load the frequency and comfort score data
    bigram_freqs = get_bigram_frequencies()
    comfort_scores = get_comfort_scores()
    
    # Print some diagnostics
    print("\nExample comfort scores:")
    print("Home row adjacent (sd):", comfort_scores.get(('s', 'd')))
    print("Same finger vertical (rf):", comfort_scores.get(('r', 'f')))
    print("Cross hand (df):", comfort_scores.get(('d', 'f')))
    
    # Optimize vowel placement
    vowels = 'aeiou'
    print(f"\nFinding best placements for vowels...")
    vowel_placements = find_best_placements(
        vowels, RIGHT_POSITIONS, bigram_freqs, comfort_scores, is_right_side=True, top_n=top_n
    )
    
    # Get top consonants and optimize their placement
    consonants = get_top_consonants(num_consonants)
    print(f"\nFinding best placements for consonants: {consonants}")
    consonant_placements = find_best_placements(
        consonants, LEFT_POSITIONS, bigram_freqs, comfort_scores, is_right_side=False, top_n=top_n
    )
    
    return vowel_placements, consonant_placements

def print_placements(placements: List[Tuple[float, Tuple[str, ...]]], letters: str):
    """Pretty print the placements with their scores."""
    for i, (score, positions) in enumerate(placements, 1):
        print(f"\nPlacement {i} (score: {score:.3f}):")
        for letter, pos in zip(letters, positions):
            print(f"  {letter}: {pos}")

if __name__ == "__main__":
    num_consonants = 8  # Place 8 most frequent consonants
    top_n = 10         # Return top 10 placements
    
    vowel_placements, consonant_placements = optimize_layout(num_consonants, top_n)
    
    print("\nTop vowel placements:")
    print_placements(vowel_placements, 'aeiou')
    
    consonants = get_top_consonants(num_consonants)
    print("\nTop consonant placements:")
    print_placements(consonant_placements, consonants)