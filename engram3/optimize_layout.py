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

# First, define both mappings
RIGHT_POSITIONS = ['u', 'i', 'o', 'p', 'j', 'k', 'l', ';', 'm', ',', '.', '/']
LEFT_POSITIONS = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']

# Create the position mapping (right to left)
POSITION_MAP = {
    'u': 'r', 'i': 'e', 'o': 'w', 'p': 'q',  # Top row
    'j': 'f', 'k': 'd', 'l': 's', ';': 'a',  # Home row
    'm': 'v', ',': 'c', '.': 'x', '/': 'z'   # Bottom row
}

def analyze_vowel_bigrams():
    """Analyze all vowel-vowel bigrams and their frequencies."""
    vowels = 'aeiou'
    vowel_bigrams = {}
    
    # Get frequencies for all vowel pairs in both directions
    for v1 in vowels:
        for v2 in vowels:
            if v1 != v2:
                bigram = v1 + v2
                freq = bigram_frequencies.get(bigram, 0)
                vowel_bigrams[bigram] = freq
    
    # Sort by frequency
    sorted_bigrams = sorted(vowel_bigrams.items(), key=lambda x: x[1], reverse=True)
    print("\nVowel Bigram Frequencies:")
    for bigram, freq in sorted_bigrams:
        print(f"{bigram}: {freq:.6f}")
    
    return vowel_bigrams

def analyze_comfort_scores(comfort_scores):
    """Analyze raw comfort scores to verify our data."""
    # Look at comfort scores for each position
    position_comfort = {}
    for (pos1, pos2), score in comfort_scores.items():
        if pos1 not in position_comfort:
            position_comfort[pos1] = []
        position_comfort[pos1].append(score)
    
    # Print average comfort score for each position
    print("\nAverage comfort scores by position:")
    for pos, scores in position_comfort.items():
        avg_score = sum(scores) / len(scores)
        print(f"{pos}: {avg_score:.4f}")

def analyze_position_comfort():
    """Analyze comfort scores for each row position."""
    comfort_scores = get_comfort_scores()
    
    # Analyze home row positions
    print("\nHome row comfort scores:")
    home_pairs = [('f','d'), ('d','s'), ('s','a'), ('d','f'), ('s','d'), ('a','s')]
    for p1, p2 in home_pairs:
        score = comfort_scores.get((p1, p2), float('-inf'))
        print(f"{p1}->{p2}: {score:.4f}")
    
    # Analyze top row positions
    print("\nTop row comfort scores:")
    top_pairs = [('r','e'), ('e','w'), ('w','q'), ('e','r'), ('w','e'), ('q','w')]
    for p1, p2 in top_pairs:
        score = comfort_scores.get((p1, p2), float('-inf'))
        print(f"{p1}->{p2}: {score:.4f}")
    
    # Analyze bottom row positions
    print("\nBottom row comfort scores:")
    bottom_pairs = [('v','c'), ('c','x'), ('x','z'), ('c','v'), ('x','c'), ('z','x')]
    for p1, p2 in bottom_pairs:
        score = comfort_scores.get((p1, p2), float('-inf'))
        print(f"{p1}->{p2}: {score:.4f}")

def print_keyboard_layout(positions: Dict[str, str], title: str = "Layout"):
    """
    Print a visual representation of the keyboard layout.
    positions: Dictionary mapping positions to letters (or empty strings for unused positions)
    """
    # Standard QWERTY layout positions for reference
    layout_template = """
╭───────────────────────────────────────────────╮
│ Layout: {title:<34}    │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
│ {q:^3} │ {w:^3} │ {e:^3} │ {r:^3} ║ {u:^3} │ {i:^3} │ {o:^3} │ {p:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
│ {a:^3} │ {s:^3} │ {d:^3} │ {f:^3} ║ {j:^3} │ {k:^3} │ {l:^3} │ {sc:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│  Z  │  X  │  C  │  V  ║  M  │  ,  │  .  │  /  │
│ {z:^3} │ {x:^3} │ {c:^3} │ {v:^3} ║ {m:^3} │ {cm:^3} │ {dt:^3} │ {sl:^3} │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯
"""
    
    # Create a dictionary for all possible positions, with spaces as defaults
    layout_chars = {
        'q': ' ', 'w': ' ', 'e': ' ', 'r': ' ',
        'u': ' ', 'i': ' ', 'o': ' ', 'p': ' ',
        'a': ' ', 's': ' ', 'd': ' ', 'f': ' ',
        'j': ' ', 'k': ' ', 'l': ' ', 'sc': ' ',
        'z': ' ', 'x': ' ', 'c': ' ', 'v': ' ',
        'm': ' ', 'cm': ' ', 'dt': ' ', 'sl': ' '
    }
    
    # Convert special characters in positions to their keys in layout_chars
    position_conversion = {
        ',': 'cm',
        '.': 'dt',
        '/': 'sl',
        ';': 'sc'
    }
    
    # Update with provided positions, converting special characters
    for pos, letter in positions.items():
        layout_key = position_conversion.get(pos, pos)
        layout_chars[layout_key] = letter.upper()
    
    # Print the layout
    print(layout_template.format(title=title, **layout_chars))

def print_detailed_placement(score: float, positions: Tuple[str, ...], 
                           bigram_scores: Dict[str, float], letters: str):
    """Print detailed analysis of a placement with visual keyboard layout."""
    print(f"\nPlacement Score: {score:.6f}")
    
    # Create position mapping
    pos_map = {}
    if isinstance(letters, str) and letters == 'aeiou':
        # Handle vowels (right side)
        # Map each position in the standard layout to its assigned letter
        position_to_letter = dict(zip(positions, letters))
        for pos in ['u', 'i', 'o', 'p', 'j', 'k', 'l', ';', 'm', ',', '.', '/']:
            if pos in position_to_letter:
                pos_map[pos] = position_to_letter[pos]
    else:
        # Handle consonants (left side)
        position_to_letter = dict(zip(positions, letters))
        for pos in ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']:
            if pos in position_to_letter:
                pos_map[pos] = position_to_letter[pos]
    
    # Debug print to see the mapping
    print("Debug - Letters being placed:", letters)
    print("Debug - Positions assigned:", positions)
    print("Debug - Final position mapping:", pos_map)
    
    # Print visual keyboard layout
    print_keyboard_layout(pos_map, f"Score: {score:.4f}")
    
    # Print bigram contributions
    print("\nTop Bigram Contributions (sorted by impact):")
    sorted_bigrams = sorted(bigram_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    for bigram, contribution in sorted_bigrams[:10]:  # Show top 10 contributions
        if contribution != 0:
            print(f"  {bigram}: {contribution:>10.6f}")

def get_bigram_frequencies() -> Dict[str, float]:
    """Load bigram frequencies from the imported bigram_frequencies module."""
    return bigram_frequencies

def get_comfort_scores() -> Dict[Tuple[str, str], float]:
    """Load comfort scores from CSV file."""
    df = pd.read_csv('estimated_bigram_scores.csv')
    comfort_scores = {}
    for _, row in df.iterrows():
        bigram = (row['first_char'], row['second_char'])
        comfort_scores[bigram] = row['comfort_score']
    return comfort_scores
    
def find_best_vowel_placements(vowels: str, positions: List[str],
                             bigram_freqs: Dict[str, float],
                             comfort_scores: Dict[Tuple[str, str], float],
                             top_n: int) -> List[Tuple[float, List[str], Dict[str, float]]]:
    """Find optimal placements for vowels based on frequency-weighted comfort scores."""
    print(f"\nSearching for best placements of vowels {vowels}")
    print(f"Available positions: {positions}")
    print(f"Will return top {top_n} placements")
    
    best_placements = []
    count = 0
    
    # Try all possible positions
    for pos_perm in permutations(positions, len(vowels)):
        weighted_score, bigram_scores = score_vowel_permutation(vowels, pos_perm, bigram_freqs, comfort_scores)
        
        # Keep track of top N with their bigram breakdowns
        if len(best_placements) < top_n:
            heapq.heappush(best_placements, (-weighted_score, pos_perm, bigram_scores))
        elif -weighted_score > best_placements[0][0]:
            heapq.heappop(best_placements)
            heapq.heappush(best_placements, (-weighted_score, pos_perm, bigram_scores))
        
        count += 1
        if count % 1000 == 0:
            print(f"Evaluated {count} permutations...")
    
    print(f"Evaluated total of {count} permutations")
    print(f"Found {len(best_placements)} best placements")
    
    # Convert and sort (remove negative scores we used for heap)
    return sorted([(-score, perm, b_scores) for score, perm, b_scores in best_placements], 
                 reverse=True)

def score_vowel_permutation(vowels: str, positions: List[str], 
                          bigram_freqs: Dict[str, float],
                          comfort_scores: Dict[Tuple[str, str], float]) -> Tuple[float, Dict[str, float]]:
    """Score a permutation by average comfort of all possible vowel pair transitions.
    
    1. Try all possible placements of the 5 vowels in the 12 right-side positions
    2. For each placement, calculate the average (frequency x comfort score) of all possible vowel-to-vowel transitions
    3. Find the placement that maximizes this average comfort score
    
    """
    position_map = dict(zip(vowels, positions))
    total_score = 0
    bigram_scores = {}
    
    # Debug first permutation
    static_count = getattr(score_vowel_permutation, 'count', 0)
    if static_count == 0:
        print("\nDebug first permutation:")
        print(f"Vowel to position mapping: {position_map}")
    score_vowel_permutation.count = static_count + 1
    
    # For each possible vowel pair
    for v1 in vowels:
        for v2 in vowels:
            if v1 != v2:  # Don't score same-vowel transitions
                bigram = f"{v1}{v2}"
                freq = bigram_freqs.get(bigram, 0)
                
                # Get assigned positions
                pos1, pos2 = position_map[v1], position_map[v2]
                
                # Map to left-side positions for comfort lookup
                left_pos1 = POSITION_MAP[pos1]
                left_pos2 = POSITION_MAP[pos2]
                
                # Get comfort score and weight by frequency
                comfort = comfort_scores.get((left_pos1, left_pos2), float('-inf'))
                weighted_score = freq * comfort
                bigram_scores[bigram] = weighted_score
                total_score += weighted_score
                
                if static_count == 0:
                    print(f"\nBigram {bigram}:")
                    print(f"  Positions: {pos1}->{pos2}")
                    print(f"  Maps to left side: {left_pos1}->{left_pos2}")
                    print(f"  Frequency: {freq:.6f}")
                    print(f"  Comfort score: {comfort:.4f}")
                    print(f"  Weighted score: {weighted_score:.6f}")
    
    return total_score, bigram_scores

def optimize_layout(num_consonants: int, top_n: int = 10):
    """Find optimal placements for vowels."""
    # Load data
    bigram_freqs = get_bigram_frequencies()
    comfort_scores = get_comfort_scores()
    
    # Find best vowel placements
    vowels = 'aeiou'
    print(f"\nFinding best placements for vowels in positions: {RIGHT_POSITIONS}")
    print(f"Testing {len(list(permutations(RIGHT_POSITIONS, len(vowels))))} permutations...")
    
    vowel_placements = find_best_vowel_placements(
        vowels, RIGHT_POSITIONS, bigram_freqs, comfort_scores, top_n
    )
    
    # Print detailed results
    print("\nTop Vowel Placements:")
    for i, (score, positions, bigram_scores) in enumerate(vowel_placements, 1):
        print(f"\nPlacement {i}:")
        print_detailed_placement(score, positions, bigram_scores, vowels)
    
    return vowel_placements

if __name__ == "__main__":
    top_n = 1  # Return top placements

    #comfort_scores = get_comfort_scores()
    #analyze_comfort_scores(comfort_scores)

    #print("Analyzing position comfort scores...")
    #analyze_position_comfort()

    vowel_placements = optimize_layout(None, top_n)


