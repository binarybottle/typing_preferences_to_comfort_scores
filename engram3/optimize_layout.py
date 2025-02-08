# optimize_layout.py
from itertools import permutations
import heapq
from typing import List, Dict, Tuple
import pandas as pd
from features.bigram_frequencies import (
    bigrams, bigram_frequencies,
    onegrams, onegram_frequencies
)

# Define keyboard positions
RIGHT_POSITIONS = ['u', 'i', 'o', 'p', 'j', 'k', 'l', ';', 'm', ',', '.', '/']
LEFT_POSITIONS = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']

#--------------------------------------------
# Core functions
#--------------------------------------------   
def get_comfort_scores() -> Dict[Tuple[str, str], float]:
    """Load comfort scores from CSV file."""
    df = pd.read_csv('estimated_bigram_scores.csv')
    comfort_scores = {}
    for _, row in df.iterrows():
        bigram = (row['first_char'], row['second_char'])
        comfort_scores[bigram] = row['comfort_score']
    return comfort_scores

def get_top_consonants(n: int) -> str:
    """Get the N most frequent consonants from the frequency data."""
    consonants = [c for c in 'bcdfghjklmnpqrstvwxz']
    consonant_freqs = {c: onegram_frequencies.get(c, 0) for c in consonants}
    
    print("\nConsonant frequencies:")
    for c, freq in sorted(consonant_freqs.items(), key=lambda x: x[1], reverse=True):
        print(f"{c}: {freq:.6f}")
    
    return ''.join(sorted(consonants, key=lambda x: consonant_freqs[x], reverse=True)[:n])

#--------------------------------------------
# Scoring functions
#--------------------------------------------   
def score_permutation(letters: str, positions: List[str], 
                     bigram_freqs: Dict[str, float],
                     comfort_scores: Dict[Tuple[str, str], float]) -> Tuple[float, Dict[str, float]]:
    """Score a permutation by frequency-weighted comfort of letter transitions."""
    position_map = dict(zip(letters, positions))
    total_score = 0
    bigram_scores = {}
    
    # For each possible letter pair
    for l1 in letters:
        for l2 in letters:
            if l1 != l2:
                bigram = f"{l1}{l2}"
                freq = bigram_freqs.get(bigram, 0)
                
                # Get assigned positions
                pos1, pos2 = position_map[l1], position_map[l2]
                
                # If these are right-side positions, map to left-side equivalents
                if pos1 in RIGHT_POSITIONS:
                    pos1 = {
                        'u': 'r', 'i': 'e', 'o': 'w', 'p': 'q',  # Top row
                        'j': 'f', 'k': 'd', 'l': 's', ';': 'a',  # Home row
                        'm': 'v', ',': 'c', '.': 'x', '/': 'z'   # Bottom row
                    }[pos1]
                if pos2 in RIGHT_POSITIONS:
                    pos2 = {
                        'u': 'r', 'i': 'e', 'o': 'w', 'p': 'q',  # Top row
                        'j': 'f', 'k': 'd', 'l': 's', ';': 'a',  # Home row
                        'm': 'v', ',': 'c', '.': 'x', '/': 'z'   # Bottom row
                    }[pos2]
                
                # Get comfort score
                comfort = comfort_scores.get((pos1, pos2), float('-inf'))
                weighted_score = freq * comfort
                bigram_scores[bigram] = weighted_score
                total_score += weighted_score
    
    return total_score, bigram_scores

def find_best_placements(letters: str, positions: List[str],
                        bigram_freqs: Dict[str, float],
                        comfort_scores: Dict[Tuple[str, str], float],
                        top_n: int) -> List[Tuple[float, List[str], Dict[str, float]]]:
    """Find optimal placements for letters based on frequency-weighted comfort scores."""
    print(f"\nSearching for best placements of letters {letters}")
    print(f"Available positions: {positions}")
    print(f"Will return top {top_n} placements")
    
    best_placements = []
    count = 0
    
    # Try all possible positions
    for pos_perm in permutations(positions, len(letters)):
        weighted_score, bigram_scores = score_permutation(
            letters, pos_perm, bigram_freqs, comfort_scores
        )
        
        # Keep track of top N with their bigram breakdowns
        if len(best_placements) < top_n:
            heapq.heappush(best_placements, (weighted_score, pos_perm, bigram_scores))
        elif weighted_score > best_placements[0][0]:
            heapq.heappop(best_placements)
            heapq.heappush(best_placements, (weighted_score, pos_perm, bigram_scores))
        
        count += 1
        if count % 1000 == 0:
            print(f"Evaluated {count} permutations...")
    
    print(f"Evaluated total of {count} permutations")
    print(f"Found {len(best_placements)} best placements")
    
    return sorted(best_placements, reverse=True)

#--------------------------------------------
# Layout optimization functions
#--------------------------------------------   
def optimize_consonant_layout(num_consonants: int, top_n: int = 10):
    """Find optimal placements for N consonants on the left side."""
    bigram_freqs = bigram_frequencies
    comfort_scores = get_comfort_scores()
    
    # Get most frequent consonants
    consonants = get_top_consonants(num_consonants)
    print(f"\nFinding best placements for consonants: {consonants}")
    print(f"in left-side positions: {LEFT_POSITIONS}")
    print(f"Testing {len(list(permutations(LEFT_POSITIONS, len(consonants))))} permutations...")
    
    consonant_placements = find_best_placements(
        consonants, LEFT_POSITIONS, bigram_freqs, comfort_scores, top_n
    )
    
    # Print detailed results
    print("\nTop Consonant Placements:")
    for i, (score, positions, bigram_scores) in enumerate(consonant_placements, 1):
        print(f"\nPlacement {i}:")
        print_detailed_placement(score, positions, bigram_scores, consonants)
    
    return consonant_placements

def optimize_vowel_layout(top_n: int = 10):
    """Find optimal placements for vowels on the right side."""
    bigram_freqs = bigram_frequencies
    comfort_scores = get_comfort_scores()
    
    vowels = 'aeiou'
    print(f"\nFinding best placements for vowels: {vowels}")
    print(f"in right-side positions: {RIGHT_POSITIONS}")
    print(f"Testing {len(list(permutations(RIGHT_POSITIONS, len(vowels))))} permutations...")
    
    vowel_placements = find_best_placements(
        vowels, RIGHT_POSITIONS, bigram_freqs, comfort_scores, top_n
    )
    
    # Print detailed results
    print("\nTop Vowel Placements:")
    for i, (score, positions, bigram_scores) in enumerate(vowel_placements, 1):
        print(f"\nPlacement {i}:")
        print_detailed_placement(score, positions, bigram_scores, vowels)
    
    return vowel_placements

#--------------------------------------------
# Display functions
#--------------------------------------------   
def print_keyboard_layout(positions: Dict[str, str], title: str = "Layout"):
    """Print a visual representation of the keyboard layout."""
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
        position_to_letter = dict(zip(positions, letters))
        for pos in RIGHT_POSITIONS:
            if pos in position_to_letter:
                pos_map[pos] = position_to_letter[pos]
    else:
        # Handle consonants (left side)
        position_to_letter = dict(zip(positions, letters))
        for pos in LEFT_POSITIONS:
            if pos in position_to_letter:
                pos_map[pos] = position_to_letter[pos]
    
    # Print visual keyboard layout
    print_keyboard_layout(pos_map, f"Score: {score:.4f}")
    
    # Print bigram contributions
    print("\nTop Bigram Contributions (sorted by impact):")
    sorted_bigrams = sorted(bigram_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    for bigram, contribution in sorted_bigrams[:10]:  # Show top 10 contributions
        if contribution != 0:
            print(f"  {bigram}: {contribution:>10.6f}")


if __name__ == "__main__":

    # Optimize consonant placement
    num_consonants = 5  # Try with most frequent consonants
    top_n = 1  # Return top placements

    # Optimize consonant placement
    consonant_placements = optimize_consonant_layout(num_consonants, top_n)
    
    # Optimize vowel placement
    vowel_placements = optimize_vowel_layout(top_n)




