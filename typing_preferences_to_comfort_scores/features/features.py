# features/features.py
"""
Individual feature calculations for typing mechanics.
Implements binary [0,1] or normalized (0-1) metrics such as:
  - Same-finger usage
  - Row transitions
  - Angle between keys
Used by feature extraction system.
"""
from math import atan2, degrees

from typing_preferences_to_comfort_scores.features.keymaps import *
from typing_preferences_to_comfort_scores.features.bigram_frequencies import bigrams, bigram_frequencies_array
from typing_preferences_to_comfort_scores.utils.logging import LoggingManager
logger = LoggingManager.getLogger(__name__)

#---------------------#
# Angles between keys #
#---------------------#
def calculate_angle(pos1, pos2):
    """Calculate angle (0-90 degrees) between two positions"""
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    
    # Calculate smallest angle (0-90)
    angle = abs(degrees(atan2(dy, dx)))
    if angle > 90:
        angle = 180 - angle
        
    return round(angle, 3)

# Create the metrics map with both directions
angles = {}
letters = 'qwertyuiopasdfghjkl;zxcvbnm,./'

# Calculate angles between all possible keys, storing both directions
for c1 in letters:
    for c2 in letters:
        if c1 != c2:
            pos1 = staggered_position_map[c1]
            pos2 = staggered_position_map[c2]
            angle = calculate_angle(pos1, pos2)
            # Store both directions
            angles[(c1, c2)] = angle
            angles[(c2, c1)] = angle
    
#------------------------#
# Same/adjacent features #
#------------------------#
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

def same_key(char1, char2):
    """
    Check if both keys are the same.
      1: same key
      0: different keys
    """
    if char1 == char2:
        return 1
    else:
        return 0

def adjacent_finger(char1, char2, column_map, finger_map):
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

def adj_finger_diff_row(char1, char2, column_map, row_map, finger_map):
    """
    Check if adjacent fingers on the same hand type the two keys on different rows.
      1: adjacent fingers, different rows
      0: non-adjacent fingers and/or same row
    """
    if same_hand(char1, char2, column_map) == 1:
        if adjacent_finger(char1, char2, column_map, finger_map) == 1 and \
           rows_apart(char1, char2, column_map, row_map) > 0:
            return 1
        else:
            return 0
    else:
        return 0

def adj_finger_skip(char1, char2, column_map, row_map, finger_map):
    """
    Check if adjacent fingers on the same hand type skip the home row.
      1: yes
      0: no
    """
    if same_hand(char1, char2, column_map) == 1:
        if adjacent_finger(char1, char2, column_map, finger_map) == 1 and \
           rows_apart(char1, char2, column_map, row_map) == 2:
            return 1
        else:
            return 0
    else:
        return 0

#---------------------#
# Separation features #
#---------------------#
def rows_apart(char1, char2, column_map, row_map):
    """
    Measure how many rows apart the two characters are (typed by the same hand).
      0: same row
      1/2: 1 row apart
      2/2: 2 rows apart
    """
    if same_hand(char1, char2, column_map) == 1:
        return abs(row_map[char2] - row_map[char1]) / 2.0
    else:
        return 0

def skip_home(char1, char2, column_map, row_map):
    """
    Skip home row?
      1: yes
      0: no
    """
    if same_hand(char1, char2, column_map) == 1:
        if abs(row_map[char2] - row_map[char1]) == 2:
            return 1
        else:
            return 0
    else:
        return 0

def columns_apart(char1, char2, column_map):
    """
    Measure how many columns apart the two characters are (typed by the same hand).
      0: same column
      1/4: 1 column apart
      2/4: 2 columns apart
      3/4: 3 columns apart
      4/4: 4 columns apart
    """
    if same_hand(char1, char2, column_map) == 1: # and middle_columns(char1, char2, column_map) == 0:
        return abs(column_map[char2] - column_map[char1]) / 4.0
    else:
        return 0

def angle_apart(char1, char2, column_map, angles):
    """Measure angle between the two QWERTY characters' keys (typed by the same hand)."""
    # If same character, angle is 0
    if char1 == char2:
        return 0.0
        
    if same_hand(char1, char2, column_map) == 1: # and middle_columns(char1, char2, column_map) == 0:
        return angles[(char1, char2)] / 90.0
    else:
        return 0

#--------------------#
# Direction features #
#--------------------#
def outward_roll(char1, char2, column_map, finger_map):
    """
    Check if the keys were typed in an outward rolling direction with the same hand.
    outward:  right-to-left for the left hand, left-to-right for the right hand
    inward:   left-to-right for the left hand, right-to-left for the right hand
      1: outward
      0: not outward
    """
    if same_hand(char1, char2, column_map) == 1:
        if same_finger(char1, char2, column_map, finger_map) == 0:
            if finger_map[char1] < finger_map[char2]:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
def outward_roll_same_row(char1, char2, column_map, row_map, finger_map):
    """
    Check if the keys were typed with an outward roll on the same row with the same hand.
      1: yes
      0: no
    """
    if rows_apart(char1, char2, column_map, row_map) == 0:
        if same_finger(char1, char2, column_map, finger_map) == 0:
            if finger_map[char1] < finger_map[char2]:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
def outward_skip(char1, char2, column_map, finger_map):
    """
    Check if the keys were typed in an outward rolling direction 
    with the same hand and skipping the home row.
      1: yes
      0: no
    """
    if same_hand(char1, char2, column_map) == 1:
        if outward_roll(char1, char2, column_map, finger_map) == 1 and skip_home(char1, char2, column_map, row_map) == 1:
            return 1
        else:
            return 0
    else:
        return 0
    
#-------------------#
# Position features #
#-------------------#
def middle_column(char1, char2, column_map):
    """
    Check if finger1 types a key in a middle column of the keyboard.
      1: Yes
      0: No
    """
    if column_map[char1] in [5, 6] or column_map[char2] in [5, 6]:
        return 1
    else:
        return 0
        
def sum_engram_position_values(char1, char2, column_map, engram_position_values):
    """Sum engram_position_values for the two characters (typed by the same hand)."""
    if same_hand(char1, char2, column_map) == 1: # and middle_columns(char1, char2, column_map) == 0:
        return (engram_position_values[char1] + engram_position_values[char2]) / (2 * max_engram_position_value)
    else:
        return 0

def sum_row_position_values(char1, char2, column_map, row_position_values):
    """Sum row_position_values for the two characters (typed by the same hand)."""
    if same_hand(char1, char2, column_map) == 1: # and middle_columns(char1, char2, column_map) == 0:
        return (row_position_values[char1] + row_position_values[char2]) / (2 * max_row_position_value)
    else:
        return 0

#-----------------#
# Finger features #
#-----------------#
def sum_finger_values(char1, char2, finger_map):
    """Sum finger_map values for the two characters (typed by the same hand)."""
    if same_hand(char1, char2, column_map) == 1: # and middle_columns(char1, char2, column_map) == 0:
        return (finger_map[char1] + finger_map[char2]) / (2 * max_finger_map_value)
    else:
        return 0

def finger1or4_top_above(char1, char2, column_map, row_map):
    """
    Check if finger 1 or 4 are on the top row and above the other finger.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        finger1or4_top = ["q", "r", "t", "y", "u", "p"]
        if (char1 in finger1or4_top and row_map[char1] > row_map[char2]) or \
           (char2 in finger1or4_top and row_map[char2] > row_map[char1]):
            return 1
        else:
            return 0
    else:
        return 0

def finger2or3_bottom_below(char1, char2, column_map, row_map):
    """
    Check if finger 2 or 3 are on the bottom row and below the other finger.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        finger2or3_bottom = ["x", "c", ",", "."]
        if (char1 in finger2or3_bottom and row_map[char1] < row_map[char2]) or \
           (char2 in finger2or3_bottom and row_map[char2] < row_map[char1]):
            return 1
        else:
            return 0
    else:
        return 0

#---------------------------------#
# Qwerty bigram frequency feature #
#---------------------------------#
def qwerty_bigram_frequency(char1, char2, bigrams, bigram_frequencies_array):
    """
    Look up normalized frequency of a bigram from Norvig's analysis.
    Normalizes by dividing by the maximum frequency ("th" = 0.0356).
    
    Parameters:
    - char1: First character of bigram (case-insensitive)
    - char2: Second character of bigram (case-insensitive)
    - bigrams: List of bigrams ordered by frequency
    - bigram_frequencies_array: Array of corresponding frequency values
    
    Returns:
    - float: Normalized frequency of the bigram if found, 0.0 if not found
             (value between 0 and 1, where "th" = 1.0)
    """
    # Maximum frequency is the first value in the array (corresponds to "th")
    max_freq = bigram_frequencies_array[0]  # ~0.0356
    
    # Create bigram string and convert to lowercase
    bigram = (char1 + char2).lower()
    
    # Look up bigram index in list and normalize
    try:
        idx = bigrams.index(bigram)
        return float(bigram_frequencies_array[idx] / max_freq)
    except ValueError:
        return 0.0