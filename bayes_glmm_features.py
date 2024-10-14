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

def finger1skip2(char1, char2, column_map, row_map, finger_map):
    """
    Check if adjacent fingers 1 & 2 on the same hand type two keys skipping the home row.
      1: yes
      0: no
    """
    if same_hand(char1, char2, column_map) == 1:
        if skip_home(char1, char2, column_map, row_map) == 1:
            if (finger_map[char1] == 1 and finger_map[char2] == 2) or \
               (finger_map[char1] == 2 and finger_map[char2] == 1):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger2skip3(char1, char2, column_map, row_map, finger_map):
    """
    Check if adjacent fingers 2 & 3 on the same hand type two keys skipping the home row.
      1: yes
      0: no
    """
    if same_hand(char1, char2, column_map) == 1:
        if skip_home(char1, char2, column_map, row_map) == 1:
            if (finger_map[char1] == 2 and finger_map[char2] == 3) or \
               (finger_map[char1] == 3 and finger_map[char2] == 2):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger3skip4(char1, char2, column_map, row_map, finger_map):
    """
    Check if adjacent fingers 3 & 4 on the same hand type two keys skipping the home row.
      1: yes
      0: no
    """
    if same_hand(char1, char2, column_map) == 1:
        if skip_home(char1, char2, column_map, row_map) == 1:
            if (finger_map[char1] == 3 and finger_map[char2] == 4) or \
               (finger_map[char1] == 4 and finger_map[char2] == 3):
                return 1
            else:
                return 0
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

def finger2_above(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger2 types a key on a row above the other key typed by another finger on the same hand.
      1: finger2 above
      0: finger2 not above
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger2(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
            if (finger_map[char1] == 2 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 2 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger3_above(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger3 types a key on a row above the other key typed by another finger on the same hand.
      1: finger3 above
      0: finger3 not above
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger3(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
            if (finger_map[char1] == 3 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 3 and row_map[char2] > row_map[char1]):
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
        if finger4(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
            if (finger_map[char1] == 4 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 4 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger1_below(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger1 types a key on a row below the other key typed by another finger on the same hand.
      1: finger1 below
      0: finger1 not below
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger1(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
            if (finger_map[char1] == 1 and row_map[char1] < row_map[char2]) or \
               (finger_map[char2] == 1 and row_map[char2] < row_map[char1]):
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
        if finger2(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
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
        if finger3(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
            if (finger_map[char1] == 3 and row_map[char1] < row_map[char2]) or \
               (finger_map[char2] == 3 and row_map[char2] < row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger4_below(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger4 types a key on a row below the other key typed by another finger on the same hand.
      1: finger4 below
      0: finger4 not below
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger4(char1, char2, finger_map) == 1 and same_finger(char1, char2, column_map, finger_map) == 0:
            if (finger_map[char1] == 4 and row_map[char1] < row_map[char2]) or \
               (finger_map[char2] == 4 and row_map[char2] < row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger1above2(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger1 types a key on a row above finger2 on the same hand.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger1(char1, char2, finger_map) == 1 and finger2(char1, char2, finger_map) == 1:
            if (finger_map[char1] == 1 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 1 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger2above1(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger2 types a key on a row above finger1 on the same hand.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger1(char1, char2, finger_map) == 1 and finger2(char1, char2, finger_map) == 1:
            if (finger_map[char1] == 2 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 2 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger2above3(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger2 types a key on a row above finger3 on the same hand.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger2(char1, char2, finger_map) == 1 and finger3(char1, char2, finger_map) == 1:
            if (finger_map[char1] == 2 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 2 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger3above2(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger3 types a key on a row above finger2 on the same hand.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger2(char1, char2, finger_map) == 1 and finger3(char1, char2, finger_map) == 1:
            if (finger_map[char1] == 3 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 3 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger3above4(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger3 types a key on a row above finger4 on the same hand.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger3(char1, char2, finger_map) == 1 and finger4(char1, char2, finger_map) == 1:
            if (finger_map[char1] == 3 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 3 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def finger4above3(char1, char2, column_map, row_map, finger_map):
    """
    Check if finger4 types a key on a row above finger3 on the same hand.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        if finger3(char1, char2, finger_map) == 1 and finger4(char1, char2, finger_map) == 1:
            if (finger_map[char1] == 4 and row_map[char1] > row_map[char2]) or \
               (finger_map[char2] == 4 and row_map[char2] > row_map[char1]):
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0
    
def finger2or3down(char1, char2, column_map):
    """
    Check if finger 2 or 3 are on the bottom row.
      1: yes
      0: no 
    """
    if same_hand(char1, char2, column_map) == 1:
        finger2or3_bottom = ["x", "c", ",", "."]
        if char1 in finger2or3_bottom or char2 in finger2or3_bottom:
            return 1
        else:
            return 0
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