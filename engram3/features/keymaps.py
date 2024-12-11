# features/keymaps.py
"""
Keyboard layout definitions and mapping utilities.
Contains physical layout information including:
  - Key positions
  - Finger assignments
  - Row designations
Used for feature calculations.
"""

#-------------------------------------#
# Keyboard layout and finger mappings #
#-------------------------------------#
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

matrix_position_map = {
    'q': (3, 1), 'w': (3, 2), 'e': (3, 3), 'r': (3, 4), 't': (3, 5),
    'a': (2, 1), 's': (2, 2), 'd': (2, 3), 'f': (2, 4), 'g': (2, 5),
    'z': (1, 1), 'x': (1, 2), 'c': (1, 3), 'v': (1, 4), 'b': (1, 5),
    'y': (3, 6), 'u': (3, 7), 'i': (3, 8), 'o': (3, 9), 'p': (3, 10),
    'h': (2, 6), 'j': (2, 7), 'k': (2, 8), 'l': (2, 9), ';': (2, 10),
    'n': (1, 6), 'm': (1, 7), ',': (1, 8), '.': (1, 9), '/': (1, 10)
}

# Positions of keys on a staggered keyboard with 19mm interkey spacing, 
# and 5mm and 10mm stagger between top/home and home/bottom rows
staggered_position_map = {
    # Top row (no stagger reference point)
    'q': (0, 0),    'w': (19, 0),   'e': (38, 0),   'r': (57, 0),   't': (76, 0),
    # Home row (staggered 5mm right from top row)
    'a': (5, 19),   's': (24, 19),  'd': (43, 19),  'f': (62, 19),  'g': (81, 19),
    # Bottom row (staggered 10mm right from home row)
    'z': (15, 38),  'x': (34, 38),  'c': (53, 38),  'v': (72, 38),  'b': (91, 38),
    # Top row continued
    'y': (95, 0),   'u': (114, 0),  'i': (133, 0),  'o': (152, 0),  'p': (171, 0),
    # Home row continued
    'h': (100, 19), 'j': (119, 19), 'k': (138, 19), 'l': (157, 19), ';': (176, 19),
    # Bottom row continued
    'n': (110, 38), 'm': (129, 38), ',': (148, 38), '.': (167, 38), '/': (186, 38)
}

#-----------------------------------#
# Assigned keyboard position values #
#-----------------------------------#
# original Engram position values, penalizing fingers 1 & 4 up, 2 & 3 down, and middle columns:
engram_position_values = {
    'q': 1, 'w': 0, 'e': 0, 'r': 1, 't': 2,
    'a': 0, 's': 0, 'd': 0, 'f': 0, 'g': 2,
    'z': 0, 'x': 1, 'c': 1, 'v': 0, 'b': 2,
    'y': 2, 'u': 1, 'i': 0, 'o': 0, 'p': 1,
    'h': 2, 'j': 0, 'k': 0, 'l': 0, ';': 0, 
    'n': 2, 'm': 0, ',': 1, '.': 1, '/': 0
}

# position values determined by row (home=0, top=1, bottom=2), without penalizing middle columns:
row_position_values = {
    'q': 1, 'w': 1, 'e': 1, 'r': 1, 't': 1,
    'a': 0, 's': 0, 'd': 0, 'f': 0, 'g': 0,
    'z': 2, 'x': 2, 'c': 2, 'v': 2, 'b': 2,
    'y': 1, 'u': 1, 'i': 1, 'o': 1, 'p': 1,
    'h': 0, 'j': 0, 'k': 0, 'l': 0, ';': 0, 
    'n': 2, 'm': 2, ',': 2, '.': 2, '/': 2
}

