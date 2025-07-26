#!/usr/bin/env python3
"""
Extend comfort scores for missing keys using median of same-finger adjacent-row bigrams.

This script:
1. Loads the existing comfort scores (24 keys)
2. Identifies "same finger, adjacent row" bigrams in the data
3. Computes median comfort score for these awkward movements
4. Extends the dataset with this median for bigrams involving:
   - Middle columns: t, g, b, y, h, n
   - Right of right pinkie: keys right of p, ;, /

Usage:
    python extend_comfort_scores.py
    python extend_comfort_scores.py --input-file custom_comfort.csv --output-file extended_comfort.csv
"""

import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

# QWERTY keyboard layout with (row, finger, hand) mapping
# From dvorak9_scorer.py
QWERTY_LAYOUT = {
    # Number row (row 0)
    '1': (0, 4, 'L'), '2': (0, 3, 'L'), '3': (0, 2, 'L'), '4': (0, 1, 'L'), '5': (0, 1, 'L'),
    '6': (0, 1, 'R'), '7': (0, 1, 'R'), '8': (0, 2, 'R'), '9': (0, 3, 'R'), '0': (0, 4, 'R'),
    
    # Top row (row 1)
    'Q': (1, 4, 'L'), 'W': (1, 3, 'L'), 'E': (1, 2, 'L'), 'R': (1, 1, 'L'), 'T': (1, 1, 'L'),
    'Y': (1, 1, 'R'), 'U': (1, 1, 'R'), 'I': (1, 2, 'R'), 'O': (1, 3, 'R'), 'P': (1, 4, 'R'),
    
    # Home row (row 2) 
    'A': (2, 4, 'L'), 'S': (2, 3, 'L'), 'D': (2, 2, 'L'), 'F': (2, 1, 'L'), 'G': (2, 1, 'L'),
    'H': (2, 1, 'R'), 'J': (2, 1, 'R'), 'K': (2, 2, 'R'), 'L': (2, 3, 'R'), ';': (2, 4, 'R'),
    
    # Bottom row (row 3)
    'Z': (3, 4, 'L'), 'X': (3, 3, 'L'), 'C': (3, 2, 'L'), 'V': (3, 1, 'L'), 'B': (3, 1, 'L'),
    'N': (3, 1, 'R'), 'M': (3, 1, 'R'), ',': (3, 2, 'R'), '.': (3, 3, 'R'), '/': (3, 4, 'R'),
    
    # Additional common keys
    "'": (2, 4, 'R'), '[': (1, 4, 'R'), ']': (1, 4, 'R'), '\\': (1, 4, 'R'),
    '-': (0, 4, 'R'), '=': (0, 4, 'R'),
}

# Define the 24 keys in the original comfort dataset
COMFORT_24_KEYS = set('qwerasdfzxcvuiopjkl;m,./')

# Define the 30 standard typing keys (24 + 6 missing)
STANDARD_30_KEYS = set('qwertyuiopasdfghjkl;zxcvbnm,./')

# Keys missing from the 24-key comfort dataset (within the 30 standard keys)
MISSING_FROM_COMFORT = STANDARD_30_KEYS - COMFORT_24_KEYS  # Should be: {t,g,y,h,b,n}

def get_key_info(key):
    """Get (row, finger, hand) for a key"""
    key = key.upper()
    return QWERTY_LAYOUT.get(key, None)

def is_same_finger_adjacent_row(bigram):
    """Check if bigram uses same finger on adjacent rows"""
    if len(bigram) != 2:
        return False
    
    key1, key2 = bigram[0], bigram[1]
    info1 = get_key_info(key1)
    info2 = get_key_info(key2)
    
    if not info1 or not info2:
        return False
    
    row1, finger1, hand1 = info1
    row2, finger2, hand2 = info2
    
    # Same finger and hand
    if finger1 != finger2 or hand1 != hand2:
        return False
    
    # Adjacent rows (difference of 1)
    if abs(row1 - row2) == 1:
        return True
    
    return False

def is_outside_24_keys(bigram):
    """Check if bigram has at least one key outside the 24-key comfort dataset (within 30 standard keys)"""
    return any(key.lower() in MISSING_FROM_COMFORT for key in bigram)

def generate_missing_bigrams():
    """Generate bigrams needed to complete coverage of 30 standard keys"""
    
    missing_bigrams = set()
    
    # Generate all possible 2-character combinations from the 30 standard keys
    for key1 in STANDARD_30_KEYS:
        for key2 in STANDARD_30_KEYS:
            bigram = key1 + key2
            # Include if at least one key is missing from the original 24-key dataset
            if any(key in MISSING_FROM_COMFORT for key in bigram):
                missing_bigrams.add(bigram)
    
    return sorted(missing_bigrams)

def analyze_comfort_data(df):
    """Analyze existing comfort data to find same-finger adjacent-row patterns"""
    
    print(f"Analyzing {len(df)} existing comfort scores...")
    
    # Find same-finger adjacent-row bigrams
    same_finger_adjacent = []
    same_finger_scores = []
    
    for _, row in df.iterrows():
        bigram = row['position_pair'].upper()
        
        if is_same_finger_adjacent_row(bigram):
            same_finger_adjacent.append(bigram)
            same_finger_scores.append(row['score'])
    
    print(f"Found {len(same_finger_adjacent)} same-finger adjacent-row bigrams:")
    
    # Show examples
    for i, (bigram, score) in enumerate(zip(same_finger_adjacent[:10], same_finger_scores[:10])):
        key1, key2 = bigram[0], bigram[1]
        info1 = get_key_info(key1)
        info2 = get_key_info(key2)
        
        if info1 and info2:
            hand = info1[2]
            finger = info1[1]
            finger_names = {1: 'index', 2: 'middle', 3: 'ring', 4: 'pinky'}
            finger_name = finger_names.get(finger, f'finger{finger}')
            
            print(f"  {i+1:2d}. '{bigram}': {score:6.3f} ({hand} {finger_name}, rows {info1[0]}→{info2[0]})")
    
    if len(same_finger_scores) > 10:
        print(f"  ... and {len(same_finger_scores) - 10} more")
    
    # Calculate statistics
    if same_finger_scores:
        median_score = np.median(same_finger_scores)
        mean_score = np.mean(same_finger_scores)
        std_score = np.std(same_finger_scores)
        
        print(f"\nSame-finger adjacent-row statistics:")
        print(f"  Count: {len(same_finger_scores)}")
        print(f"  Median: {median_score:.4f}")
        print(f"  Mean: {mean_score:.4f} ± {std_score:.4f}")
        print(f"  Range: {min(same_finger_scores):.3f} to {max(same_finger_scores):.3f}")
        
        return median_score, same_finger_adjacent, same_finger_scores
    else:
        print("⚠️ No same-finger adjacent-row bigrams found!")
        return None, [], []

def extend_comfort_dataset(df, median_score):
    """Extend comfort dataset with missing bigrams using median score"""
    
    print(f"\nExtending comfort dataset...")
    print(f"Using median comfort score: {median_score:.4f}")
    
    # Generate missing bigrams
    missing_bigrams = generate_missing_bigrams()
    
    # Filter to only those not already in the dataset
    existing_bigrams = set(df['position_pair'].str.lower())
    truly_missing = [bg for bg in missing_bigrams if bg not in existing_bigrams]
    
    print(f"Generated {len(missing_bigrams)} potential missing bigrams")
    print(f"Actually missing from dataset: {len(truly_missing)}")
    
    # Create extension records
    extension_records = []
    
    for bigram in truly_missing:
        extension_records.append({
            'position_pair': bigram.upper(),
            'score': median_score,
            'uncertainty': 0.0,  # Synthetic data has no uncertainty
            'source': 'extended_from_median'
        })
    
    # Create extended dataframe
    if extension_records:
        extension_df = pd.DataFrame(extension_records)
        
        # Add source column to original data
        df_with_source = df.copy()
        df_with_source['source'] = 'original_data'
        
        # Combine datasets
        extended_df = pd.concat([df_with_source, extension_df], ignore_index=True)
        
        print(f"Extended dataset: {len(df)} original + {len(extension_records)} synthetic = {len(extended_df)} total")
        
        # Show some examples of extensions
        print(f"\nExample extensions:")
        examples = extension_records[:10]
        for i, record in enumerate(examples):
            bigram = record['position_pair'].lower()
            missing_keys = [c for c in bigram if c in MISSING_FROM_COMFORT]
            
            if missing_keys:
                reason_str = f"includes missing key(s): {', '.join(missing_keys)}"
            else:
                reason_str = "completes 30-key coverage"
                
            print(f"  {i+1:2d}. '{bigram}': {median_score:.4f} ({reason_str})")
        
        if len(extension_records) > 10:
            print(f"  ... and {len(extension_records) - 10} more extensions")
        
        return extended_df
    else:
        print("No extensions needed - all bigrams already covered")
        return df

def main():
    parser = argparse.ArgumentParser(description='Extend comfort scores using same-finger adjacent-row median')
    parser.add_argument('--input-file', default='output/data/estimated_bigram_scores.csv',
                       help='Input comfort scores CSV file')
    parser.add_argument('--output-file', default='output/data/estimated_bigram_scores_extended.csv',
                       help='Output extended comfort scores CSV file')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Only analyze existing data, do not extend')
    
    args = parser.parse_args()
    
    print("Extending Comfort Scores for Missing Keys")
    print("=" * 50)
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print()
    
    # Load existing comfort data
    try:
        df = pd.read_csv(args.input_file)
        print(f"✅ Loaded {len(df)} comfort scores")
        print(f"   Columns: {list(df.columns)}")
        
        # Convert input format to expected format
        if 'bigram' in df.columns and 'comfort_score' in df.columns:
            print("Converting input format to output format...")
            df = df.rename(columns={
                'bigram': 'position_pair',
                'comfort_score': 'score'
            })
            # Convert position_pair to uppercase
            df['position_pair'] = df['position_pair'].str.upper()
            print(f"   Converted bigrams to uppercase position pairs")
        
        # Verify required columns exist
        required_cols = ['position_pair', 'score', 'uncertainty']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return
            
        print(f"   Final columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error loading comfort data: {e}")
        return
    
    # Analyze existing data
    median_score, same_finger_bigrams, same_finger_scores = analyze_comfort_data(df)
    
    if median_score is None:
        print("❌ Cannot extend dataset - no same-finger adjacent-row reference data found")
        return
    
    if args.analysis_only:
        print(f"\n✅ Analysis complete (analysis-only mode)")
        return
    
    # Extend dataset
    extended_df = extend_comfort_dataset(df, median_score)
    
    # Save extended dataset
    try:
        extended_df.to_csv(args.output_file, index=False)
        print(f"\n✅ Extended comfort dataset saved to: {args.output_file}")
        
        # Summary statistics
        original_count = len(df)
        extended_count = len(extended_df)
        added_count = extended_count - original_count
        
        print(f"\n📊 Summary:")
        print(f"   Original records: {original_count}")
        print(f"   Added records: {added_count}")
        print(f"   Total records: {extended_count}")
        print(f"   Extension median score: {median_score:.4f}")
        
    except Exception as e:
        print(f"❌ Error saving extended dataset: {e}")

if __name__ == "__main__":
    main()