"""
Analyze the precision of numerical values in a model file.
This script loads a model file, analyzes the precision of its numerical values,
and provides a detailed report on the number of decimal places, types, and distribution.

Precision was determined to be at least 16 decimal places.

>> python analyze_precision.py /Users/arno/Software/typing_preferences_to_comfort_scores/output/bigram_score_prediction_model.pkl

"""
import pickle
import numpy as np
import sys
import re

class ModuleRedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('bigram_typing_preferences_to_comfort_scores'):
            # Always create dynamic classes for the old module structure
            class DynamicClass:
                def __init__(self, *args, **kwargs):
                    for i, arg in enumerate(args):
                        setattr(self, f'_arg_{i}', arg)
                    for key, value in kwargs.items():
                        setattr(self, key, value)
                
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)
                
                def __getstate__(self):
                    return self.__dict__
            
            DynamicClass.__name__ = name
            DynamicClass.__qualname__ = name
            return DynamicClass
        
        return super().find_class(module, name)

def count_decimal_places(number):
    """Count the number of decimal places in a float"""
    if isinstance(number, (int, np.integer)):
        return 0
    
    # Convert to string and analyze
    str_num = f"{float(number):.17f}".rstrip('0')
    if '.' in str_num:
        return len(str_num.split('.')[1])
    return 0

def analyze_precision(obj, path="", max_samples=10):
    """Analyze the precision of numerical values in an object"""
    results = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            results.extend(analyze_precision(value, new_path, max_samples))
            
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj[:max_samples]):  # Limit samples for large lists
            new_path = f"{path}[{i}]"
            results.extend(analyze_precision(value, new_path, max_samples))
            
    elif isinstance(obj, np.ndarray):
        if obj.dtype in [np.float32, np.float64]:
            # Sample a few values from the array
            flat = obj.flatten()
            sample_size = min(max_samples, len(flat))
            for i in range(sample_size):
                decimal_places = count_decimal_places(flat[i])
                results.append({
                    'path': f"{path}[{i}]",
                    'value': float(flat[i]),
                    'decimal_places': decimal_places,
                    'type': str(obj.dtype)
                })
                
    elif isinstance(obj, (float, np.floating)):
        decimal_places = count_decimal_places(obj)
        results.append({
            'path': path,
            'value': float(obj),
            'decimal_places': decimal_places,
            'type': type(obj).__name__
        })
        
    elif isinstance(obj, tuple) and len(obj) == 2:
        # Check if it's a (mean, std) tuple
        try:
            val1, val2 = obj
            if isinstance(val1, (float, int, np.number)) and isinstance(val2, (float, int, np.number)):
                results.append({
                    'path': f"{path}[0]",
                    'value': float(val1),
                    'decimal_places': count_decimal_places(val1),
                    'type': type(val1).__name__
                })
                results.append({
                    'path': f"{path}[1]",
                    'value': float(val2),
                    'decimal_places': count_decimal_places(val2),
                    'type': type(val2).__name__
                })
        except:
            pass
    
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_precision.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print(f"Analyzing precision in: {model_path}")
    print("=" * 60)
    
    try:
        # Load the original model
        with open(model_path, 'rb') as f:
            unpickler = ModuleRedirectUnpickler(f)
            model = unpickler.load()
        
        print("Model loaded successfully!")
        
        # Analyze precision
        precision_data = analyze_precision(model)
        
        if not precision_data:
            print("No numerical data found to analyze.")
            return
        
        # Group by path prefix for better organization
        by_section = {}
        for item in precision_data:
            section = item['path'].split('.')[0] if '.' in item['path'] else item['path'].split('[')[0]
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(item)
        
        # Print summary by section
        print(f"\nFound {len(precision_data)} numerical values")
        print("\nPrecision Analysis by Section:")
        print("-" * 60)
        
        overall_decimal_places = []
        
        for section, items in by_section.items():
            decimal_places = [item['decimal_places'] for item in items]
            overall_decimal_places.extend(decimal_places)
            
            print(f"\n{section}:")
            print(f"  Count: {len(items)}")
            print(f"  Decimal places - Min: {min(decimal_places)}, Max: {max(decimal_places)}, Avg: {sum(decimal_places)/len(decimal_places):.1f}")
            
            # Show a few example values
            print("  Examples:")
            for item in items[:3]:
                print(f"    {item['path']}: {item['value']} ({item['decimal_places']} decimal places)")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more")
        
        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Total values analyzed: {len(overall_decimal_places)}")
        print(f"  Decimal places - Min: {min(overall_decimal_places)}, Max: {max(overall_decimal_places)}")
        print(f"  Average decimal places: {sum(overall_decimal_places)/len(overall_decimal_places):.2f}")
        print(f"  Most common precision: {max(set(overall_decimal_places), key=overall_decimal_places.count)} decimal places")
        
        # Precision distribution
        precision_counts = {}
        for dp in overall_decimal_places:
            precision_counts[dp] = precision_counts.get(dp, 0) + 1
        
        print(f"\nPrecision Distribution:")
        for precision in sorted(precision_counts.keys()):
            count = precision_counts[precision]
            percentage = (count / len(overall_decimal_places)) * 100
            print(f"  {precision} decimal places: {count} values ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()