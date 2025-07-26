"""
Quantize and compress a pickle model file with custom handling for specific data structures.
This script loads a model, quantizes numerical data, removes large objects, and saves the compressed model.
It also handles dynamic class creation for compatibility with older module names.  

Model precision was determined to be at least 16 decimal places by analyze_model_precision.py.

>> poetry run python3 quantize_compress_model.py --decimals 9 /Users/arno/Software/typing_preferences_to_comfort_scores/output/data/bigram_score_prediction_model.pkl
"""
import pickle
import gzip
import numpy as np
import os
import sys
import argparse

class ModuleRedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('bigram_typing_preferences_to_comfort_scores'):
            print(f"Creating dynamic class: {name}")
            
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

def convert_dynamic_classes_to_dicts(obj):
    """Convert dynamic classes to dictionaries so they can be pickled"""
    if hasattr(obj, '__dict__') and hasattr(obj, '__class__') and hasattr(obj.__class__, '__name__'):
        # This is likely a dynamic class instance
        class_name = obj.__class__.__name__
        if class_name in ['Config', 'PathsConfig', 'ModelSettings', 'FeatureSelectionSettings', 
                         'FeaturesConfig', 'DataConfig', 'RecommendationsConfig', 
                         'LoggingConfig', 'VisualizationConfig', 'FeatureExtractor', 'FeatureConfig']:
            print(f"Converting {class_name} to dict")
            # Convert to dict and add a marker so we know what it was
            result = {'__class_name__': class_name}
            result.update(obj.__dict__)
            # Recursively process nested objects
            for key, value in result.items():
                if key != '__class_name__':
                    result[key] = convert_dynamic_classes_to_dicts(value)
            return result
    
    if isinstance(obj, dict):
        return {key: convert_dynamic_classes_to_dicts(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_dynamic_classes_to_dicts(item) for item in obj)
    else:
        return obj

def quantize_specific_data(model, decimals=4):
    """Quantize only specific known data structures without recursion"""
    
    if not isinstance(model, dict):
        return model
    
    print("Looking for quantizable data...")
    
    # Handle feature_weights specifically
    if 'feature_weights' in model:
        print("Quantizing feature_weights")
        fw = model['feature_weights']
        if isinstance(fw, dict):
            for key, value in fw.items():
                if isinstance(value, tuple) and len(value) == 2:
                    # (mean, std) tuple
                    model['feature_weights'][key] = (
                        round(float(value[0]), decimals), 
                        round(float(value[1]), decimals)
                    )
                elif isinstance(value, (int, float)):
                    model['feature_weights'][key] = round(float(value), decimals)
    
    # Handle weights specifically  
    if 'weights' in model:
        print("Quantizing weights")
        weights = model['weights']
        if isinstance(weights, dict):
            for key, value in weights.items():
                if isinstance(value, tuple) and len(value) == 2:
                    model['weights'][key] = (
                        round(float(value[0]), decimals), 
                        round(float(value[1]), decimals)
                    )
                elif isinstance(value, (int, float)):
                    model['weights'][key] = round(float(value), decimals)
        elif isinstance(weights, np.ndarray):
            if weights.dtype in [np.float32, np.float64]:
                model['weights'] = np.around(weights, decimals=decimals)
    
    # Look for any numpy arrays in top-level dict
    for key, value in model.items():
        if isinstance(value, np.ndarray) and value.dtype in [np.float32, np.float64]:
            print(f"Quantizing numpy array: {key}")
            model[key] = np.around(value, decimals=decimals)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Quantize and compress a pickle model file')
    parser.add_argument('model_path', help='Path to the pickle model file')
    parser.add_argument('--decimals', type=int, default=4, help='Number of decimal places for quantization (default: 4)')
    parser.add_argument('--output', '-o', help='Output path (default: quantized_<original_name>.pkl.gz)')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: File not found: {model_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        compressed_path = args.output
    else:
        model_dir = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)
        model_name = os.path.splitext(model_basename)[0]
        compressed_path = os.path.join(model_dir, f"quantized_{model_name}.pkl.gz")

    try:
        # Load using the custom unpickler
        print("Loading model...")
        with open(model_path, 'rb') as f:
            unpickler = ModuleRedirectUnpickler(f)
            model = unpickler.load()
        
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Show model structure
        if isinstance(model, dict):
            print(f"\nModel has {len(model)} top-level keys:")
            for key in model.keys():
                val = model[key]
                size_info = ""
                if hasattr(val, 'shape'):
                    size_info = f" (shape: {val.shape})"
                elif isinstance(val, dict):
                    size_info = f" ({len(val)} items)"
                elif isinstance(val, (list, tuple)):
                    size_info = f" ({len(val)} items)"
                
                print(f"  {key}: {type(val).__name__}{size_info}")
        
        # Remove large objects first
        if isinstance(model, dict):
            keys_to_remove = []
            for key, val in model.items():
                # Remove known large/unnecessary objects
                if (key == 'fit_result' or 
                    (hasattr(val, '__class__') and 'CmdStanMCMC' in str(val.__class__)) or
                    key == 'dataset'):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                print(f"Removing large object: {key}")
                del model[key]
        
        # Apply safe quantization
        print("\nQuantizing numerical data...")
        model = quantize_specific_data(model, decimals=args.decimals)
        
        # Convert dynamic classes to dictionaries for pickling
        print("Converting dynamic classes to dictionaries...")
        model = convert_dynamic_classes_to_dicts(model)
        
        # Save compressed model
        print(f"Saving compressed model to: {compressed_path}")
        with gzip.open(compressed_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Show results
        original_size = os.path.getsize(model_path)
        compressed_size = os.path.getsize(compressed_path)
        print(f"\nResults:")
        print(f"Original size: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
        print(f"Compressed size: {compressed_size:,} bytes ({compressed_size/1024/1024:.1f} MB)")
        print(f"Size reduction: {(1 - compressed_size/original_size)*100:.1f}%")
        
        print(f"\nSuccess! Quantized model saved to: {compressed_path}")
        print("Note: Dynamic class instances have been converted to dictionaries for compatibility.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()