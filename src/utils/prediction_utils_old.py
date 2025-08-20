"""
Alternative approach to handle sklearn feature name warnings during deployment.
This script provides utility functions that can be imported in your main app.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning, DataConversionWarning

# Suppress all sklearn warnings related to feature names and version mismatches
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

def predict_with_traditional_ml(model, features, model_path=""):
    """
    Make predictions with traditional ML models while avoiding sklearn warnings.
    
    Args:
        model: The loaded sklearn model
        features: Input features (list or array)
        model_path: Path to the model (used for feature naming)
    
    Returns:
        tuple: (prediction, confidence) where confidence is the max probability
    """
    try:
        # Convert features to numpy array
        if isinstance(features, list):
            if isinstance(features[0], list):
                feature_array = np.array(features)
            else:
                feature_array = np.array([features])
        else:
            feature_array = np.array(features)
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(1, -1)
        
        # Method 1: Try direct prediction with numpy array (sklearn should handle this)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                # First try with numpy array
                raw_prediction = model.predict(feature_array)[0]
                
                # Get confidence if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_array)[0]
                    confidence = max(probabilities)
                else:
                    confidence = None
                
                return raw_prediction, confidence
                
            except Exception as e1:
                # Method 2: Try with DataFrame and generic feature names
                try:
                    n_features = feature_array.shape[1]
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                    features_df = pd.DataFrame(feature_array, columns=feature_names)
                    
                    raw_prediction = model.predict(features_df)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_df)[0]
                        confidence = max(probabilities)
                    else:
                        confidence = None
                    
                    return raw_prediction, confidence
                    
                except Exception as e2:
                    # Method 3: Final fallback - try with simple array conversion
                    try:
                        # Convert to basic numpy array with explicit dtype
                        simple_array = np.asarray(features, dtype=np.float64).reshape(1, -1)
                        
                        raw_prediction = model.predict(simple_array)[0]
                        
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(simple_array)[0]
                            confidence = max(probabilities)
                        else:
                            confidence = None
                        
                        return raw_prediction, confidence
                        
                    except Exception as e3:
                        raise Exception(f"All prediction methods failed. Last error: {str(e3)}")
    
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def safe_model_predict(model, features, model_path=""):
    """
    Safe wrapper for model predictions that handles various edge cases.
    
    Returns:
        dict: Result dictionary with prediction, confidence, and success status
    """
    result = {
        'success': False,
        'prediction': None,
        'confidence': None,
        'error': None
    }
    
    try:
        prediction, confidence = predict_with_traditional_ml(model, features, model_path)
        
        result['success'] = True
        result['prediction'] = prediction
        result['confidence'] = confidence
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def convert_prediction_to_label(prediction):
    """Convert numerical prediction to meaningful label"""
    if isinstance(prediction, (int, float, np.integer, np.floating)):
        return "DNA Binding" if prediction == 1 or prediction > 0.5 else "Non-DNA Binding"
    else:
        return str(prediction)

# Export the main functions
__all__ = ['predict_with_traditional_ml', 'safe_model_predict', 'convert_prediction_to_label']
