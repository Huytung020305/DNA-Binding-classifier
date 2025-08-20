"""
Prediction utilities for DNA binding protein classification.
Handles model loading, feature processing, and safe prediction execution.
"""

import os
import pandas as pd
import numpy as np
import warnings
import joblib
from .feature_utils import process_sequence_for_prediction, get_feature_names

def safe_load_traditional_model(model_path):
    """Safely load traditional ML models with proper error handling"""
    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def safe_load_cnn_model(model_path):
    """Safely load CNN models with TensorFlow warning suppression"""
    try:
        import tensorflow as tf
        
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Compile with basic settings to avoid warnings
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        return model, None
    except Exception as e:
        return None, f"Error loading CNN model: {str(e)}"

def prepare_features_for_model(sequence, model_type):
    """Prepare features based on model type"""
    if 'TF-IDF' in model_type:
        # For TF-IDF models, we need special handling
        # Return raw sequence for now - TF-IDF processing needs the trained vectorizer
        return sequence
    elif 'PseAAC' in model_type or 'pseAAC' in model_type:
        features = process_sequence_for_prediction(sequence, 'pseaac')
        feature_names = get_feature_names('pseaac')
    elif 'Physicochemical' in model_type:
        features = process_sequence_for_prediction(sequence, 'physicochemical')
        feature_names = get_feature_names('physicochemical')
    else:
        # Default to amino acid composition
        features = process_sequence_for_prediction(sequence, 'amino_acid')
        feature_names = get_feature_names('amino_acid')
    
    # Create DataFrame with proper feature names
    feature_df = pd.DataFrame([features], columns=feature_names)
    return feature_df

def predict_with_traditional_model(model, sequence, model_type):
    """Make predictions with traditional ML models"""
    try:
        if 'TF-IDF' in model_type:
            # For TF-IDF models, we need the vectorizer
            # This is a simplified approach - in production, vectorizer should be saved separately
            return None, "TF-IDF models require special preprocessing not implemented in this demo"
        
        # Prepare features
        features = prepare_features_for_model(sequence, model_type)
        
        # Make prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(features)[0]
            probability = None
            
            # Try to get probability if available
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(features)[0]
                    probability = max(proba)
                except:
                    pass
        
        return prediction, probability
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def sequence_to_cnn_input(sequence, max_length=1000):
    """Convert protein sequence to CNN input format"""
    # Amino acid to number mapping
    aa_to_num = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    
    # Convert sequence to numbers
    sequence = sequence.upper()
    sequence_nums = [aa_to_num.get(aa, 0) for aa in sequence]
    
    # Pad or truncate to max_length
    if len(sequence_nums) > max_length:
        sequence_nums = sequence_nums[:max_length]
    else:
        sequence_nums.extend([0] * (max_length - len(sequence_nums)))
    
    return np.array([sequence_nums])

def predict_with_cnn_model(model, sequence):
    """Make predictions with CNN models"""
    try:
        # Convert sequence to CNN input
        cnn_input = sequence_to_cnn_input(sequence)
        
        # Make prediction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction_proba = model.predict(cnn_input, verbose=0)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
        
        return prediction, prediction_proba
    except Exception as e:
        return None, f"CNN prediction error: {str(e)}"

def get_model_info():
    """Get information about available models"""
    model_info = {
        'Traditional ML - PseAAC': {
            'algorithms': ['Decision Tree', 'KNN', 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVM'],
            'feature_type': 'pseaac',
            'description': 'Models trained on Pseudo Amino Acid Composition features'
        },
        'Traditional ML - Physicochemical Properties': {
            'algorithms': ['Decision Tree', 'KNN', 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVM'],
            'feature_type': 'physicochemical',
            'description': 'Models trained on physicochemical properties of amino acids'
        },
        'Traditional ML - TF-IDF': {
            'algorithms': ['Decision Tree', 'KNN', 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'SVM'],
            'feature_type': 'tfidf',
            'description': 'Models trained on TF-IDF features of k-mers'
        },
        'CNN': {
            'algorithms': ['CNN1', 'CNN2', 'ProtCNN1', 'ProtCNN2'],
            'feature_type': 'sequence',
            'description': 'Convolutional Neural Networks for sequence classification'
        }
    }
    return model_info

def format_prediction_result(prediction, probability, model_name):
    """Format prediction results for display"""
    if prediction is None:
        return f"❌ Prediction failed for {model_name}"
    
    result_text = "🧬 DNA Binding" if prediction == 1 else "🚫 Non-DNA Binding"
    
    if probability is not None:
        confidence = probability * 100 if prediction == 1 else (1 - probability) * 100
        return f"{result_text} (Confidence: {confidence:.1f}%)"
    else:
        return result_text
