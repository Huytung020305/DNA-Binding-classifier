"""
Additional deployment fixes for TensorFlow and Streamlit issues.
This module provides solutions for common deployment problems.
"""

import os
import warnings
import logging

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings

# Configure logging to reduce noise
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

def suppress_tensorflow_warnings():
    """Suppress TensorFlow and related warnings for production deployment"""
    
    # Suppress TensorFlow warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
    
    # Suppress absl warnings (used by TensorFlow)
    warnings.filterwarnings('ignore', category=UserWarning, module='absl')
    warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')
    
    # Additional TensorFlow-specific suppressions
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
        # Suppress specific model compilation warnings
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
    except ImportError:
        pass  # TensorFlow not available

def load_keras_model_safely(model_path):
    """
    Load Keras model with proper warning suppression and error handling.
    
    Args:
        model_path (str): Path to the .h5 model file
        
    Returns:
        model: Loaded Keras model or None if failed
    """
    try:
        # Import TensorFlow with warning suppression
        import tensorflow as tf
        
        # Temporarily suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load model
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Compile the model to avoid the compilation warning
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
    except Exception as e:
        print(f"Error loading Keras model {model_path}: {str(e)}")
        return None

def configure_streamlit_for_production():
    """
    Configure Streamlit settings to avoid file watcher issues in production.
    This should be called before running the Streamlit app.
    """
    
    # Set environment variables to disable file watching
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Additional production-friendly settings
    os.environ['STREAMLIT_SERVER_PORT'] = os.environ.get('PORT', '8501')
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    
def create_streamlit_config():
    """
    Create a Streamlit config file to prevent file watcher issues.
    This creates a .streamlit/config.toml file with production settings.
    """
    
    # Create .streamlit directory if it doesn't exist
    config_dir = '.streamlit'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    # Streamlit configuration for production
    config_content = """
[server]
port = 8501
address = "0.0.0.0"
headless = true
fileWatcherType = "none"
runOnSave = false
gatherUsageStats = false
enableCORS = false

[browser]
gatherUsageStats = false

[global]
developmentMode = false

[logger]
level = "error"
"""
    
    config_path = os.path.join(config_dir, 'config.toml')
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    print(f"Created Streamlit config at {config_path}")

def setup_production_environment():
    """
    Complete setup for production deployment.
    Call this at the beginning of your app.py file.
    """
    
    # Suppress TensorFlow warnings
    suppress_tensorflow_warnings()
    
    # Configure Streamlit for production
    configure_streamlit_for_production()
    
    # Create config file
    try:
        create_streamlit_config()
    except Exception as e:
        print(f"Warning: Could not create Streamlit config: {e}")
    
    print("✅ Production environment configured successfully")

# Export main functions
__all__ = [
    'suppress_tensorflow_warnings',
    'load_keras_model_safely',
    'configure_streamlit_for_production',
    'create_streamlit_config',
    'setup_production_environment'
]
