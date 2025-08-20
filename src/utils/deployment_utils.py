"""
Deployment utilities for production environment setup.
Handles TensorFlow warnings, Streamlit configuration, and environment setup.
"""

import os
import warnings
import logging

# Suppress TensorFlow warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging to reduce noise
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

def suppress_tensorflow_warnings():
    """Suppress TensorFlow and related warnings for production deployment"""
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=UserWarning, module='absl')
    warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')
    
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except ImportError:
        pass

def configure_streamlit_for_production():
    """Configure Streamlit settings to avoid file watcher issues in production"""
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_PORT'] = os.environ.get('PORT', '8501')
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'

def setup_production_environment():
    """Complete setup for production deployment"""
    # Suppress sklearn warnings
    from sklearn.exceptions import DataConversionWarning, InconsistentVersionWarning
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
    
    suppress_tensorflow_warnings()
    configure_streamlit_for_production()
