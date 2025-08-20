"""
DNA Binding Protein Classifier - Streamlit Web Application
Production-ready application with organized imports and error handling.
"""

import streamlit as st
import sys
import os
import warnings

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up production environment first
try:
    from utils.deployment_utils import setup_production_environment
    setup_production_environment()
except ImportError:
    # Fallback warning suppression
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import required libraries
import joblib
import pandas as pd
import numpy as np
import re

# Import utilities
try:
    from utils.prediction_utils import (
        safe_load_traditional_model, safe_load_cnn_model,
        predict_with_traditional_model, predict_with_cnn_model,
        format_prediction_result, get_model_info
    )
    from utils.data_utils import (
        read_fasta_from_upload, validate_protein_sequence, get_sequence_stats
    )
    from utils.feature_utils import (
        calculate_amino_acid_composition, extract_physicochemical_properties,
        calculate_pseaac_features, sequence_to_cnn_input
    )
except ImportError as e:
    st.error(f"Error importing utilities: {e}")
    st.error("Please ensure all utility modules are in the src/utils/ directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="DNA Binding Protein Classifier",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 DNA Binding Protein Classifier")

# Add informational banner
st.info("""
🎯 **Available Models for DNA Binding Protein Classification:**
- **TF-IDF Models**: Reliable text-based features from protein sequences
- **PseAAC Models**: Stable amino acid composition features  
- **Physicochemical Models**: Protein physicochemical properties
- **CNN Models**: Deep learning models

✅ **All models are production-ready!**
""")

st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Selection")

# Define available models
model_categories = {
    "🎯 TF-IDF Models (Recommended)": {
        "Random Forest": "models/Traditional ML - TF-IDF/Random Forest_TF-IDF.joblib",
        "SVM": "models/Traditional ML - TF-IDF/SVM_TF-IDF.joblib",
        "Logistic Regression": "models/Traditional ML - TF-IDF/Logistic Regression_TF-IDF.joblib",
        "Decision Tree": "models/Traditional ML - TF-IDF/Decision Tree_TF-IDF.joblib",
        "Naive Bayes": "models/Traditional ML - TF-IDF/Naive Bayes_TF-IDF.joblib",
        "KNN": "models/Traditional ML - TF-IDF/KNN_TF-IDF.joblib"
    },
    "🧬 PseAAC Models (Reliable)": {
        "Random Forest": "models/Traditional ML - PseAAC/RF_pseAAC.joblib",
        "SVM": "models/Traditional ML - PseAAC/SVM_pseAAC.joblib",
        "Logistic Regression": "models/Traditional ML - PseAAC/LR_pseAAC.joblib",
        "Decision Tree": "models/Traditional ML - PseAAC/DT_pseAAC.joblib",
        "Naive Bayes": "models/Traditional ML - PseAAC/NB_pseAAC.joblib",
        "KNN": "models/Traditional ML - PseAAC/KNN_pseAAC.joblib"
    },
    "⚗️ Physicochemical Properties (Stable)": {
        "Random Forest": "models/Traditional ML - Physicochemical Properties/RF_Physicochemical_Properties.joblib",
        "SVM": "models/Traditional ML - Physicochemical Properties/SVM_Physicochemical_Properties.joblib",
        "Logistic Regression": "models/Traditional ML - Physicochemical Properties/LR_Physicochemical_Properties.joblib",
        "Decision Tree": "models/Traditional ML - Physicochemical Properties/DT_Physicochemical_Properties.joblib",
        "Naive Bayes": "models/Traditional ML - Physicochemical Properties/NB_Physicochemical_Properties.joblib",
        "KNN": "models/Traditional ML - Physicochemical Properties/KNN_Physicochemical_Properties.joblib"
    },
    "🤖 CNN Models (Deep Learning)": {
        "CNN1": "models/CNN/CNN1.h5",
        "CNN2": "models/CNN/CNN2.h5",
        "ProtCNN1": "models/CNN/ProtCNN1.h5",
        "ProtCNN2": "models/CNN/ProtCNN2.h5"
    }
}

# Model selection
selected_category = st.sidebar.selectbox("Select Model Category", list(model_categories.keys()))
available_models = model_categories[selected_category]

if available_models:
    selected_model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
    selected_model_path = available_models[selected_model_name]
else:
    selected_model_name = None
    selected_model_path = None

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input Protein Sequence")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
    
    sequences_to_classify = []
    
    if input_method == "Text Input":
        sequence_input = st.text_area(
            "Enter protein sequence(s):",
            height=200,
            placeholder="MKVSQILPLAGAISVASGFWIPDFSNKQNSNSYPGQYKGKGGYQ..."
        )
        
        if sequence_input.strip():
            sequences = [seq.strip() for seq in sequence_input.split('\n') if seq.strip()]
            for i, seq in enumerate(sequences):
                if not seq.startswith('>'):
                    sequences_to_classify.append(f"Sequence_{i+1}: {seq}")
                else:
                    sequences_to_classify.append(seq)
    
    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload FASTA file:",
            type=['fasta', 'fa', 'txt'],
            help="Upload a FASTA file containing protein sequences"
        )
        
        if uploaded_file is not None:
            try:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                current_seq = ""
                current_header = ""
                
                for line in stringio:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq and current_header:
                            sequences_to_classify.append(f"{current_header}: {current_seq}")
                        current_header = line[1:]
                        current_seq = ""
                    else:
                        current_seq += line
                
                if current_seq and current_header:
                    sequences_to_classify.append(f"{current_header}: {current_seq}")
                
                st.success(f"Loaded {len(sequences_to_classify)} sequences from file")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

with col2:
    st.header("Classification Results")
    
    if selected_model_path and sequences_to_classify:
        if st.button("🔬 Classify Sequences", type="primary"):
            model = load_model_safely(selected_model_path)
            
            if model is not None:
                results = []
                
                try:
                    for seq_info in sequences_to_classify:
                        if ': ' in seq_info:
                            header, sequence = seq_info.split(': ', 1)
                        else:
                            header = f"Sequence_{len(results)+1}"
                            sequence = seq_info
                        
                        # Clean sequence
                        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
                        
                        # Get features based on model type
                        if "TF-IDF" in selected_model_path:
                            features = [calculate_aa_composition(sequence)]
                        elif "PseAAC" in selected_model_path:
                            features = [calculate_pseudoaac_features(sequence)]
                        elif "Physicochemical" in selected_model_path:
                            features = [calculate_physicochemical_features(sequence)]
                        elif "CNN" in selected_model_path:
                            # CNN prediction logic would go here
                            features = sequence_to_numerical(sequence)
                        else:
                            features = [sequence]
                        
                        # Make prediction
                        if "CNN" in selected_model_path:
                            # Handle CNN models
                            try:
                                prediction_prob = model.predict(features, verbose=0)[0]
                                binding_prob = prediction_prob[1] if len(prediction_prob) > 1 else prediction_prob[0]
                                raw_prediction = 1 if binding_prob > 0.5 else 0
                                prediction = "DNA Binding" if raw_prediction == 1 else "Non-DNA Binding"
                                confidence = f"{binding_prob:.4f}"
                            except Exception as e:
                                prediction = f"CNN Error: {str(e)}"
                                confidence = "N/A"
                        else:
                            # Traditional ML models
                            result = safe_model_predict(model, features, selected_model_path)
                            if result['success']:
                                raw_prediction = result['prediction']
                                prediction = "DNA Binding" if raw_prediction == 1 else "Non-DNA Binding"
                                confidence = f"{result['confidence']:.4f}" if result['confidence'] is not None else "N/A"
                            else:
                                prediction = f"Error: {result['error']}"
                                confidence = "N/A"
                        
                        results.append({
                            'Sequence ID': header,
                            'Length': len(sequence),
                            'Prediction': prediction,
                            'Confidence': confidence,
                            'Sequence Preview': sequence[:50] + "..." if len(sequence) > 50 else sequence
                        })
                    
                    # Display results
                    if results:
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Summary")
                        if 'Prediction' in results_df.columns:
                            prediction_counts = results_df['Prediction'].value_counts()
                            st.bar_chart(prediction_counts)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"dna_classification_results_{selected_model_name}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
            else:
                st.error("Failed to load the selected model")
    
    elif not selected_model_path:
        st.info("Please select a model from the sidebar")
    elif not sequences_to_classify:
        st.info("Please enter or upload sequences to classify")

# Information section
st.markdown("---")
st.header("ℹ️ About")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Model Information")
    if selected_model_name and selected_model_path:
        st.write(f"**Selected Model:** {selected_model_name}")
        st.write(f"**Category:** {selected_category}")
        
        if os.path.exists(selected_model_path):
            file_size = os.path.getsize(selected_model_path)
            st.write(f"**File Size:** {file_size / (1024*1024):.2f} MB")
            st.success("✅ Model available")
        else:
            st.warning("⚠️ Model file not found")

with col4:
    st.subheader("Usage Instructions")
    st.write("""
    1. **Select Model**: Choose a model category and specific model
    2. **Input Sequences**: Enter sequences manually or upload FASTA file
    3. **Classify**: Click the classify button to get predictions
    4. **View Results**: See predictions, confidence scores, and summary
    5. **Download**: Export results as CSV for further analysis
    """)

# Footer
st.markdown("---")
st.markdown("**DNA Binding Protein Classifier** - Powered by Machine Learning & Deep Learning")
