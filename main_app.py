"""
DNA Binding Protein Classifier - Main Application
Production-ready Streamlit web application for DNA binding protein classification.
"""

import streamlit as st
import warnings
import joblib
import pandas as pd
import numpy as np
import re
import os
from io import StringIO

# Suppress warnings for production
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Page configuration
st.set_page_config(
    page_title="DNA Binding Protein Classifier",
    page_icon="🧬",
    layout="wide"
)

def calculate_amino_acid_composition(sequence):
    """Calculate amino acid composition features for a protein sequence"""
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    if len(sequence) == 0:
        return [0.0] * len(amino_acids)
    
    from collections import Counter
    aa_count = Counter(sequence)
    
    composition = []
    for aa in amino_acids:
        composition.append(aa_count.get(aa, 0) / len(sequence))
    
    return composition

def extract_physicochemical_properties(sequence):
    """Extract physicochemical properties from protein sequence"""
    hydrophobic = set('AILVFWYMC')
    hydrophilic = set('DEKRHNQST')
    aromatic = set('FWY')
    aliphatic = set('AILV')
    charged = set('DEKR')
    polar = set('STNQCYWH')
    
    sequence = sequence.upper()
    length = len(sequence)
    
    if length == 0:
        return [0.0] * 6
    
    properties = []
    properties.append(sum(1 for aa in sequence if aa in hydrophobic) / length)
    properties.append(sum(1 for aa in sequence if aa in hydrophilic) / length)
    properties.append(sum(1 for aa in sequence if aa in aromatic) / length)
    properties.append(sum(1 for aa in sequence if aa in aliphatic) / length)
    properties.append(sum(1 for aa in sequence if aa in charged) / length)
    properties.append(sum(1 for aa in sequence if aa in polar) / length)
    
    return properties

def calculate_pseaac_features(sequence, lambda_val=10):
    """Calculate simplified PseAAC features"""
    aa_composition = calculate_amino_acid_composition(sequence)
    
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                     'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                     'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                     'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
    
    correlation_factors = []
    for lag in range(1, min(lambda_val + 1, len(sequence))):
        if len(sequence) > lag:
            correlations = []
            for i in range(len(sequence) - lag):
                aa1, aa2 = sequence[i], sequence[i + lag]
                if aa1 in hydrophobicity and aa2 in hydrophobicity:
                    correlations.append((hydrophobicity[aa1] - hydrophobicity[aa2]) ** 2)
            
            if correlations:
                correlation_factors.append(np.mean(correlations))
            else:
                correlation_factors.append(0.0)
        else:
            correlation_factors.append(0.0)
    
    while len(correlation_factors) < lambda_val:
        correlation_factors.append(0.0)
    
    return aa_composition + correlation_factors[:lambda_val]

def sequence_to_cnn_input(sequence, max_length=1000):
    """Convert protein sequence to CNN input format"""
    aa_to_num = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    
    sequence = sequence.upper()
    sequence_nums = [aa_to_num.get(aa, 0) for aa in sequence]
    
    if len(sequence_nums) > max_length:
        sequence_nums = sequence_nums[:max_length]
    else:
        sequence_nums.extend([0] * (max_length - len(sequence_nums)))
    
    return np.array([sequence_nums])

def safe_load_model(model_path):
    """Safely load model with error handling and fallback for missing models"""
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    
    try:
        if model_path.endswith('.h5'):
            try:
                import tensorflow as tf
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = tf.keras.models.load_model(model_path, compile=False)
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model, None
            except ImportError:
                return None, "TensorFlow not available in this deployment. CNN models are not supported."
            except Exception as e:
                return None, f"Error loading TensorFlow model: {str(e)}"
        else:
            model = joblib.load(model_path)
            return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def make_prediction(model, sequence, model_type):
    """Make prediction with appropriate feature extraction"""
    try:
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
        
        if 'CNN' in model_type:
            try:
                features = sequence_to_cnn_input(sequence)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prediction_proba = model.predict(features, verbose=0)[0][0]
                    prediction = 1 if prediction_proba > 0.5 else 0
                    confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
            except Exception as e:
                return None, f"CNN prediction error: {str(e)}"
        else:
            if 'PseAAC' in model_type or 'pseAAC' in model_type:
                features = calculate_pseaac_features(sequence)
                amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
                feature_names = [f'AA_{aa}' for aa in amino_acids] + [f'Lambda_{i+1}' for i in range(10)]
            elif 'Physicochemical' in model_type:
                features = extract_physicochemical_properties(sequence)
                feature_names = ['Hydrophobic', 'Hydrophilic', 'Aromatic', 'Aliphatic', 'Charged', 'Polar']
            else:
                features = calculate_amino_acid_composition(sequence)
                amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
                feature_names = [f'AA_{aa}' for aa in amino_acids]
            
            # Create DataFrame with proper feature names
            feature_df = pd.DataFrame([features], columns=feature_names)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = model.predict(feature_df)[0]
                
                confidence = None
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(feature_df)[0]
                        confidence = max(proba)
                    except:
                        pass
        
        return prediction, confidence
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# Check TensorFlow availability
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

# Main Application
st.title("🧬 DNA Binding Protein Classifier")

st.info("""
🎯 **Available Models for DNA Binding Protein Classification:**
- **PseAAC Models**: Reliable amino acid composition and sequence order features
- **Physicochemical Models**: Protein physicochemical properties
- **CNN Models**: Deep learning models for sequence analysis

✅ **All models are production-ready with proper warning suppression!**
""")

if not TENSORFLOW_AVAILABLE:
    st.warning("""
    ⚠️ **TensorFlow not available in this deployment**
    CNN models are not supported. Traditional ML models (PseAAC and Physicochemical) are fully functional.
    """)

st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Selection")

model_categories = {
    "🧬 PseAAC Models (Recommended)": {
        "Random Forest": "models/Traditional ML - PseAAC/RF_pseAAC.joblib",
        "SVM": "models/Traditional ML - PseAAC/SVM_pseAAC.joblib",
        "Logistic Regression": "models/Traditional ML - PseAAC/LR_pseAAC.joblib",
        "Decision Tree": "models/Traditional ML - PseAAC/DT_pseAAC.joblib",
        "Naive Bayes": "models/Traditional ML - PseAAC/NB_pseAAC.joblib",
        "KNN": "models/Traditional ML - PseAAC/KNN_pseAAC.joblib"
    },
    "⚗️ Physicochemical Properties": {
        "Random Forest": "models/Traditional ML - Physicochemical Properties/RF_Physicochemical_Properties.joblib",
        "SVM": "models/Traditional ML - Physicochemical Properties/SVM_Physicochemical_Properties.joblib",
        "Logistic Regression": "models/Traditional ML - Physicochemical Properties/LR_Physicochemical_Properties.joblib",
        "Decision Tree": "models/Traditional ML - Physicochemical Properties/DT_Physicochemical_Properties.joblib",
        "Naive Bayes": "models/Traditional ML - Physicochemical Properties/NB_Physicochemical_Properties.joblib",
        "KNN": "models/Traditional ML - Physicochemical Properties/KNN_Physicochemical_Properties.joblib"
    }
}

# Add CNN models only if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    model_categories["🤖 CNN Models (Deep Learning)"] = {
        "CNN1": "models/CNN/CNN1.h5",
        "CNN2": "models/CNN/CNN2.h5",
        "ProtCNN1": "models/CNN/ProtCNN1.h5",
        "ProtCNN2": "models/CNN/ProtCNN2.h5"
    }

selected_category = st.sidebar.selectbox("Select Model Category", list(model_categories.keys()))
available_models = model_categories[selected_category]

# Filter available models based on what actually exists
existing_models = {}
for name, path in available_models.items():
    if os.path.exists(path):
        existing_models[name] = path
    else:
        existing_models[f"{name} (Not Available)"] = path

selected_model_name = st.sidebar.selectbox("Select Model", list(existing_models.keys()))
selected_model_path = existing_models[selected_model_name]

# Show availability status
if "Not Available" in selected_model_name:
    st.sidebar.warning("⚠️ Selected model not available in deployment")
else:
    st.sidebar.success("✅ Model available")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input Protein Sequence")
    
    input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
    
    sequences_to_classify = []
    
    if input_method == "Text Input":
        sequence_input = st.text_area(
            "Enter protein sequence(s) (one per line):",
            height=200,
            placeholder="MKVSQILPLAGAISVASGFWIPDFSNKQNSNSYPGQYKGKGGYQ..."
        )
        
        if sequence_input.strip():
            sequences = [seq.strip() for seq in sequence_input.split('\n') if seq.strip() and not seq.startswith('>')]
            for i, seq in enumerate(sequences):
                sequences_to_classify.append((f"Sequence_{i+1}", seq))
    
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
                            sequences_to_classify.append((current_header, current_seq))
                        current_header = line[1:]
                        current_seq = ""
                    else:
                        current_seq += line
                
                if current_seq and current_header:
                    sequences_to_classify.append((current_header, current_seq))
                
                st.success(f"Loaded {len(sequences_to_classify)} sequences from file")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

with col2:
    st.header("Classification Results")
    
    if sequences_to_classify:
        if st.button("🔬 Classify Sequences", type="primary"):
            model, error = safe_load_model(selected_model_path)
            
            if model is not None:
                results = []
                
                with st.progress(0) as progress_bar:
                    for i, (header, sequence) in enumerate(sequences_to_classify):
                        progress_bar.progress((i + 1) / len(sequences_to_classify))
                        
                        prediction, confidence = make_prediction(model, sequence, selected_model_name)
                        
                        if prediction is not None:
                            pred_text = "🧬 DNA Binding" if prediction == 1 else "🚫 Non-DNA Binding"
                            conf_text = f"{confidence:.4f}" if confidence is not None else "N/A"
                        else:
                            pred_text = "❌ Error"
                            conf_text = "N/A"
                        
                        results.append({
                            'Sequence ID': header,
                            'Length': len(sequence),
                            'Prediction': pred_text,
                            'Confidence': conf_text,
                            'Sequence Preview': sequence[:50] + "..." if len(sequence) > 50 else sequence
                        })
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary
                    st.subheader("Summary")
                    prediction_counts = results_df['Prediction'].value_counts()
                    st.bar_chart(prediction_counts)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"dna_classification_results_{selected_model_name}.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"Failed to load model: {error}")
    else:
        st.info("Please enter or upload sequences to classify")

# Information section
st.markdown("---")
st.header("ℹ️ Model Information")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Current Selection")
    st.write(f"**Model:** {selected_model_name}")
    st.write(f"**Category:** {selected_category}")
    
    if os.path.exists(selected_model_path):
        file_size = os.path.getsize(selected_model_path)
        st.write(f"**File Size:** {file_size / (1024*1024):.2f} MB")
        st.success("✅ Model available")
    else:
        st.error("❌ Model file not found")
        st.info("💡 This model is not included in the cloud deployment to reduce size. Available models: Random Forest (PseAAC), Random Forest (Physicochemical), CNN1")

with col4:
    st.subheader("Usage Instructions")
    st.write("""
    1. **Select Model**: Choose category and specific model
    2. **Input Sequences**: Enter text or upload FASTA file
    3. **Classify**: Click classify button for predictions
    4. **View Results**: See predictions and confidence scores
    5. **Download**: Export results as CSV
    """)

# Footer
st.markdown("---")
st.markdown("**DNA Binding Protein Classifier** - Production Ready Application")
