"""
Lightweight DNA Binding Protein Classifier - Streamlit Cloud Version
Optimized for deployment with minimal model loading.
"""

import streamlit as st
import sys
import os
import warnings
import pandas as pd
import numpy as np
import re
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

@st.cache_resource
def load_lightweight_model(model_type):
    """Load only essential models for cloud deployment"""
    try:
        import joblib
        
        # Only load one representative model per type for cloud deployment
        model_paths = {
            'PseAAC': 'models/Traditional ML - PseAAC/RF_pseAAC.joblib',
            'Physicochemical': 'models/Traditional ML - Physicochemical Properties/RF_Physicochemical_Properties.joblib'
        }
        
        if model_type in model_paths and os.path.exists(model_paths[model_type]):
            model = joblib.load(model_paths[model_type])
            return model, None
        else:
            return None, f"Model {model_type} not available in lightweight deployment"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def make_prediction(model, sequence, model_type):
    """Make prediction with appropriate feature extraction"""
    try:
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
        
        if model_type == 'PseAAC':
            features = calculate_pseaac_features(sequence)
            amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            feature_names = [f'AA_{aa}' for aa in amino_acids] + [f'Lambda_{i+1}' for i in range(10)]
        elif model_type == 'Physicochemical':
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

# Main Application
st.title("🧬 DNA Binding Protein Classifier")
st.subheader("Lightweight Cloud Deployment")

st.info("""
🎯 **Available Models for DNA Binding Protein Classification:**
- **PseAAC Model**: Random Forest with amino acid composition and sequence order features
- **Physicochemical Model**: Random Forest with protein physicochemical properties

⚡ **Optimized for cloud deployment with essential models only!**
""")

st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Selection")

available_models = {
    "🧬 PseAAC (Recommended)": "PseAAC",
    "⚗️ Physicochemical Properties": "Physicochemical"
}

selected_display_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
selected_model_type = available_models[selected_display_name]

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
            with st.spinner("Loading model..."):
                model, error = load_lightweight_model(selected_model_type)
            
            if model is not None:
                results = []
                
                with st.progress(0) as progress_bar:
                    for i, (header, sequence) in enumerate(sequences_to_classify):
                        progress_bar.progress((i + 1) / len(sequences_to_classify))
                        
                        prediction, confidence = make_prediction(model, sequence, selected_model_type)
                        
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
                        file_name=f"dna_classification_results_{selected_model_type}.csv",
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
    st.write(f"**Model:** {selected_display_name}")
    st.write(f"**Type:** {selected_model_type}")
    st.write("**Algorithm:** Random Forest")
    
    # Check if model file exists
    model_paths = {
        'PseAAC': 'models/Traditional ML - PseAAC/RF_pseAAC.joblib',
        'Physicochemical': 'models/Traditional ML - Physicochemical Properties/RF_Physicochemical_Properties.joblib'
    }
    
    if selected_model_type in model_paths:
        model_path = model_paths[selected_model_type]
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            st.write(f"**File Size:** {file_size / (1024*1024):.2f} MB")
            st.success("✅ Model available")
        else:
            st.warning("⚠️ Model file not found")

with col4:
    st.subheader("Usage Instructions")
    st.write("""
    1. **Select Model**: Choose between PseAAC or Physicochemical
    2. **Input Sequences**: Enter text or upload FASTA file
    3. **Classify**: Click classify button for predictions
    4. **View Results**: See predictions and confidence scores
    5. **Download**: Export results as CSV
    """)

# Model Information
st.markdown("---")
st.header("📊 Model Details")

model_info = {
    "PseAAC": {
        "description": "Pseudo Amino Acid Composition model combining amino acid frequencies with sequence order effects",
        "features": "20 amino acid compositions + 10 sequence correlation factors",
        "best_for": "General protein classification with good balance of accuracy and interpretability"
    },
    "Physicochemical": {
        "description": "Model based on physicochemical properties of amino acids",
        "features": "6 properties: hydrophobic, hydrophilic, aromatic, aliphatic, charged, polar",
        "best_for": "Understanding protein properties and fast predictions"
    }
}

if selected_model_type in model_info:
    info = model_info[selected_model_type]
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Features:** {info['features']}")
    st.write(f"**Best for:** {info['best_for']}")

# Footer
st.markdown("---")
st.markdown("**DNA Binding Protein Classifier** - Lightweight Cloud Deployment")
st.caption("Optimized for Streamlit Cloud with essential models only")
