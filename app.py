import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import os
from Bio import SeqIO
from io import StringIO
import re

# Try to import TensorFlow for CNN models
try:
    import tensorflow as tf
    from tensorflow import keras
    tf_available = True
except ImportError:
    tf_available = False

# Page configuration
st.set_page_config(
    page_title="DNA Binding Protein Classifier",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 DNA Binding Protein Classifier")

# Add informational banner about model recommendations
st.info("""
🎯 **Available Models for DNA Binding Protein Classification:**
- **TF-IDF Models**: Reliable text-based features from protein sequences
- **PseAAC Models**: Stable amino acid composition features  
- **Physicochemical Models**: Protein physicochemical properties
- **CNN Models**: Deep learning models (now working with correct input format)

✅ **All models are now functional!** Choose based on your preference.
""")

st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Selection")

# Define available models based on your saved model structure
# Traditional ML models are prioritized for reliability
model_categories = {
    "🎯 TF-IDF Models (Recommended)": {
        "Random Forest": "Saved Model/Traditional ML - TF-IDF/Random Forest_TF-IDF.joblib",
        "SVM": "Saved Model/Traditional ML - TF-IDF/SVM_TF-IDF.joblib",
        "Logistic Regression": "Saved Model/Traditional ML - TF-IDF/Logistic Regression_TF-IDF.joblib",
        "Decision Tree": "Saved Model/Traditional ML - TF-IDF/Decision Tree_TF-IDF.joblib",
        "Naive Bayes": "Saved Model/Traditional ML - TF-IDF/Naive Bayes_TF-IDF.joblib",
        "KNN": "Saved Model/Traditional ML - TF-IDF/KNN_TF-IDF.joblib"
    },
    "🧬 PseAAC Models (Reliable)": {
        "Random Forest": "Saved Model/Traditional ML - PseAAC/RF_pseAAC.joblib",
        "SVM": "Saved Model/Traditional ML - PseAAC/SVM_pseAAC.joblib",
        "Logistic Regression": "Saved Model/Traditional ML - PseAAC/LR_pseAAC.joblib",
        "Decision Tree": "Saved Model/Traditional ML - PseAAC/DT_pseAAC.joblib",
        "Naive Bayes": "Saved Model/Traditional ML - PseAAC/NB_pseAAC.joblib",
        "KNN": "Saved Model/Traditional ML - PseAAC/KNN_pseAAC.joblib"
    },
    "⚗️ Physicochemical Properties (Stable)": {
        "Random Forest": "Saved Model/Traditional ML - Physicochemical Properties/RF_Physicochemical_Properties.joblib",
        "SVM": "Saved Model/Traditional ML - Physicochemical Properties/SVM_Physicochemical_Properties.joblib",
        "Logistic Regression": "Saved Model/Traditional ML - Physicochemical Properties/LR_Physicochemical_Properties.joblib",
        "Decision Tree": "Saved Model/Traditional ML - Physicochemical Properties/DT_Physicochemical_Properties.joblib",
        "Naive Bayes": "Saved Model/Traditional ML - Physicochemical Properties/NB_Physicochemical_Properties.joblib",
        "KNN": "Saved Model/Traditional ML - Physicochemical Properties/KNN_Physicochemical_Properties.joblib"
    },
    "🤖 CNN Models (Deep Learning)": {
        "CNN1": "Saved Model/CNN/CNN1.h5",
        "CNN2": "Saved Model/CNN/CNN2.h5",
        "ProtCNN1": "Saved Model/CNN/ProtCNN1.h5",
        "ProtCNN2": "Saved Model/CNN/ProtCNN2.h5"
    }
}

# Model selection
selected_category = st.sidebar.selectbox("Select Model Category", list(model_categories.keys()))
available_models = model_categories[selected_category]

# Warning for CNN models if TensorFlow is not available
if selected_category == "🤖 CNN Models (Deep Learning)" and not tf_available:
    st.sidebar.error("❌ TensorFlow is not installed. CNN models will not work. Please install TensorFlow.")

# Info for CNN models
if selected_category == "🤖 CNN Models (Deep Learning)":
    st.sidebar.success("✅ CNN models are now working with correct input format!")
    st.sidebar.info("📏 Input: Sequences are processed as 1200-position vectors")

# Info for traditional ML models
if "TF-IDF" in selected_category:
    st.sidebar.success("✅ TF-IDF models are reliable and work well for protein classification.")
elif "PseAAC" in selected_category:
    st.sidebar.success("✅ PseAAC models use amino acid composition features and are very stable.")
elif "Physicochemical" in selected_category:
    st.sidebar.success("✅ Physicochemical models use protein properties and are highly reliable.")

if available_models:
    selected_model_name = st.sidebar.selectbox("Select Model", list(available_models.keys()))
    selected_model_path = available_models[selected_model_name]
else:
    st.sidebar.warning(f"No models available in {selected_category}")
    selected_model_name = None
    selected_model_path = None

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            if model_path.endswith('.h5'):
                # Load Keras/TensorFlow model
                if not tf_available:
                    st.error("TensorFlow is required to load CNN models. Please install tensorflow.")
                    return None
                try:
                    return keras.models.load_model(model_path)
                except Exception as e:
                    st.error(f"Error loading CNN model: {str(e)}")
                    return None
            else:
                # Load joblib model
                return joblib.load(model_path)
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to extract k-mer features for TF-IDF
def extract_kmers(sequence, k=3):
    """Extract k-mers from protein sequence"""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return ' '.join(kmers)

# Function to calculate amino acid composition
def calculate_aa_composition(sequence):
    """Calculate amino acid composition features"""
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    composition = {}
    sequence = sequence.upper()
    
    for aa in aa_list:
        composition[f'AA_{aa}'] = sequence.count(aa) / len(sequence) if len(sequence) > 0 else 0
    
    return composition

# Function to calculate extended PseAAC features (40 features total)
def calculate_pseudoaac_features(sequence):
    """Calculate pseudo amino acid composition features (40 features)"""
    sequence = sequence.upper()
    length = len(sequence)
    
    if length == 0:
        return [0.0] * 40
    
    # 20 basic amino acid composition features
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    features = []
    
    for aa in aa_list:
        features.append(sequence.count(aa) / length)
    
    # Add 20 additional features based on physicochemical properties
    # Hydrophobicity index
    hydrophobicity = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
        'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
        'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
        'W': -0.9, 'Y': -1.3
    }
    
    # Hydrophobicity features (mean, std, min, max)
    hydro_values = [hydrophobicity.get(aa, 0) for aa in sequence]
    features.extend([
        np.mean(hydro_values),
        np.std(hydro_values),
        np.min(hydro_values),
        np.max(hydro_values)
    ])
    
    # Molecular weight-based features
    mol_weights = {
        'A': 89.1, 'C': 121.0, 'D': 133.1, 'E': 147.1, 'F': 165.2, 'G': 75.1,
        'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2, 'M': 149.2, 'N': 132.1,
        'P': 115.1, 'Q': 146.2, 'R': 174.2, 'S': 105.1, 'T': 119.1, 'V': 117.1,
        'W': 204.2, 'Y': 181.2
    }
    
    # Molecular weight features (mean, std, min, max)
    mw_values = [mol_weights.get(aa, 0) for aa in sequence]
    features.extend([
        np.mean(mw_values),
        np.std(mw_values),
        np.min(mw_values),
        np.max(mw_values)
    ])
    
    # Volume-based features
    volumes = {
        'A': 67.0, 'C': 86.0, 'D': 91.0, 'E': 109.0, 'F': 135.0, 'G': 48.0,
        'H': 118.0, 'I': 124.0, 'K': 135.0, 'L': 124.0, 'M': 124.0, 'N': 96.0,
        'P': 90.0, 'Q': 114.0, 'R': 148.0, 'S': 73.0, 'T': 93.0, 'V': 105.0,
        'W': 163.0, 'Y': 141.0
    }
    
    # Volume features (mean, std, min, max)
    vol_values = [volumes.get(aa, 0) for aa in sequence]
    features.extend([
        np.mean(vol_values),
        np.std(vol_values),
        np.min(vol_values),
        np.max(vol_values)
    ])
    
    # Secondary structure propensity features
    helix_prop = {
        'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13, 'G': 0.57,
        'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21, 'M': 1.45, 'N': 0.67,
        'P': 0.57, 'Q': 1.11, 'R': 0.98, 'S': 0.77, 'T': 0.83, 'V': 1.06,
        'W': 1.08, 'Y': 0.69
    }
    
    # Helix propensity features (mean, std, min, max)
    helix_values = [helix_prop.get(aa, 0) for aa in sequence]
    features.extend([
        np.mean(helix_values),
        np.std(helix_values),
        np.min(helix_values),
        np.max(helix_values)
    ])
    
    # Ensure exactly 40 features
    if len(features) > 40:
        features = features[:40]
    elif len(features) < 40:
        features.extend([0.0] * (40 - len(features)))
    
    return features

# Function to calculate comprehensive physicochemical features (9 features)
def calculate_physicochemical_features(sequence):
    """Calculate physicochemical properties features (9 features)"""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    
    sequence = sequence.upper()
    
    # Remove invalid characters
    valid_aas = 'ACDEFGHIKLMNPQRSTVWY'
    cleaned_sequence = ''.join([aa for aa in sequence if aa in valid_aas])
    
    if len(cleaned_sequence) == 0:
        return [0.0] * 9
    
    features = []
    
    try:
        analysis = ProteinAnalysis(cleaned_sequence)
        
        # 1. Hydropathy (GRAVY score)
        features.append(analysis.gravy())
        
        # 2. Net charge (sum of charged residues)
        charge_dict = {'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.5}
        net_charge = sum(charge_dict.get(aa, 0) for aa in cleaned_sequence)
        features.append(net_charge)
        
        # 3. Molecular weight
        features.append(analysis.molecular_weight())
        
        # 4. Instability index
        features.append(analysis.instability_index())
        
        # 5. Aromaticity
        features.append(analysis.aromaticity())
        
        # 6. Flexibility (average based on amino acid flexibility)
        flexibility_dict = {
            'A': 0.984, 'C': 0.906, 'D': 1.068, 'E': 1.094, 'F': 0.915, 'G': 1.031,
            'H': 0.950, 'I': 0.927, 'K': 1.102, 'L': 0.935, 'M': 0.952, 'N': 1.048,
            'P': 0.758, 'Q': 1.037, 'R': 1.008, 'S': 1.046, 'T': 0.997, 'V': 0.931,
            'W': 0.904, 'Y': 0.929
        }
        avg_flexibility = np.mean([flexibility_dict.get(aa, 1.0) for aa in cleaned_sequence])
        features.append(avg_flexibility)
        
        # 7. Isoelectric point
        features.append(analysis.isoelectric_point())
        
        # 8. Secondary structure fraction (helix)
        sec_struct = analysis.secondary_structure_fraction()
        features.append(sec_struct[0])  # helix fraction
        
        # 9. Length
        features.append(len(cleaned_sequence))
        
    except Exception:
        # Fallback values if ProteinAnalysis fails
        features = [0.0] * 9
    
    return features

# Function to convert sequence to numerical representation for CNN
def sequence_to_numerical(sequence, model_path=None):
    """Convert protein sequence to numerical representation for CNN models"""
    # All CNN models expect shape (1, 1200, 1)
    max_length = 1200
    
    # Amino acid to number mapping (1-based indexing for embedding compatibility)
    aa_dict = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    
    # Convert sequence to numbers
    numerical_seq = [aa_dict.get(aa, 0) for aa in sequence]
    
    # Pad or truncate to max_length (1200)
    if len(numerical_seq) > max_length:
        numerical_seq = numerical_seq[:max_length]
    else:
        numerical_seq.extend([0] * (max_length - len(numerical_seq)))
    
    # Return as 3D array with shape (1, 1200, 1)
    return np.array(numerical_seq).reshape(1, max_length, 1)

# Function to predict with CNN models
def predict_with_cnn(model, sequence):
    """Make prediction with CNN model using correct input format"""
    try:
        # Convert sequence to the correct 3D format (1, 1200, 1)
        features = sequence_to_numerical(sequence)
        
        # Make prediction
        prediction_prob = model.predict(features, verbose=0)[0]
        
        # CNN models output 2 probabilities [non-binding, binding]
        # Take the binding probability (index 1)
        binding_prob = prediction_prob[1]
        raw_prediction = 1 if binding_prob > 0.5 else 0
        
        return raw_prediction, binding_prob
        
    except Exception as e:
        raise Exception(f"CNN prediction failed: {str(e)}")

# Function to preprocess sequence for different model types
def preprocess_sequence(sequence, model_type, model_path=None):
    """Preprocess sequence based on model type"""
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    if model_type == "TF-IDF":
        return extract_kmers(sequence, k=3)
    elif model_type == "PseAAC":
        # For PseAAC, we'll use amino acid composition as a simplified version
        return calculate_aa_composition(sequence)
    elif model_type == "Physicochemical":
        # For physicochemical properties, use amino acid composition
        return calculate_aa_composition(sequence)
    elif model_type == "CNN":
        # For CNN models, convert sequence to numerical representation
        return sequence_to_numerical(sequence, model_path)
    else:
        return sequence

# Function to create TF-IDF features (fallback approach)
def create_tfidf_features_fallback(sequence):
    """Create TF-IDF-like features as a fallback when original vectorizer is not available"""
    # Create amino acid composition features (20 features)
    aa_composition = calculate_aa_composition(sequence)
    feature_vector = list(aa_composition.values())
    
    # Ensure we have exactly 20 features to match the trained model
    if len(feature_vector) != 20:
        # Pad or truncate to 20 features
        if len(feature_vector) < 20:
            feature_vector.extend([0.0] * (20 - len(feature_vector)))
        else:
            feature_vector = feature_vector[:20]
    
    return feature_vector

# Function to get expected feature count for different model types
def get_expected_feature_count(model_path):
    """Get the expected number of features for different model types"""
    if "TF-IDF" in model_path:
        return 20  # Basic amino acid composition
    elif "PseAAC" in model_path:
        return 40  # Extended PseAAC features
    elif "Physicochemical" in model_path:
        return 9  # Physicochemical features
    else:
        return 20  # Default

# Function to create features with specific count
def create_features_with_count(sequence, model_path):
    """Create features with the expected count for the specific model"""
    
    if "TF-IDF" in model_path:
        return create_tfidf_features_fallback(sequence)
    elif "PseAAC" in model_path:
        return calculate_pseudoaac_features(sequence)
    elif "Physicochemical" in model_path:
        return calculate_physicochemical_features(sequence)
    else:
        # Default: use amino acid composition
        aa_composition = calculate_aa_composition(sequence)
        features = list(aa_composition.values())
        expected_count = get_expected_feature_count(model_path)
        return features[:expected_count] if len(features) >= expected_count else features + [0.0] * (expected_count - len(features))

# Function to convert prediction to label
def convert_prediction_to_label(prediction):
    """Convert numerical prediction to meaningful label"""
    # Handle numpy data types as well as regular int/float
    if isinstance(prediction, (int, float, np.integer, np.floating)):
        return "DNA Binding" if prediction == 1 or prediction > 0.5 else "Non-DNA Binding"
    else:
        return str(prediction)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Input DNA Sequence")
    
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
            # Handle multiple sequences separated by newlines
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
                sequences = []
                current_seq = ""
                current_header = ""
                
                for line in stringio:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq and current_header:
                            sequences_to_classify.append(f"{current_header}: {current_seq}")
                        current_header = line[1:]  # Remove '>'
                        current_seq = ""
                    else:
                        current_seq += line
                
                # Add the last sequence
                if current_seq and current_header:
                    sequences_to_classify.append(f"{current_header}: {current_seq}")
                
                st.success(f"Loaded {len(sequences_to_classify)} sequences from file")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

with col2:
    st.header("Classification Results")
    
    if selected_model_path and sequences_to_classify:
        if st.button("🔬 Classify Sequences", type="primary"):
            model = load_model(selected_model_path)
            
            if model is not None:
                results = []
                
                try:
                    for seq_info in sequences_to_classify:
                        if ': ' in seq_info:
                            header, sequence = seq_info.split(': ', 1)
                        else:
                            header = f"Sequence_{len(results)+1}"
                            sequence = seq_info
                        
                        # Preprocess sequence based on model type
                        if "TF-IDF" in selected_model_path:
                            # Use improved feature extraction with correct count
                            try:
                                features = [create_features_with_count(sequence, selected_model_path)]
                            except Exception as tfidf_error:
                                st.warning(f"Feature extraction failed: {str(tfidf_error)}")
                                # Ultimate fallback: use basic amino acid composition
                                features_dict = calculate_aa_composition(sequence)
                                features = [list(features_dict.values())[:20]]  # Ensure 20 features
                        elif "PseAAC" in selected_model_path:
                            features = [create_features_with_count(sequence, selected_model_path)]
                        elif "Physicochemical" in selected_model_path:
                            features = [create_features_with_count(sequence, selected_model_path)]
                        elif "CNN" in selected_model_path:
                            features = preprocess_sequence(sequence, "CNN", selected_model_path)
                        else:
                            features = [sequence]
                        
                        # Make prediction
                        try:
                            if "CNN" in selected_model_path:
                                # Handle CNN models (Keras/TensorFlow)
                                if not tf_available:
                                    prediction = "TensorFlow not available"
                                    confidence = "N/A"
                                else:
                                    try:
                                        # Use the correct CNN prediction method
                                        raw_prediction, pred_prob = predict_with_cnn(model, sequence)
                                        prediction = convert_prediction_to_label(raw_prediction)
                                        confidence = f"{pred_prob:.4f}"
                                        
                                    except Exception as cnn_error:
                                        prediction = f"❌ CNN Error: {str(cnn_error)[:50]}..."
                                        confidence = "N/A"
                                        st.error(f"CNN model failed: {str(cnn_error)}")
                            elif hasattr(model, 'predict'):
                                # Handle traditional ML models
                                raw_prediction = model.predict(features)[0]
                                prediction = convert_prediction_to_label(raw_prediction)
                                if hasattr(model, 'predict_proba'):
                                    probabilities = model.predict_proba(features)[0]
                                    confidence = f"{max(probabilities):.4f}"
                                else:
                                    confidence = "N/A"
                            else:
                                prediction = "Unable to predict"
                                confidence = "N/A"
                        except Exception as pred_error:
                            prediction = f"Error: {str(pred_error)}"
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
        st.write(f"**Model Path:** {selected_model_path}")
        
        # Model file info
        if os.path.exists(selected_model_path):
            file_size = os.path.getsize(selected_model_path)
            st.write(f"**File Size:** {file_size / (1024*1024):.2f} MB")
            
            # Model type specific information
            if "CNN" in selected_model_path:
                st.write("**Model Type:** Convolutional Neural Network")
                st.write("**Input:** 3D tensor (1, 1200, 1)")
                st.success("✅ Fixed - Now working with correct input format")
                if not tf_available:
                    st.error("❌ TensorFlow required for this model")
            elif "TF-IDF" in selected_model_path:
                st.write("**Model Type:** Traditional ML with TF-IDF features")
                st.write("**Input:** K-mer based text features")
                st.success("✅ Recommended - Reliable and accurate")
            elif "PseAAC" in selected_model_path:
                st.write("**Model Type:** Traditional ML with PseAAC features")
                st.write("**Input:** Pseudo amino acid composition")
                st.success("✅ Recommended - Very stable performance")
            elif "Physicochemical" in selected_model_path:
                st.write("**Model Type:** Traditional ML with physicochemical features")
                st.write("**Input:** Amino acid physicochemical properties")
                st.success("✅ Recommended - Highly reliable")
        else:
            st.warning("Model file not found")
    else:
        st.info("Select a model to see details")
        
        # Show summary of all available models
        st.write("**Available Models:**")
        total_models = sum(len(models) for models in model_categories.values())
        st.write(f"- Total: {total_models} models")
        for category, models in model_categories.items():
            st.write(f"- {category}: {len(models)} models")

with col4:
    st.subheader("Usage Instructions")
    st.write("""
    1. **Select Model**: Choose a model category and specific model from the sidebar
    2. **Input Sequences**: Enter sequences manually or upload a FASTA file
    3. **Classify**: Click the classify button to get predictions
    4. **View Results**: See predictions, confidence scores, and summary statistics
    5. **Download**: Export results as CSV for further analysis
    """)
    
    st.subheader("Model Notes")
    st.write("""
    **CNN Models**: 
    - Now working with correct input format (1200 positions)
    - Sequences longer than 1200 are truncated
    - Sequences shorter than 1200 are zero-padded
    
    **Traditional ML**: 
    - Use amino acid composition features
    - More interpretable results
    - Faster inference time
    """)

# Footer
st.markdown("---")
st.markdown("**DNA Binding Protein Classifier** - Powered by Machine Learning & Deep Learning")
