"""
DNA Binding Protein Classifier
Full-featured machine learning version with trained models.
"""

import streamlit as st
import warnings
import re
import os
from io import StringIO

# Suppress warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="DNA Binding Protein Classifier",
    page_icon="🧬",
    layout="wide"
)

# Try to import required packages
PANDAS_AVAILABLE = False
NUMPY_AVAILABLE = False
SKLEARN_AVAILABLE = False
JOBLIB_AVAILABLE = False
TENSORFLOW_AVAILABLE = False
BIOPYTHON_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    pass

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    pass

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

def safe_load_model(model_path):
    """Safely load model with error handling"""
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
                return None, "TensorFlow not available - CNN models not supported"
        else:
            if JOBLIB_AVAILABLE:
                import joblib
                model = joblib.load(model_path)
                return model, None
            else:
                return None, "Joblib not available - Traditional ML models not supported"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def make_prediction_with_model(model, sequence, model_type):
    """Make prediction using loaded ML model"""
    try:
        sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
        
        if 'CNN' in model_type:
            # CNN prediction logic
            features = sequence_to_cnn_input(sequence)
            if features is None:
                return None, "NumPy not available for CNN prediction"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction_proba = model.predict(features, verbose=0)[0][0]
                prediction = 1 if prediction_proba > 0.5 else 0
                confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
        else:
            # Traditional ML prediction
            if 'PseAAC' in model_type or 'pseAAC' in model_type:
                features = calculate_pseaac_features(sequence)
                # For PseAAC, use array input directly since model expects 40 features but only has 36 feature names
                prediction = model.predict([features])[0]
                
                confidence = None
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba([features])[0]
                        confidence = max(proba)
                    except:
                        pass
            elif 'Physicochemical' in model_type:
                features = extract_physicochemical_properties(sequence)
                feature_names = ['Hydrophobic', 'Hydrophilic', 'Aromatic', 'Aliphatic', 'Charged', 'Polar']
                
                if PANDAS_AVAILABLE:
                    import pandas as pd
                    feature_df = pd.DataFrame([features], columns=feature_names)
                    prediction = model.predict(feature_df)[0]
                    
                    confidence = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(feature_df)[0]
                            confidence = max(proba)
                        except:
                            pass
                else:
                    # Fallback without pandas
                    prediction = model.predict([features])[0]
                    confidence = None
            else:
                # TF-IDF and other models
                features = calculate_amino_acid_composition(sequence)
                amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
                feature_names = [f'AA_{aa}' for aa in amino_acids]
                
                if PANDAS_AVAILABLE:
                    import pandas as pd
                    feature_df = pd.DataFrame([features], columns=feature_names)
                    prediction = model.predict(feature_df)[0]
                    
                    confidence = None
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(feature_df)[0]
                            confidence = max(proba)
                        except:
                            pass
                else:
                    # Fallback without pandas
                    prediction = model.predict([features])[0]
                    confidence = None
        
        return prediction, confidence
    except Exception as e:
        return None, f"Model prediction error: {str(e)}"

def calculate_pseaac_features(sequence, lamda=4, weight=0.05):
    """Calculate PseAAC features matching the trained model's expected format (40 features)"""
    import numpy as np
    
    # Clean sequence - remove any invalid amino acids
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    if len(sequence) == 0:
        return [0.0] * 40  # Return 40 zeros for empty sequence
    
    # 1. Calculate amino acid composition (20 features: AA_A, AA_C, etc.)
    aa_composition = calculate_amino_acid_composition(sequence)
    
    # 2. Calculate physicochemical property statistics (16 features)
    # Properties from the original propy implementation
    hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                     'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                     'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                     'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
    
    # Molecular weight (approximate)
    molecular_weight = {'A': 89.1, 'C': 121.0, 'D': 133.1, 'E': 147.1, 'F': 165.2,
                       'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
                       'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
                       'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2}
    
    # Volume (approximate)
    volume = {'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
             'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
             'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
             'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6}
    
    # Helix propensity (approximate)
    helix_prop = {'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
                 'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
                 'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
                 'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69}
    
    # Calculate statistics for each property
    properties = [hydrophobicity, molecular_weight, volume, helix_prop]
    
    property_features = []
    
    for prop_dict in properties:
        # Get property values for the sequence
        prop_values = [prop_dict.get(aa, 0) for aa in sequence]
        
        if prop_values:
            # Calculate mean, std, min, max
            prop_array = np.array(prop_values)
            property_features.extend([
                np.mean(prop_array),  # mean
                np.std(prop_array),   # std
                np.min(prop_array),   # min
                np.max(prop_array)    # max
            ])
        else:
            # If no values, add zeros
            property_features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 3. Calculate correlation factors (4 additional lambda features)
    correlation_factors = []
    for lag in range(1, lamda + 1):
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
    
    # Ensure we have exactly 4 correlation factors
    while len(correlation_factors) < lamda:
        correlation_factors.append(0.0)
    
    # Combine all features: 20 AA composition + 16 property statistics + 4 correlation = 40 total
    all_features = aa_composition + property_features + correlation_factors[:lamda]
    
    return all_features

def sequence_to_cnn_input(sequence, max_length=1200):
    """Convert protein sequence to CNN input format"""
    if not NUMPY_AVAILABLE:
        return None
    
    import numpy as np
    
    aa_to_num = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
        'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    
    sequence = sequence.upper()
    sequence_nums = [aa_to_num.get(aa, 0) for aa in sequence]
    
    # Pad or truncate to max_length (1200)
    if len(sequence_nums) > max_length:
        sequence_nums = sequence_nums[:max_length]
    else:
        sequence_nums.extend([0] * (max_length - len(sequence_nums)))
    
    # Reshape to (1, 1200, 1) for CNN input
    return np.array(sequence_nums).reshape(1, max_length, 1)



# Main Application
st.title("🧬 DNA Binding Protein Classifier")
st.subheader("Machine Learning Powered Classification")

# Show dependency status
# Model selection
st.sidebar.header("Model Selection")

model_categories = {
    "CNN Models": {
        "CNN1": "models/CNN/CNN1.h5",
        "CNN2": "models/CNN/CNN2.h5",
        "ProtCNN1": "models/CNN/ProtCNN1.h5",
        "ProtCNN2": "models/CNN/ProtCNN2.h5"
    },
    "Traditional ML - TF-IDF": {
        "Logistic Regression": "models/Traditional ML - TF-IDF/Logistic Regression_TF-IDF.joblib",
        "SVM": "models/Traditional ML - TF-IDF/SVM_TF-IDF.joblib",
        "Random Forest": "models/Traditional ML - TF-IDF/Random Forest_TF-IDF.joblib",
        "Naive Bayes": "models/Traditional ML - TF-IDF/Naive Bayes_TF-IDF.joblib",
        "Decision Tree": "models/Traditional ML - TF-IDF/Decision Tree_TF-IDF.joblib",
        "KNN": "models/Traditional ML - TF-IDF/KNN_TF-IDF.joblib"
    },
    "Traditional ML - PseAAC": {
        "Logistic Regression": "models/Traditional ML - PseAAC/LR_pseAAC.joblib",
        "SVM": "models/Traditional ML - PseAAC/SVM_pseAAC.joblib",
        "Random Forest": "models/Traditional ML - PseAAC/RF_pseAAC.joblib",
        "Naive Bayes": "models/Traditional ML - PseAAC/NB_pseAAC.joblib",
        "Decision Tree": "models/Traditional ML - PseAAC/DT_pseAAC.joblib",
        "KNN": "models/Traditional ML - PseAAC/KNN_pseAAC.joblib"
    },
    "Traditional ML - Physicochemical": {
        "Logistic Regression": "models/Traditional ML - Physicochemical Properties/LR_Physicochemical_Properties.joblib",
        "SVM": "models/Traditional ML - Physicochemical Properties/SVM_Physicochemical_Properties.joblib",
        "Random Forest": "models/Traditional ML - Physicochemical Properties/RF_Physicochemical_Properties.joblib",
        "Naive Bayes": "models/Traditional ML - Physicochemical Properties/NB_Physicochemical_Properties.joblib",
        "Decision Tree": "models/Traditional ML - Physicochemical Properties/DT_Physicochemical_Properties.joblib",
        "KNN": "models/Traditional ML - Physicochemical Properties/KNN_Physicochemical_Properties.joblib"
    }
}

# Check if required ML dependencies are available
if not (TENSORFLOW_AVAILABLE or JOBLIB_AVAILABLE):
    st.error("❌ Required ML libraries not available. Please install tensorflow and/or scikit-learn.")
    st.stop()

# Model selection
selected_category = st.sidebar.selectbox("Select Model Category", list(model_categories.keys()))
selected_model_name = st.sidebar.selectbox("Select Model", list(model_categories[selected_category].keys()))
selected_model_path = model_categories[selected_category][selected_model_name]
selected_model_type = f"{selected_model_name} ({selected_category})"

# Display selected model info
st.sidebar.info(f"**Selected Model:**\n{selected_model_type}")

# Check dependencies status
missing_deps = []
if not PANDAS_AVAILABLE:
    missing_deps.append("pandas")
if not NUMPY_AVAILABLE:
    missing_deps.append("numpy")
if not SKLEARN_AVAILABLE:
    missing_deps.append("scikit-learn")
if not JOBLIB_AVAILABLE:
    missing_deps.append("joblib")
if not TENSORFLOW_AVAILABLE and "CNN" in selected_category:
    missing_deps.append("tensorflow")

if missing_deps:
    st.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
else:
    st.success("✅ All required dependencies available")

st.info("""
🎯 **DNA Binding Protein Classification**
This tool predicts whether a protein sequence binds to DNA based on its amino acid composition and physicochemical properties.
""")

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
            results = []
            
            # Load model
            with st.spinner("Loading model..."):
                model, model_error = safe_load_model(selected_model_path)
                
            if model_error:
                st.error(f"Model loading failed: {model_error}")
                st.stop()
            
            with st.spinner("Classifying sequences..."):
                progress_bar = st.progress(0)
                for i, (header, sequence) in enumerate(sequences_to_classify):
                    progress_bar.progress((i + 1) / len(sequences_to_classify))
                    
                    # Clean sequence
                    cleaned_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
                    
                    if len(cleaned_seq) == 0:
                        pred_text = "❌ Invalid sequence"
                        conf_text = "N/A"
                        method = "N/A"
                    else:
                        prediction, confidence = make_prediction_with_model(model, cleaned_seq, selected_model_type)
                        method = selected_model_type
                        
                        if prediction is None:
                            pred_text = f"❌ Prediction failed: {confidence}"
                            conf_text = "N/A"
                        else:
                            pred_text = "🧬 DNA Binding" if prediction == 1 else "🚫 Non-DNA Binding"
                            conf_text = f"{confidence:.3f}" if confidence is not None else "N/A"
                    
                    results.append({
                        'Sequence ID': header,
                        'Length': len(cleaned_seq),
                        'Prediction': pred_text,
                        'Confidence': conf_text,
                        'Method': method,
                        'Sequence Preview': cleaned_seq[:50] + "..." if len(cleaned_seq) > 50 else cleaned_seq
                    })
            
            # Display results
            if results and PANDAS_AVAILABLE:
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
                    file_name="dna_classification_results.csv",
                    mime="text/csv"
                )
            elif results:
                # Display without pandas
                st.subheader("Results")
                for result in results:
                    st.write(f"**{result['Sequence ID']}**: {result['Prediction']} (Confidence: {result['Confidence']})")
            
    else:
        st.info("Please enter or upload sequences to classify")

# Information section
st.markdown("---")
st.header("ℹ️ About This Tool")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Method")
    st.write("**Mode:** Machine Learning Models")
    st.write("**Models:** CNN, SVM, Random Forest, etc.")
    st.write("**Features:** Amino acid composition, physicochemical properties, PseAAC")
    st.write("**Input:** Protein sequences (FASTA format)")
    st.write("**Output:** DNA binding prediction + confidence")

with col4:
    st.subheader("Usage Instructions")
    st.write("""
    1. **Select Model**: Choose a model category and specific model
    2. **Input**: Enter sequences or upload FASTA file
    3. **Classify**: Click the classify button
    4. **Results**: View predictions and confidence scores
    5. **Download**: Export results as CSV
    
    **Note:** Requires ML dependencies (tensorflow, scikit-learn, etc.)
    """)

# Footer
st.markdown("---")
st.markdown("**DNA Binding Protein Classifier** - Machine Learning Powered")
st.caption("🧬 Using trained ML models for accurate DNA binding prediction")
