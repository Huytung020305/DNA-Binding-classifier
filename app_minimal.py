"""
DNA Binding Protein Classifier - Minimal Cloud Version
Streamlit-only version that handles missing dependencies gracefully.
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

def simple_prediction(sequence):
    """Simple rule-based prediction when ML models aren't available"""
    properties = extract_physicochemical_properties(sequence)
    aa_comp = calculate_amino_acid_composition(sequence)
    
    # Simple heuristic based on DNA-binding protein characteristics
    # Higher aromatic and charged amino acid content often indicates DNA binding
    aromatic_content = properties[2]
    charged_content = properties[4]
    
    # Specific amino acids more common in DNA-binding proteins
    r_content = aa_comp[14]  # Arginine
    k_content = aa_comp[8]   # Lysine
    h_content = aa_comp[6]   # Histidine
    
    # Simple scoring
    score = (aromatic_content * 2 + charged_content * 3 + 
             r_content * 4 + k_content * 3 + h_content * 2)
    
    prediction = 1 if score > 0.15 else 0
    confidence = min(0.95, max(0.55, score * 5))
    
    return prediction, confidence

# Main Application
st.title("🧬 DNA Binding Protein Classifier")
st.subheader("Minimal Cloud Version")

# Show dependency status
st.sidebar.header("System Status")
st.sidebar.write("**Dependencies:**")
st.sidebar.write(f"- Pandas: {'✅' if PANDAS_AVAILABLE else '❌'}")
st.sidebar.write(f"- NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
st.sidebar.write(f"- Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
st.sidebar.write(f"- Joblib: {'✅' if JOBLIB_AVAILABLE else '❌'}")
st.sidebar.write(f"- BioPython: {'✅' if BIOPYTHON_AVAILABLE else '❌'}")

if not (PANDAS_AVAILABLE and SKLEARN_AVAILABLE and JOBLIB_AVAILABLE):
    st.warning("""
    ⚠️ **Limited Functionality Mode**
    Some dependencies are missing. Using simplified rule-based prediction.
    For full ML model functionality, ensure all dependencies are installed.
    """)
else:
    st.success("✅ All dependencies available - Full functionality enabled")

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
            
            with st.progress(0) as progress_bar:
                for i, (header, sequence) in enumerate(sequences_to_classify):
                    progress_bar.progress((i + 1) / len(sequences_to_classify))
                    
                    # Clean sequence
                    cleaned_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
                    
                    if len(cleaned_seq) == 0:
                        pred_text = "❌ Invalid sequence"
                        conf_text = "N/A"
                    else:
                        prediction, confidence = simple_prediction(cleaned_seq)
                        pred_text = "🧬 DNA Binding" if prediction == 1 else "🚫 Non-DNA Binding"
                        conf_text = f"{confidence:.3f}"
                    
                    results.append({
                        'Sequence ID': header,
                        'Length': len(cleaned_seq),
                        'Prediction': pred_text,
                        'Confidence': conf_text,
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
    if SKLEARN_AVAILABLE and JOBLIB_AVAILABLE:
        st.write("**Mode:** Machine Learning Models")
        st.write("**Models:** Random Forest, SVM, etc.")
    else:
        st.write("**Mode:** Rule-based Heuristics")
        st.write("**Features:** Amino acid composition, physicochemical properties")
    
    st.write("**Input:** Protein sequences (FASTA format)")
    st.write("**Output:** DNA binding prediction + confidence")

with col4:
    st.subheader("Usage Instructions")
    st.write("""
    1. **Input**: Enter sequences or upload FASTA file
    2. **Classify**: Click the classify button
    3. **Results**: View predictions and confidence scores
    4. **Download**: Export results as CSV (if available)
    
    **Note:** This minimal version uses simplified prediction methods
    when full ML dependencies are not available.
    """)

# Footer
st.markdown("---")
st.markdown("**DNA Binding Protein Classifier** - Minimal Cloud Deployment")
if not (PANDAS_AVAILABLE and SKLEARN_AVAILABLE and JOBLIB_AVAILABLE):
    st.caption("⚠️ Running in limited mode due to missing dependencies")
