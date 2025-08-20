"""
Feature extraction utilities for protein sequence analysis.
Handles amino acid composition, TF-IDF, PseAAC, and physicochemical properties.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

def calculate_amino_acid_composition(sequence):
    """Calculate amino acid composition features for a protein sequence"""
    # Standard amino acids
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Clean sequence (remove any non-amino acid characters)
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    if len(sequence) == 0:
        return [0.0] * len(amino_acids)
    
    # Count occurrences
    aa_count = Counter(sequence)
    
    # Calculate composition (percentage)
    composition = []
    for aa in amino_acids:
        composition.append(aa_count.get(aa, 0) / len(sequence))
    
    return composition

def extract_physicochemical_properties(sequence):
    """Extract physicochemical properties from protein sequence"""
    # Amino acid property groups
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

def calculate_tfidf_features(sequences, max_features=100):
    """Calculate TF-IDF features for protein sequences"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Convert sequences to k-mer representation (k=3)
    def sequence_to_kmers(seq, k=3):
        return ' '.join([seq[i:i+k] for i in range(len(seq)-k+1)])
    
    kmer_sequences = [sequence_to_kmers(seq) for seq in sequences]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(kmer_sequences)
    
    return tfidf_matrix.toarray(), vectorizer

def calculate_pseaac_features(sequence, lambda_val=10):
    """Calculate Pseudo Amino Acid Composition (PseAAC) features"""
    # Simplified PseAAC implementation
    aa_composition = calculate_amino_acid_composition(sequence)
    
    # Add sequence order correlation factors (simplified)
    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    correlation_factors = []
    for lag in range(1, min(lambda_val + 1, len(sequence))):
        if len(sequence) > lag:
            # Simple correlation based on hydrophobicity difference
            hydrophobicity = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
            
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
    
    # Pad with zeros if needed
    while len(correlation_factors) < lambda_val:
        correlation_factors.append(0.0)
    
    return aa_composition + correlation_factors[:lambda_val]

def get_feature_names(feature_type, max_features=None):
    """Get feature names for different feature extraction methods"""
    if feature_type == 'amino_acid':
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        return [f'AA_{aa}' for aa in amino_acids]
    
    elif feature_type == 'physicochemical':
        return ['Hydrophobic', 'Hydrophilic', 'Aromatic', 'Aliphatic', 'Charged', 'Polar']
    
    elif feature_type == 'tfidf':
        if max_features:
            return [f'TFIDF_{i}' for i in range(max_features)]
        return [f'TFIDF_{i}' for i in range(100)]  # Default
    
    elif feature_type == 'pseaac':
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        aa_features = [f'AA_{aa}' for aa in amino_acids]
        lambda_features = [f'Lambda_{i+1}' for i in range(10)]
        return aa_features + lambda_features
    
    else:
        return []

def process_sequence_for_prediction(sequence, feature_type):
    """Process a single sequence for prediction based on feature type"""
    if feature_type == 'amino_acid':
        return calculate_amino_acid_composition(sequence)
    elif feature_type == 'physicochemical':
        return extract_physicochemical_properties(sequence)
    elif feature_type == 'pseaac':
        return calculate_pseaac_features(sequence)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
